import functools
import math
from collections.abc import Callable, Iterable
from itertools import batched, chain
from typing import Literal, NamedTuple

import jax
import jax.numpy as jnp

from lalamo.data.lalamo_completions import LalamoCompletion
from lalamo.data.utils import get_prefixes_ending_in_user_message
from lalamo.message_processor import Message
from lalamo.models import LanguageModel


def build_bounds(max_len: int) -> list[int]:
    bounds: list[int] = []
    true_bound = 16.0
    while True:
        bound = round(true_bound)
        if bound > max_len:
            break
        bounds.append(bound)
        true_bound *= math.sqrt(2.0)
    if bounds[-1] != max_len:
        bounds.append(max_len)
    return bounds


class CollectTracesEvent(NamedTuple):
    sequences_processed: int
    tokens_generated: int


def inference_collect_traces(
    model: LanguageModel,
    conversations: Iterable[Iterable[Message]],
    num_top_logits_to_collect: int = 8,
    batch_size: int = 1,
    max_input_length: int = 1024,
    max_output_length: int = 1024,
    tokens_to_generate: int | None = None,
    progress_callback: Callable[[CollectTracesEvent], None] | None = None,
    kind: Literal["adaptive", "standard"] = "standard",
) -> Iterable[LalamoCompletion]:
    # kind corresponds to the way we recompile the forward pass to improve performance
    # 'standard': the default, fix batch size, fix max input length, call generate
    # 'adaptive': does not provide much benefit, and sometimes harms performance;
    #             we split the prefixes into chunks (N * batch_size, for some fixed N)
    #             such that they have a common, small max_input_length, which enables
    #             significantly larger batch sizes (up to x6 for common workloads),
    #             but since using different batch sizes would violate "use batch_size=K"
    #             assumption, we don't do that (yet). So this is just a, kind of, draft.
    prefixes = chain.from_iterable(map(get_prefixes_ending_in_user_message, conversations))

    tokenized_prefixes = map(model.message_processor.tokenize_request, prefixes)
    filtered_prefixes = [conv for conv in tokenized_prefixes if len(conv) <= max_input_length]
    filtered_prefixes.sort(key=len)
    prefix_lengths = [len(prefix) for prefix in filtered_prefixes]

    tokens_generated, sequences_processed = 0, 0

    compiled_cache: dict[int, Callable] = {}

    def get_compiled(bound: int) -> Callable:
        compiled = compiled_cache.get(bound)
        if compiled is None:
            # the compilation with non-default, non-zero autotune emits lots of logs
            # i didn't find a better way to hide them
            compiled = (
                jax.jit(
                    functools.partial(
                        LanguageModel.generate_tokens,
                        max_output_length=max_output_length,
                        num_top_logits_to_return=num_top_logits_to_collect,
                    ),
                )
                .lower(
                    model,
                    prompt_token_ids=jax.ShapeDtypeStruct((batch_size, bound), jnp.int32),
                    prompt_lengths_without_padding=jax.ShapeDtypeStruct((batch_size,), jnp.int32),
                )
                # the autotune levels are (according to https://guides.lw1.at/all-xla-options/#--xla_gpu_autotune_level)
                # 0 - no autotune, gpu shouldn't be touched
                # 1 - basic level, gpu should be touched veeery little
                # 2,3 - gpu touched more and more
                # 4 (default) - gpu might allocate more memory than the run would require!
                .compile(compiler_options={"xla_gpu_autotune_level": 2})
            )
            compiled_cache[bound] = compiled
        return compiled

    if kind == "adaptive":
        # splits up the prefixes into blocks
        bounds = build_bounds(max_input_length)
        chunks: list[tuple[list[list[int]], int]] = []
        prefix_idx = 0
        limit = 16 * batch_size  # magic number, each block must include at least `limit` prefixes
        while prefix_idx < len(filtered_prefixes):
            chosen_bound = None
            end = prefix_idx
            count = 0
            for bound in bounds:
                while end < len(filtered_prefixes) and prefix_lengths[end] <= bound and count < limit:
                    end += 1
                    count += 1
                if count >= limit:
                    chosen_bound = bound
                    block_end = prefix_idx + limit
                    break

            if chosen_bound is None:
                chosen_bound = bounds[-1] if bounds else max_input_length
                block_end = end

            chunks.append((filtered_prefixes[prefix_idx:block_end], chosen_bound))
            prefix_idx = block_end
    else:
        chunks = [(filtered_prefixes, max_input_length)]

    for block, chosen_bound in chunks:
        generate_tokens_compiled = get_compiled(chosen_bound)

        for real_batch in batched(block, n=batch_size):
            batch_padding = batch_size - len(real_batch)
            batch = (*real_batch, *(([0],) * batch_padding))

            length_without_padding = jnp.array(list(map(len, batch)))

            padded = jnp.array(
                [jnp.pad(jnp.array(tokens), (0, chosen_bound - len(tokens)), constant_values=0) for tokens in batch],
            )

            generated = generate_tokens_compiled(
                model,
                prompt_token_ids=padded,
                prompt_lengths_without_padding=length_without_padding,
            )

            assert generated.top_k_token_ids is not None and generated.top_k_token_logits is not None

            for conv_idx in range(len(real_batch)):
                token_ids = generated.token_ids[conv_idx].tolist()
                seqlen = next((i + 1 for i, t in enumerate(token_ids) if t in model.stop_token_ids), len(token_ids))
                if tokens_to_generate is not None:
                    seqlen = min(seqlen, tokens_to_generate - tokens_generated)
                tokens_generated += seqlen
                sequences_processed += 1

                token_ids = token_ids[:seqlen]
                token_logits_ids = generated.top_k_token_ids[conv_idx, : len(token_ids)].tolist()
                token_logits_values = generated.top_k_token_logits[conv_idx, : len(token_ids)].tolist()
                token_logits = [
                    dict(zip(keys, values, strict=True))
                    for keys, values in zip(token_logits_ids, token_logits_values, strict=True)
                ]

                yield LalamoCompletion(batch[conv_idx], token_ids, token_logits)

                if tokens_to_generate is not None and tokens_generated >= tokens_to_generate:
                    break

            if progress_callback is not None:
                progress_callback(CollectTracesEvent(sequences_processed, tokens_generated))

            if tokens_to_generate is not None and tokens_generated >= tokens_to_generate:
                break

        if tokens_to_generate is not None and tokens_generated >= tokens_to_generate:
            break
