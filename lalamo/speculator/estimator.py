import functools
import itertools
import os
from pathlib import Path
from collections.abc import Callable
from contextlib import contextmanager
from typing import NamedTuple, Literal

import equinox as eqx
import jax
import jax.numpy as jnp

from lalamo.message_processor import UserMessage
from lalamo.model_import import REPO_TO_MODEL, import_model
from lalamo.models import LanguageModel
from lalamo.models.language_model import LanguageModelConfig
from lalamo.speculator.peak_mem_sampler import PeakMemSampler

INT_INF = 10**30


def get_default_device_memory() -> int | None:
    memory_stats = jax.local_devices()[0].memory_stats()
    if memory_stats is None or "bytes_limit" not in memory_stats:
        return None
    return memory_stats["bytes_limit"]


def estimate_memory_from_batchsize(
    batch_size: int,
    *,
    model: LanguageModel | None,
    max_input_length: int,
    max_output_length: int,
    num_logits_per_token: int,
    kind: Literal["cpu", "on_device", "hybrid"] = "on_device",
) -> int | None:
    if kind == "cpu":
        memory_analysis = (
            jax.jit(
                functools.partial(
                    LanguageModel.generate_tokens,
                    max_output_length=max_output_length,
                    num_top_logits_to_return=num_logits_per_token,
                ),
                backend="cpu",  # cuda backend tries to allocate in .compile() and ooms
            )
            .lower(
                model,
                prompt_token_ids=jax.ShapeDtypeStruct((batch_size, max_input_length), jnp.int32),
                prompt_lengths_without_padding=jax.ShapeDtypeStruct((batch_size,), jnp.int32),
            )
            .compile()
            .memory_analysis()
        )

        assert hasattr(memory_analysis, "argument_size_in_bytes")
        assert hasattr(memory_analysis, "output_size_in_bytes")
        assert hasattr(memory_analysis, "temp_size_in_bytes")

        return (
            memory_analysis.argument_size_in_bytes
            + memory_analysis.output_size_in_bytes
            + memory_analysis.temp_size_in_bytes
        )
    elif kind == "on_device":
        device = jax.devices()[0]

        start_stats = device.memory_stats()
        start_current = None
        if start_stats is not None:
            start_current = start_stats.get("bytes_in_use") or start_stats.get("device_memory_in_use")

        with _suppress_stdio():
            try:
                with PeakMemSampler(device=device) as sampler:
                    with jax.default_device(device):
                        prompt = [UserMessage("Count from 1 to 20 separated by spaces.")]
                        prompt_token_ids = jnp.array(
                            model.message_processor.tokenize_request(prompt),
                            dtype=jnp.int32,
                        )
                        max_input_length_local = max(max_input_length, prompt_token_ids.size)

                        prompt_len = prompt_token_ids.size
                        if prompt_len > max_input_length_local:
                            raise ValueError("prompt exceeds max_input_length")

                        prompt_token_ids = jax.device_put(prompt_token_ids, device)
                        padded = jnp.pad(
                            prompt_token_ids,
                            (0, max_input_length_local - prompt_len),
                            constant_values=0,
                        )
                        batch_prompt_ids = jnp.repeat(padded[None, :], batch_size, axis=0)
                        batch_prompt_lengths = jnp.array([prompt_len] * batch_size, dtype=jnp.int32)

                        results = eqx.filter_jit(model.generate_tokens)(
                            batch_prompt_ids,
                            prompt_lengths_without_padding=batch_prompt_lengths,
                            max_output_length=max_output_length,
                            num_top_logits_to_return=num_logits_per_token,
                        )
                    jax.block_until_ready(results.token_ids)
            except Exception:
                return INT_INF

        peak_bytes = sampler.peak or 0

        del model
        del results
        del batch_prompt_ids
        del batch_prompt_lengths
        jax.clear_caches()

        if peak_bytes == 0 and start_current is None:
            raise RuntimeError("device.memory_stats missing bytes_in_use/device_memory_in_use")

        if start_current is None:
            return int(peak_bytes) if peak_bytes else None
    return int(max(0, peak_bytes - start_current))


@contextmanager
def _suppress_stdio():
    null_fds = [os.open(os.devnull, os.O_RDWR) for _ in range(2)]
    save_fds = [os.dup(1), os.dup(2)]
    try:
        os.dup2(null_fds[0], 1)
        os.dup2(null_fds[1], 2)
        yield
    finally:
        os.dup2(save_fds[0], 1)
        os.dup2(save_fds[1], 2)
        for fd in null_fds + save_fds:
            os.close(fd)


class EstimateBatchsizeFromMemoryEvent(NamedTuple):
    lo: int
    hi: int | None


def estimate_batchsize_from_memory(
    model: str,
    max_input_length: int,
    max_output_length: int,
    num_logits_per_token: int,
    target_mem: int,
    progress: Callable[[EstimateBatchsizeFromMemoryEvent], None] | None = None,
    *,
    kind: Literal["cpu", "hybrid"] = "cpu",
) -> int:
    fn = functools.partial(
        estimate_memory_from_batchsize,
        model=model,
        max_input_length=max_input_length,
        max_output_length=max_output_length,
        num_logits_per_token=num_logits_per_token,
        kind="cpu",
    )

    if kind == "hybrid":
        safety_factor = 0.95  # the closer to 1, the bigger the batch size

        cpu_limit = estimate_batchsize_from_memory(
            model=model,
            max_input_length=max_input_length,
            max_output_length=max_output_length,
            num_logits_per_token=num_logits_per_token,
            target_mem=target_mem,
            progress=progress,
            kind="cpu",
        )
        print(f"[hybrid] cpu_limit={cpu_limit}")
        # 1.5 is a magic number: it's memory spent on something, and I don't know what
        # when calling stuff with any batch size on the first run it spends 1/2 of the
        # model size on _something_ - but on the future runs it doesn't, probably because
        # this computation is cached. I'm not sure what it is, so for now just use this
        # hacky solution
        zero_limit = int(fn(batch_size=0) * 1.5)
        print(f"[hybrid] zero limit is {zero_limit / (2**30)}")

        # candidate batches are chosen as 1,2 and some smallish number
        # to make sure that we don't cause an OOM, and that we have points
        # far enough to capture (at least somewhat) the nonlinearities
        candidate_batches = [
            1,
            max(cpu_limit // 6, 2),
            max(cpu_limit // 3, 3),
        ]

        points: list[tuple[int, int]] = []
        for batch_size in candidate_batches:
            mem = estimate_memory_from_batchsize(
                batch_size,
                model=model,
                max_input_length=max_input_length,
                max_output_length=max_output_length,
                num_logits_per_token=num_logits_per_token,
                kind="on_device",
            )
            print(f"[hybrid] batch_size={batch_size} mem={mem / (2**30):.2f}")
            if mem is None or mem == INT_INF:
                # we might sometimes hit an oom if someone calls estimate_memory_from_batchsize with very little
                # memory left. While it's unlikely that 1/5th of the memory won't be free, there is still a chance
                # Then, we want to be on the safer side, and return a small number
                return max(1, min(batch_size // 2 - 1, cpu_limit // 2 - 1))
            points.append((batch_size, mem))

        # fit the curve with mse
        xs = [float(x) for x, _ in points]
        ys = [float(y) for _, y in points]
        mean_x = sum(xs) / len(xs)
        mean_y = sum(ys) / len(ys)
        var_x = sum((x - mean_x) ** 2 for x in xs)
        if var_x == 0:  # if they are all on the same line - weird, return cpu based limit
            return max(0, int(cpu_limit * safety_factor / 2))  # / 2 to be on the safer side
        cov_xy = sum((x - mean_x) * (y - mean_y) for x, y in points)
        slope = cov_xy / var_x

        if slope <= 0:
            return max(0, int(cpu_limit * safety_factor))

        estimate = (target_mem - zero_limit) / slope
        return max(0, int(min(estimate, cpu_limit) * safety_factor))
    elif kind == "cpu":
        lo = 0
        hi = 0
        for candidate_exp in itertools.count():
            lo = hi
            hi = 2**candidate_exp

            if progress is not None:
                progress(EstimateBatchsizeFromMemoryEvent(lo, None))
            if target_mem < fn(batch_size=hi):
                break

        while hi - lo > 1:
            mid = (lo + hi) // 2

            if progress is not None:
                progress(EstimateBatchsizeFromMemoryEvent(lo, hi))
            if target_mem < fn(batch_size=mid):
                hi = mid
            else:
                lo = mid

        return lo
    raise RuntimeError(f"Unsupported kind: {kind}")
