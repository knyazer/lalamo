import json
import os
import sys
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import equinox as eqx
import jax
import jax.numpy as jnp

from lalamo.message_processor import UserMessage
from lalamo.model_import import REPO_TO_MODEL, import_model
from lalamo.models.language_model import LanguageModelConfig


def _read_peak_bytes(stats: dict[str, int] | None) -> int | None:
    if stats is None:
        return None
    if "peak_bytes_in_use" in stats:
        return stats["peak_bytes_in_use"]
    if "bytes_in_use" in stats:
        return stats["bytes_in_use"]
    return None


def _load_model(model_ref: str):
    path = Path(model_ref)
    if path.exists():
        return LanguageModelConfig.load_model(path)
    raise ValueError(f"Unknown model reference: {model_ref!r}")


def main() -> None:
    model_ref = sys.argv[1]
    batch_size = int(sys.argv[2])

    gpu_devices = jax.devices("gpu")
    if not gpu_devices:
        raise RuntimeError("No GPU devices detected; set JAX to use a GPU for this script.")
    device = gpu_devices[0]

    model = _load_model(model_ref)
    prompt = [UserMessage("Count from 1 to 20 separated by spaces.")]
    prompt_token_ids = jnp.array(model.message_processor.tokenize_request(prompt), dtype=jnp.int32)

    max_input_length = 1024
    max_output_length = 1024
    num_logits_per_token = 8

    prompt_len = prompt_token_ids.size
    if prompt_len > max_input_length:
        raise ValueError("prompt exceeds max_input_length")

    prompt_token_ids = jax.device_put(prompt_token_ids, device)
    padded = jnp.pad(
        prompt_token_ids,
        (0, max_input_length - prompt_len),
        constant_values=0,
    )
    batch_prompt_ids = jnp.repeat(padded[None, :], batch_size, axis=0)
    batch_prompt_lengths = jnp.array([prompt_len] * batch_size, dtype=jnp.int32)

    with jax.default_device(device):
        results = eqx.filter_jit(model.generate_tokens)(
            batch_prompt_ids,
            prompt_lengths_without_padding=batch_prompt_lengths,
            max_output_length=max_output_length,
            num_top_logits_to_return=num_logits_per_token,
        )
    jax.block_until_ready(results.token_ids)

    after_stats = device.memory_stats()
    peak_bytes = _read_peak_bytes(after_stats)

    payload = {
        "model": model_ref,
        "batch_size": batch_size,
        "peak_bytes_in_use": peak_bytes / (2**30),
    }
    print(json.dumps(payload))


if __name__ == "__main__":
    main()
