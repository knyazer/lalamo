"""Device offloading utilities for large models."""

import os
from collections.abc import Callable
from typing import TypeVar

import jax
import jax.tree_util as jtu
from jaxtyping import PyTree

T = TypeVar("T", bound=PyTree)


def to_device(tree: T, device: jax.Device) -> T:
    """Move a PyTree to the specified device."""
    return jtu.tree_map(lambda x: jax.device_put(x, device) if hasattr(x, "shape") else x, tree)


def offload_aware_apply(fn: Callable[[T], PyTree], module: T) -> Callable[[T], PyTree]:
    """
    Wrapper that moves module to GPU before computation, returns result on GPU.

    For CPU-loaded models, this enables selective layer-by-layer GPU execution:
    - Module stays on CPU when idle
    - Moved to GPU only during forward pass
    - Result stays on GPU for downstream layers

    Usage:
        layer_fn = offload_aware_apply(lambda m: m.forward, cpu_layer)
        gpu_output = layer_fn(cpu_layer)(gpu_input)
    """
    gpu_available = any(d.platform == "gpu" for d in jax.devices())
    enabled = os.getenv("LALAMO_GPU_OFFLOAD", "1") == "1" and gpu_available

    if not enabled:
        return fn

    gpu = next((d for d in jax.devices() if d.platform == "gpu"), jax.devices()[0])

    def wrapped(module: T) -> Callable:
        # Move module to GPU
        gpu_module = to_device(module, gpu)
        # Return the function bound to GPU module
        return fn(gpu_module)

    return wrapped
