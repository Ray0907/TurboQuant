"""Quantized Johnson-Lindenstrauss sketch utilities (MLX)."""

from __future__ import annotations

import math

import mlx.core as mx


def generate_jl_matrix(m: int, d: int, seed: int | None = None) -> mx.array:
    """Return a random Gaussian sketch matrix of shape ``(m, d)``."""
    if m < 1:
        raise ValueError(f"m must be positive, got {m}")
    if d < 1:
        raise ValueError(f"d must be positive, got {d}")
    if seed is not None:
        mx.random.seed(seed)
    return mx.random.normal((m, d))


def qjl_sketch(x: mx.array, jl_matrix: mx.array) -> tuple[mx.array, mx.array]:
    """Compress ``x`` with a 1-bit JL sketch.

    Args:
        x: Input tensor of shape ``(..., d)``.
        jl_matrix: Sketch matrix of shape ``(m, d)``.

    Returns:
        signs: uint8 tensor of shape ``(..., m)`` with values in {0, 1}.
        scale: float16 tensor of shape ``(...)`` storing the L2 norm of ``x``.
    """
    scale = mx.linalg.norm(x, axis=-1).astype(mx.float16)
    projections = x.astype(mx.float32) @ jl_matrix.T  # (..., m)
    signs = (projections >= 0).astype(mx.uint8)
    return signs, scale


def qjl_correct(
    q: mx.array,
    signs: mx.array,
    scale: mx.array,
    jl_matrix: mx.array,
) -> mx.array:
    """Estimate ``<q, x>`` from a 1-bit JL sketch.

    Uses the unbiased estimator:
    ``sqrt(pi/2) * (scale / m) * <G q, 2s - 1>``

    Args:
        q: Query tensor of shape ``(..., d)``.
        signs: uint8 sign bits of shape ``(..., m)``.
        scale: float16 L2 norm of the key, shape ``(...)``.
        jl_matrix: Sketch matrix of shape ``(m, d)``.

    Returns:
        Estimate of ``<q, x>`` with shape ``(...)``.
    """
    m = jl_matrix.shape[0]
    sketch_q = q.astype(mx.float32) @ jl_matrix.T  # (..., m)
    s_pm1 = signs.astype(mx.float32) * 2 - 1  # {0,1} -> {-1,+1}
    dot = mx.sum(sketch_q * s_pm1, axis=-1)
    return math.sqrt(math.pi / 2.0) * (scale.astype(mx.float32) / m) * dot
