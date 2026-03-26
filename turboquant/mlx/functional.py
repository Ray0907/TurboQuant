"""TurboQuant MLX functional API."""

from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx

from turboquant.core.polar_quant import polar_dequantize, polar_inner, polar_quantize
from turboquant.core.qjl import generate_jl_matrix, qjl_correct, qjl_sketch
from turboquant.core.rotation import inverse_rotate, random_rotate


@dataclass
class CompressedKV:
    """Compressed key representation produced by :func:`compress`.

    ``signs`` and ``jl_matrix`` are shared across the sequence dimension and
    must be fixed at the layer level so that a cache rebuild is idempotent.
    """

    angles: mx.array     # int32   [..., seq, num_angles]
    radius: mx.array     # float16 [..., seq]
    sign_bits: mx.array  # uint8   [..., seq, m]
    qjl_scale: mx.array  # float16 [..., seq]
    signs: mx.array      # float32 [dim_padded]
    jl_matrix: mx.array  # float32 [m, dim_padded]


def compress(
    keys: mx.array,
    bits: float = 3.5,
    m: int = 64,
    signs: mx.array | None = None,
    jl_matrix: mx.array | None = None,
) -> CompressedKV:
    """Compress key cache arrays using TurboQuant.

    Pipeline: ``random_rotate`` → ``polar_quantize`` (bits-1 bits) → ``qjl_sketch``
    (1 bit on the residual).

    Args:
        keys:      Key array ``[batch, heads, seq, dim]``.
        bits:      Total bits per element. Must be > 1.0. Recommended: 2.5–4.0.
        m:         QJL projection dimension. Higher ``m`` → lower variance.
        signs:     Optional ``+/-1`` signs ``[dim_padded]``. Reuse across calls.
        jl_matrix: Optional JL matrix ``[m, dim_padded]``. Reuse across calls.

    Returns:
        :class:`CompressedKV` holding all fields needed for
        :func:`inner_product` and :func:`decompress`.
    """
    if bits <= 1.0:
        raise ValueError(f"bits must be > 1.0, got {bits}. Minimum recommended: 2.5.")

    polar_bits = bits - 1.0

    rotated, signs = random_rotate(keys, signs=signs)
    d_pad = rotated.shape[-1]

    angles, radius = polar_quantize(rotated, polar_bits=polar_bits)

    rotated_approx = polar_dequantize(angles, radius, dim=d_pad)
    residual = rotated.astype(mx.float32) - rotated_approx

    if jl_matrix is None:
        jl_matrix = generate_jl_matrix(m=m, d=d_pad)

    sign_bits, qjl_scale = qjl_sketch(residual, jl_matrix)

    return CompressedKV(
        angles=angles,
        radius=radius,
        sign_bits=sign_bits,
        qjl_scale=qjl_scale,
        signs=signs,
        jl_matrix=jl_matrix,
    )


def inner_product(queries: mx.array, compressed: CompressedKV) -> mx.array:
    """Compute approximate attention scores ``q @ k.T`` using compressed keys.

    Args:
        queries:    Query array ``[batch, heads, q_seq, dim]``.
        compressed: Output of :func:`compress`.

    Returns:
        Attention scores ``[batch, heads, q_seq, k_seq]``.
    """
    q_rotated, _ = random_rotate(queries, signs=compressed.signs)

    polar_term = polar_inner(q_rotated, compressed.angles, compressed.radius)

    qjl_term = qjl_correct(
        q_rotated, compressed.sign_bits, compressed.qjl_scale, compressed.jl_matrix
    )

    return polar_term + qjl_term


def decompress(compressed: CompressedKV, dim: int) -> mx.array:
    """Reconstruct an approximate key array from its compressed representation.

    .. warning::
        Intentionally omits the QJL correction — the result is biased.
        Use :func:`inner_product` for attention computation.
        This function is for debugging and visualization only.

    Args:
        compressed: Output of :func:`compress`.
        dim:        Original key dimension (before padding).

    Returns:
        Approximate key array ``[..., dim]``. Biased.
    """
    d_pad = compressed.signs.shape[-1]
    k_rotated_approx = polar_dequantize(compressed.angles, compressed.radius, dim=d_pad)
    k_approx = inverse_rotate(k_rotated_approx, compressed.signs)
    return k_approx[..., :dim]
