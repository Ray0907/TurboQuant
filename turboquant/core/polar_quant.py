import math
import mlx.core as mx


def _cartesian_to_polar_pairs(x: mx.array) -> tuple[mx.array, mx.array]:
    assert x.shape[-1] % 2 == 0
    x1 = x[..., 0::2]
    x2 = x[..., 1::2]
    radii = mx.sqrt(x1 ** 2 + x2 ** 2 + 1e-12)
    angles = mx.arctan2(x2, x1)
    return radii, angles


def _quantize_angles(angles: mx.array, polar_bits: float) -> mx.array:
    bits = max(1, round(polar_bits))
    n_levels = 2 ** bits
    normalized = (angles + math.pi) / (2 * math.pi)
    codes = mx.clip((normalized * n_levels).astype(mx.int32), 0, n_levels - 1)
    return codes


def _dequantize_angles(codes: mx.array, polar_bits: float) -> mx.array:
    bits = max(1, round(polar_bits))
    n_levels = 2 ** bits
    normalized = (codes.astype(mx.float32) + 0.5) / n_levels
    return normalized * (2 * math.pi) - math.pi


def polar_quantize(x: mx.array, polar_bits: float) -> tuple[mx.array, mx.array]:
    radius = mx.linalg.norm(x, axis=-1).astype(mx.float16)
    all_angles = []
    current = x.astype(mx.float32)
    while current.shape[-1] >= 2:
        radii, angles = _cartesian_to_polar_pairs(current)
        codes = _quantize_angles(angles, polar_bits)
        all_angles.append(codes)
        current = radii
    angles_tensor = mx.concatenate(all_angles, axis=-1)
    return angles_tensor, radius


def polar_dequantize(angles: mx.array, radius: mx.array, dim: int) -> mx.array:
    raise NotImplementedError("polar_dequantize not yet implemented")


def polar_inner(q: mx.array, angles: mx.array, radius: mx.array) -> mx.array:
    raise NotImplementedError("polar_inner not yet implemented")
