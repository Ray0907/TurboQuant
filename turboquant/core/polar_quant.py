import math
from functools import lru_cache

import mlx.core as mx

_CODEBOOK_GRID_SIZE = 6001
_LLOYD_MAX_ITERS = 50


def _cartesian_to_polar_pairs(x: mx.array) -> tuple[mx.array, mx.array]:
    assert x.shape[-1] % 2 == 0
    x1 = x[..., 0::2]
    x2 = x[..., 1::2]
    radii = mx.sqrt(x1 ** 2 + x2 ** 2 + 1e-12)
    angles = mx.arctan2(x2, x1)
    return radii, angles


def _level_bit_widths(polar_bits: float) -> tuple[int, int]:
    base_bits = max(1, round(polar_bits))
    return base_bits + 1, base_bits


@lru_cache(maxsize=None)
def _level_codebook(bits: int, level: int) -> tuple[tuple[float, ...], tuple[float, ...]]:
    n_codes = 2 ** bits
    if level == 0:
        lo = -math.pi
        step = (2 * math.pi) / n_codes
        centers = tuple(lo + (idx + 0.5) * step for idx in range(n_codes))
        bounds = tuple(lo + idx * step for idx in range(1, n_codes))
        return centers, bounds

    lo = 0.0
    hi = math.pi / 2
    exponent = 2 ** level - 1
    xs = tuple(lo + (hi - lo) * idx / (_CODEBOOK_GRID_SIZE - 1) for idx in range(_CODEBOOK_GRID_SIZE))
    ws = tuple(
        (math.sin(x) * math.cos(x)) ** exponent if 0 < x < hi else 0.0
        for x in xs
    )
    centers = [lo + (idx + 0.5) * (hi - lo) / n_codes for idx in range(n_codes)]

    for _ in range(_LLOYD_MAX_ITERS):
        bounds = [(centers[idx] + centers[idx + 1]) / 2 for idx in range(n_codes - 1)]
        new_centers = []
        left = lo
        start = 0
        for idx in range(n_codes):
            right = bounds[idx] if idx < n_codes - 1 else hi
            while start + 1 < _CODEBOOK_GRID_SIZE and xs[start] < left:
                start += 1
            end = start
            while end + 1 < _CODEBOOK_GRID_SIZE and xs[end] < right:
                end += 1

            numerator = 0.0
            denominator = 0.0
            for grid_idx in range(start, end + 1):
                numerator += xs[grid_idx] * ws[grid_idx]
                denominator += ws[grid_idx]
            new_centers.append(numerator / denominator if denominator else centers[idx])
            left = right
        centers = new_centers

    bounds = tuple((centers[idx] + centers[idx + 1]) / 2 for idx in range(n_codes - 1))
    return tuple(centers), bounds


def _quantize_level_angles(angles: mx.array, bits: int, level: int) -> mx.array:
    _, bounds = _level_codebook(bits, level)
    codes = mx.zeros(angles.shape, dtype=mx.int32)
    for boundary in bounds:
        codes = codes + (angles >= boundary).astype(mx.int32)
    return codes


def _dequantize_level_angles(codes: mx.array, bits: int, level: int) -> mx.array:
    centers, _ = _level_codebook(bits, level)
    angles = codes.astype(mx.float32) * 0 + centers[0]
    for idx, center in enumerate(centers[1:], start=1):
        angles = mx.where(codes == idx, center, angles)
    return angles


def polar_quantize(x: mx.array, polar_bits: float) -> tuple[mx.array, mx.array]:
    radius = mx.linalg.norm(x, axis=-1).astype(mx.float16)
    all_angles = []
    current = x.astype(mx.float32)
    first_level_bits, deeper_level_bits = _level_bit_widths(polar_bits)
    level = 0
    while current.shape[-1] >= 2:
        radii, angles = _cartesian_to_polar_pairs(current)
        bits = first_level_bits if level == 0 else deeper_level_bits
        codes = _quantize_level_angles(angles, bits, level)
        all_angles.append(codes)
        current = radii
        level += 1
    angles_tensor = mx.concatenate(all_angles, axis=-1)
    return angles_tensor, radius


def polar_dequantize(angles: mx.array, radius: mx.array, dim: int) -> mx.array:
    if dim < 1:
        raise ValueError(f"dim must be positive, got {dim}")

    if angles.shape[-1] == 0:
        return radius.astype(mx.float32)[..., None][..., :dim]

    first_level_bits = max(1, int(mx.max(angles).item()).bit_length())
    deeper_level_bits = max(1, first_level_bits - 1)

    level_sizes = []
    size = dim // 2
    while size >= 1:
        level_sizes.append(size)
        size //= 2

    level_angles = []
    offset = 0
    for size in level_sizes:
        level_angles.append(angles[..., offset : offset + size])
        offset += size

    current = radius.astype(mx.float32)[..., None]
    for level, level_codes in reversed(list(enumerate(level_angles))):
        bits = first_level_bits if level == 0 else deeper_level_bits
        theta = _dequantize_level_angles(level_codes, bits, level)
        x1 = current * mx.cos(theta)
        x2 = current * mx.sin(theta)
        current = mx.stack([x1, x2], axis=-1).reshape(*x1.shape[:-1], 2 * x1.shape[-1])

    return current[..., :dim]


def polar_inner(q: mx.array, angles: mx.array, radius: mx.array) -> mx.array:
    k_hat = polar_dequantize(angles, radius, dim=q.shape[-1])
    return mx.einsum("...id,...jd->...ij", q, k_hat)
