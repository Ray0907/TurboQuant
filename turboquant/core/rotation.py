import math
import mlx.core as mx


def next_power_of_2(n: int) -> int:
    if n <= 1:
        return 1
    return 1 << (n - 1).bit_length()


def _wht(x: mx.array) -> mx.array:
    d = x.shape[-1]
    assert d & (d - 1) == 0, f"dim must be power of 2, got {d}"
    prefix = x.shape[:-1]
    h = 1
    while h < d:
        x = x.reshape(*prefix, d // (2 * h), 2, h)
        a = x[..., 0, :]
        b = x[..., 1, :]
        x = mx.concatenate([(a + b)[..., None, :], (a - b)[..., None, :]], axis=-2)
        x = x.reshape(*prefix, d)
        h *= 2
    return x


def random_rotate(x: mx.array, signs: mx.array | None = None) -> tuple[mx.array, mx.array]:
    d = x.shape[-1]
    d_pad = next_power_of_2(d)
    if d_pad != d:
        pad_width = [(0, 0)] * (x.ndim - 1) + [(0, d_pad - d)]
        x = mx.pad(x, pad_width)
    if signs is None:
        signs = mx.random.randint(0, 2, shape=(d_pad,)).astype(mx.float32) * 2 - 1
    x_signed = x.astype(mx.float32) * signs
    x_rotated = _wht(x_signed) / math.sqrt(d_pad)
    return x_rotated, signs


def inverse_rotate(x_rotated: mx.array, signs: mx.array) -> mx.array:
    d_pad = x_rotated.shape[-1]
    x_unwhted = _wht(x_rotated) / math.sqrt(d_pad)
    return x_unwhted * signs
