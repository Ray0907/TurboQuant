import mlx.core as mx
from turboquant.core.polar_quant import polar_quantize, polar_dequantize, polar_inner
from turboquant.core.rotation import random_rotate


def test_output_shapes(random_vectors):
    angles, radius = polar_quantize(random_vectors, polar_bits=2.5)
    assert radius.shape == random_vectors.shape[:-1]
    assert angles.shape[-1] < random_vectors.shape[-1]


def test_radius_is_l2_norm(random_vectors):
    expected_norms = mx.linalg.norm(random_vectors, axis=-1)
    _, radius = polar_quantize(random_vectors, polar_bits=2.5)
    assert mx.allclose(
        expected_norms.astype(mx.float32), radius.astype(mx.float32), rtol=1e-3
    )


def test_mse_bound(random_vectors):
    # PolarQuant MSE bound holds for RHT-rotated vectors (paper Theorem 1)
    rotated, _ = random_rotate(random_vectors)
    polar_bits = 3.0
    C = 2.7
    angles, radius = polar_quantize(rotated, polar_bits=polar_bits)
    x_hat = polar_dequantize(angles, radius, dim=rotated.shape[-1])
    rv = rotated.astype(mx.float32)
    mse = mx.mean((rv - x_hat) ** 2, axis=-1)
    bound = C * (2 ** (-2 * polar_bits)) * mx.mean(rv ** 2, axis=-1)
    assert bool(mx.all(mse <= bound + 1e-6))


def test_polar_inner_vs_dot(random_vectors, random_queries):
    # TurboQuant always applies RHT before quantizing, so test with rotated keys
    polar_bits = 4.0
    keys = random_vectors[:, :, :1, :]
    queries = random_queries
    rotated_keys, _ = random_rotate(keys)
    angles, radius = polar_quantize(rotated_keys, polar_bits=polar_bits)
    approx = polar_inner(queries, angles, radius)
    exact = mx.einsum("...id,...jd->...ij", queries, rotated_keys)
    rel_err = mx.mean(mx.abs(approx - exact) / (mx.abs(exact) + 1e-8)).item()
    assert rel_err < 0.20


def test_deterministic(random_vectors):
    a1, r1 = polar_quantize(random_vectors, polar_bits=3.0)
    a2, r2 = polar_quantize(random_vectors, polar_bits=3.0)
    assert bool(mx.all(a1 == a2)) and bool(mx.all(r1 == r2))
