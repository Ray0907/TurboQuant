import mlx.core as mx
from turboquant.core.qjl import generate_jl_matrix, qjl_sketch, qjl_correct


def test_sketch_shape(random_vectors):
    m = 32
    jl = generate_jl_matrix(m=m, d=64, seed=0)
    signs, scale = qjl_sketch(random_vectors, jl)
    assert signs.shape == (*random_vectors.shape[:-1], m)
    assert scale.shape == random_vectors.shape[:-1]
    assert signs.dtype == mx.uint8
    assert scale.dtype == mx.float16


def test_scale_is_norm(random_vectors):
    jl = generate_jl_matrix(m=32, d=64, seed=0)
    _, scale = qjl_sketch(random_vectors, jl)
    expected = mx.linalg.norm(random_vectors, axis=-1).astype(mx.float16)
    assert mx.allclose(scale.astype(mx.float32), expected.astype(mx.float32), rtol=1e-3)


def test_correction_sign(random_vectors, random_queries):
    # Use all 8 keys; qjl_correct returns [b,h,q,k], sign should match >70% of the time
    keys = random_vectors[:, :, :, :]    # (2, 4, 8, 64)
    q = random_queries                   # (2, 4, 1, 64)
    jl = generate_jl_matrix(m=64, d=64, seed=42)
    signs, scale = qjl_sketch(keys, jl)
    approx = qjl_correct(q, signs, scale, jl)              # (2, 4, 1, 8)
    exact = mx.einsum("...id,...jd->...ij", q, keys)        # (2, 4, 1, 8)
    mask = mx.abs(exact) > 1.0
    if bool(mx.any(mask)):
        match = (mx.sign(approx) == mx.sign(exact)).astype(mx.float32)
        accuracy = mx.sum(match * mask.astype(mx.float32)) / mx.sum(mask.astype(mx.float32))
        assert accuracy.item() > 0.70


def test_correction_accuracy(random_vectors, random_queries):
    keys = random_vectors[:, :, :, :]    # (2, 4, 8, 64)
    q = random_queries                   # (2, 4, 1, 64)
    jl = generate_jl_matrix(m=64, d=64, seed=42)
    signs, scale = qjl_sketch(keys, jl)
    approx = qjl_correct(q, signs, scale, jl)              # (2, 4, 1, 8)
    exact = mx.einsum("...id,...jd->...ij", q, keys)        # (2, 4, 1, 8)
    norm_q = mx.linalg.norm(q, axis=-1)[..., None]          # (2, 4, 1, 1)
    norm_k = mx.linalg.norm(keys, axis=-1)[..., None, :]    # (2, 4, 1, 8)
    denom = norm_q * norm_k + 1e-8                           # (2, 4, 1, 8)
    rel_err = mx.mean(mx.abs(approx - exact) / denom).item()
    assert rel_err < 0.50


def test_deterministic(random_vectors):
    jl = generate_jl_matrix(m=32, d=64, seed=0)
    s1, r1 = qjl_sketch(random_vectors, jl)
    s2, r2 = qjl_sketch(random_vectors, jl)
    assert bool(mx.all(s1 == s2)) and bool(mx.all(r1 == r2))
