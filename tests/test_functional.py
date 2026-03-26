import pytest
import mlx.core as mx
from turboquant.mlx.functional import compress, decompress, inner_product, CompressedKV


def test_compress_returns_compressed_kv(random_vectors):
    compressed = compress(random_vectors, bits=3.5)
    assert isinstance(compressed, CompressedKV)
    assert compressed.angles is not None
    assert compressed.radius is not None
    assert compressed.sign_bits is not None
    assert compressed.qjl_scale is not None
    assert compressed.signs is not None
    assert compressed.jl_matrix is not None


def test_compress_bits_allocation(random_vectors):
    mx.random.seed(0)
    d_pad = 64
    fixed_signs = mx.random.randint(0, 2, shape=(d_pad,)).astype(mx.float32) * 2 - 1
    fixed_jl = mx.random.normal((64, d_pad))

    c1 = compress(random_vectors, bits=2.5, signs=fixed_signs, jl_matrix=fixed_jl)
    c2 = compress(random_vectors, bits=4.0, signs=fixed_signs, jl_matrix=fixed_jl)
    assert not bool(mx.all(c1.angles == c2.angles))


def test_inner_product_decode_shape(random_vectors, random_queries):
    compressed = compress(random_vectors, bits=3.5)
    scores = inner_product(random_queries, compressed)
    # queries: [2,4,1,64], keys: [2,4,8,64] -> scores: [2,4,1,8]
    assert scores.shape == (2, 4, 1, 8), f"Unexpected shape: {scores.shape}"


def test_inner_product_is_approximately_correct(random_vectors, random_queries):
    keys = random_vectors[:, :, :4, :]   # [2,4,4,64]
    queries = random_queries              # [2,4,1,64]

    compressed = compress(keys, bits=5.0)
    approx = inner_product(queries, compressed)
    exact = mx.einsum("bhid,bhjd->bhij", queries, keys)

    # Normalize by ||q||*||k|| to avoid near-zero denominator issues
    norm_q = mx.linalg.norm(queries, axis=-1, keepdims=True)[..., None]  # [2,4,1,1]
    norm_k = mx.linalg.norm(keys, axis=-1)[..., None, :]                 # [2,4,1,4]
    denom = norm_q.squeeze(-1) * norm_k + 1e-8
    rel_err = mx.mean(mx.abs(approx - exact) / denom).item()
    assert rel_err < 0.15, f"inner_product relative error: {rel_err:.4f}"


def test_decompress_shape(random_vectors):
    compressed = compress(random_vectors, bits=3.5)
    recovered = decompress(compressed, dim=random_vectors.shape[-1])
    assert recovered.shape == random_vectors.shape


def test_decompress_is_biased():
    mx.random.seed(0)
    keys = mx.random.normal((1, 1, 4, 64))
    queries = mx.random.normal((1, 1, 1, 64))

    compressed = compress(keys, bits=3.5)
    scores_correct = inner_product(queries, compressed)
    keys_approx = decompress(compressed, dim=64)
    scores_naive = mx.einsum("bhid,bhjd->bhij", queries, keys_approx)

    assert not mx.allclose(scores_correct, scores_naive, atol=1e-3), \
        "inner_product and decompress+dot give same result — QJL correction may be missing"


def test_bits_minimum_validation():
    x = mx.random.normal((1, 1, 4, 64))
    with pytest.raises(ValueError, match="bits"):
        compress(x, bits=0.5)
