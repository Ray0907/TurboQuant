"""Integration test: TurboQuant on real Qwen2.5-0.5B K vectors.

Validates that TurboQuant inner_product approximation quality holds on actual
LLM activations, not synthetic data.

Run with:
    uv run --with "mlx-lm>=0.22" pytest tests/test_integration_qwen.py -v -m integration
"""

import pytest

mlx_lm = pytest.importorskip("mlx_lm", reason="mlx-lm not installed")

import mlx.core as mx

from turboquant.core.qjl import generate_jl_matrix
from turboquant.mlx.functional import compress, inner_product

MODEL_ID = "mlx-community/Qwen2.5-0.5B-Instruct-4bit"

# ~40 real tokens; enough for meaningful attention statistics
PROMPT = (
    "The quick brown fox jumps over the lazy dog. "
    "The dog was sleeping peacefully under the old oak tree. "
    "Suddenly, the fox stopped and looked around."
)


@pytest.fixture(scope="module")
def qwen_qk():
    """Load model and extract real Q, K tensors from layer 0."""
    try:
        model, tokenizer = mlx_lm.load(MODEL_ID)
    except Exception as e:
        pytest.skip(f"Could not load {MODEL_ID}: {e}")

    input_ids = mx.array([tokenizer.encode(PROMPT)])
    B, L = input_ids.shape
    print(f"\n  Qwen2.5-0.5B: {L} tokens, head_dim=64")

    attn    = model.model.layers[0].self_attn
    embed   = model.model.embed_tokens(input_ids)
    norm_x  = model.model.layers[0].input_layernorm(embed)
    n_q     = attn.n_heads
    n_kv    = attn.n_kv_heads
    hd      = model.args.hidden_size // n_q

    queries = attn.q_proj(norm_x).reshape(B, L, n_q,  hd).transpose(0, 2, 1, 3)
    keys    = attn.k_proj(norm_x).reshape(B, L, n_kv, hd).transpose(0, 2, 1, 3)
    queries = attn.rope(queries)
    keys    = attn.rope(keys)

    # Force evaluation
    _ = keys.tolist()
    return queries, keys, n_q, n_kv, hd


def _gqa_scores(queries, keys, n_q, n_kv, hd, bits, signs, jl_matrix):
    """Compute approx and exact attention scores handling GQA head grouping."""
    ratio = n_q // n_kv
    approx_parts, exact_parts = [], []
    for kv in range(n_kv):
        q_g = queries[:, kv * ratio : (kv + 1) * ratio]   # [1, ratio, L, hd]
        k_g = keys[:,   kv : kv + 1]                       # [1, 1,     L, hd]
        compressed = compress(k_g, bits=bits, signs=signs, jl_matrix=jl_matrix)
        approx_parts.append(inner_product(q_g, compressed))
        exact_parts.append(mx.einsum("bhid,bhjd->bhij", q_g, k_g))
    return (
        mx.concatenate(approx_parts, axis=1).astype(mx.float32),
        mx.concatenate(exact_parts,  axis=1).astype(mx.float32),
    )


def _cosine(approx, exact):
    """Mean per-row cosine similarity between approx and exact score matrices."""
    cos = mx.sum(approx * exact, axis=-1) / (
        mx.linalg.norm(approx, axis=-1) * mx.linalg.norm(exact, axis=-1) + 1e-8
    )
    return mx.mean(cos).item()


@pytest.mark.integration
def test_attention_quality_4bit(qwen_qk):
    """At 4 bits, cosine similarity of attention logits should be >= 0.90."""
    queries, keys, n_q, n_kv, hd = qwen_qk

    mx.random.seed(42)
    signs = mx.random.randint(0, 2, shape=(hd,)).astype(mx.float32) * 2 - 1
    jl    = generate_jl_matrix(m=64, d=hd, seed=0)

    approx, exact = _gqa_scores(queries, keys, n_q, n_kv, hd, bits=4.0, signs=signs, jl_matrix=jl)
    cos = _cosine(approx, exact)
    print(f"\n  4.0-bit cosine similarity: {cos:.4f}")
    assert cos >= 0.90, f"Expected >=0.90, got {cos:.4f}"


@pytest.mark.integration
def test_quality_improves_with_more_bits(qwen_qk):
    """Cosine similarity should be monotonically higher at 5 bits vs 3.5 bits."""
    queries, keys, n_q, n_kv, hd = qwen_qk

    mx.random.seed(42)
    signs = mx.random.randint(0, 2, shape=(hd,)).astype(mx.float32) * 2 - 1
    jl    = generate_jl_matrix(m=64, d=hd, seed=0)

    approx_lo, exact = _gqa_scores(queries, keys, n_q, n_kv, hd, bits=3.5, signs=signs, jl_matrix=jl)
    approx_hi, _     = _gqa_scores(queries, keys, n_q, n_kv, hd, bits=5.0, signs=signs, jl_matrix=jl)

    cos_lo = _cosine(approx_lo, exact)
    cos_hi = _cosine(approx_hi, exact)
    print(f"\n  3.5-bit cos={cos_lo:.4f}  5.0-bit cos={cos_hi:.4f}")
    assert cos_hi > cos_lo, (
        f"Expected quality to improve with more bits: 5.0-bit ({cos_hi:.4f}) "
        f"should exceed 3.5-bit ({cos_lo:.4f})"
    )
    assert cos_hi >= 0.95, f"5.0-bit cosine should be >=0.95, got {cos_hi:.4f}"
