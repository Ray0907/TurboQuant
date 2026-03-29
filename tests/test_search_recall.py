"""
Recall@k test: TurboQuant search vs exact brute-force search.

No mocks, no external dependencies — pure MLX computation.
Validates that compress_vectors + search finds the true nearest neighbours.

Run with:
    uv run --with pytest pytest tests/test_search_recall.py -v -s
"""

import mlx.core as mx
import pytest
from turboquant.mlx import compress_vectors, search


def _exact_search(query, docs, top_k):
    """Brute-force exact nearest-neighbour search."""
    if query.ndim == 1:
        scores = docs @ query                    # [N]
        indices = mx.argsort(scores)[-top_k:][::-1]
        return scores[indices], indices
    scores = mx.einsum("qd,kd->qk", query, docs) # [Q, N]
    indices = mx.argsort(scores, axis=-1)[:, -top_k:][:, ::-1]
    return mx.take_along_axis(scores, indices, axis=-1), indices


def recall_at_k(approx_indices, exact_indices):
    """Fraction of exact top-k results found in approx top-k results."""
    hits = 0
    total = 0
    for a_row, e_row in zip(approx_indices.tolist(), exact_indices.tolist()):
        a_set = set(a_row)
        hits  += sum(1 for i in e_row if i in a_set)
        total += len(e_row)
    return hits / total if total else 0.0


@pytest.mark.parametrize("bits,expected_recall", [
    (3.5, 0.50),
    (4.0, 0.60),
    (5.0, 0.75),
])
def test_recall_at_10(bits, expected_recall):
    """Recall@10 should improve monotonically with bits and stay above floor."""
    mx.random.seed(0)
    N, D, Q = 500, 64, 20
    docs    = mx.random.normal((N, D))
    queries = mx.random.normal((Q, D))
    top_k   = 10

    corpus = compress_vectors(docs, bits=bits)
    _, approx_idx = search(queries, corpus, top_k=top_k)
    _, exact_idx  = _exact_search(queries, docs, top_k=top_k)

    recall = recall_at_k(approx_idx, exact_idx)
    print(f"\n  bits={bits}: recall@{top_k} = {recall:.1%}")
    assert recall >= expected_recall, (
        f"bits={bits}: recall@{top_k} = {recall:.1%}, expected >= {expected_recall:.0%}"
    )


def test_recall_improves_with_bits():
    """Quality should increase monotonically from 3.5 → 4.0 → 5.0 bits."""
    mx.random.seed(1)
    N, D, Q = 500, 64, 30
    docs    = mx.random.normal((N, D))
    queries = mx.random.normal((Q, D))
    top_k   = 10
    _, exact_idx = _exact_search(queries, docs, top_k=top_k)

    recalls = {}
    for bits in [3.5, 4.0, 5.0]:
        corpus = compress_vectors(docs, bits=bits)
        _, approx_idx = search(queries, corpus, top_k=top_k)
        recalls[bits] = recall_at_k(approx_idx, exact_idx)

    print(f"\n  recalls: { {b: f'{r:.1%}' for b, r in recalls.items()} }")
    assert recalls[5.0] >= recalls[3.5], (
        f"5.0-bit recall ({recalls[5.0]:.1%}) should exceed 3.5-bit ({recalls[3.5]:.1%})"
    )


def test_single_query():
    """Single query (1D input) should return 1D results."""
    mx.random.seed(2)
    docs  = mx.random.normal((200, 64))
    query = mx.random.normal((64,))

    corpus = compress_vectors(docs, bits=4.0)
    scores, indices = search(query, corpus, top_k=5)

    assert scores.shape  == (5,), f"scores shape: {scores.shape}"
    assert indices.shape == (5,), f"indices shape: {indices.shape}"
    assert all(0 <= i < 200 for i in indices.tolist())


def test_exact_top1_in_approx_top5():
    """The exact nearest neighbour should appear in TurboQuant's top-5 > 80% of the time."""
    mx.random.seed(3)
    N, D, Q = 300, 64, 50
    docs    = mx.random.normal((N, D))
    queries = mx.random.normal((Q, D))

    corpus = compress_vectors(docs, bits=4.0)
    _, approx_idx = search(queries, corpus, top_k=5)
    _, exact_idx  = _exact_search(queries, docs, top_k=1)

    hits = sum(
        exact_idx[q, 0].item() in approx_idx[q].tolist()
        for q in range(Q)
    )
    rate = hits / Q
    print(f"\n  exact top-1 in approx top-5: {rate:.1%}")
    assert rate >= 0.80, f"Expected >= 80%, got {rate:.1%}"
