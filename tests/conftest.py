import pytest
import mlx.core as mx


@pytest.fixture
def random_vectors():
    """Standard test vectors: batch=2, heads=4, seq=8, dim=64"""
    mx.random.seed(42)
    return mx.random.normal((2, 4, 8, 64))


@pytest.fixture
def random_queries():
    mx.random.seed(123)
    return mx.random.normal((2, 4, 1, 64))
