import mlx.core as mx
import pytest
from turboquant.core.rotation import random_rotate, inverse_rotate, next_power_of_2


def test_next_power_of_2():
    assert next_power_of_2(1) == 1
    assert next_power_of_2(3) == 4
    assert next_power_of_2(64) == 64
    assert next_power_of_2(65) == 128


def test_norm_preservation(random_vectors):
    rotated, signs = random_rotate(random_vectors)
    orig_norms = mx.linalg.norm(random_vectors, axis=-1)
    rot_norms = mx.linalg.norm(rotated, axis=-1)
    assert mx.allclose(orig_norms, rot_norms, rtol=1e-4, atol=1e-4)


def test_inverse_rotation(random_vectors):
    rotated, signs = random_rotate(random_vectors)
    recovered = inverse_rotate(rotated, signs)
    d = random_vectors.shape[-1]
    assert mx.allclose(random_vectors, recovered[..., :d], atol=1e-4)


def test_rotation_is_not_identity(random_vectors):
    rotated, signs = random_rotate(random_vectors)
    d = random_vectors.shape[-1]
    assert not mx.allclose(random_vectors, rotated[..., :d], atol=1e-3)


def test_padding_to_power_of_2():
    x = mx.random.normal((1, 1, 4, 65))
    rotated, signs = random_rotate(x)
    assert rotated.shape[-1] == 128
    assert signs.shape[-1] == 128


def test_signs_are_plus_minus_one(random_vectors):
    _, signs = random_rotate(random_vectors)
    assert bool(mx.all((signs == 1) | (signs == -1)))
