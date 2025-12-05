import pytest

import torch
import numpy as np
from groupcl.models import dynamics
from groupcl.utils.bivector import bivector_rotation, subspace_projection_matrix, subspace_projection_matrix, are_independent

ATOL = 1e-3
RTOL = 1e-3
NUM_SEEDS = 1
DTYPE = torch.float64


def _groupcl_similarity(x, x_prime, y, y_prime):

    # R <- bivector_rotation(x, x_prime)
    R = bivector_rotation(x, x_prime)

    # P <- subspace_projection_matrix(x, x_prime)
    P = subspace_projection_matrix(x, x_prime)

    # yp, yp_prime <- normed(Py), normed(P y_prime)
    yp = torch.einsum("...ij,...j->...i", P, y)
    yp_prime = torch.einsum("...ij,...j->...i", P, y_prime)

    # normalize yp, yp_prime
    yp = yp / torch.norm(yp, dim=-1, keepdim=True)
    yp_prime = yp_prime / torch.norm(yp_prime, dim=-1, keepdim=True)

    # s <- (yp_prime).T R yp

    Ryp = torch.einsum("...ij,...j->...i", R, yp)

    s = torch.einsum("...i,...i->...", yp_prime, Ryp)
    diff = yp_prime - Ryp

    return s, diff


def simple_rotation(angle, dim, i=0, j=1):
    assert i != j, "i and j must be different"
    assert i < dim, "i must be less than dim"
    assert j < dim, "j must be less than dim"
    theta_rad = torch.tensor(np.radians(angle), dtype=DTYPE)
    Qrot = torch.eye(dim, dtype=DTYPE)
    Qrot[i, i] = torch.cos(theta_rad)
    Qrot[i, j] = -torch.sin(theta_rad)
    Qrot[j, i] = torch.sin(theta_rad)
    Qrot[j, j] = torch.cos(theta_rad)
    return Qrot


def orthogonal_pair(seed, batch_size, dim):
    torch.manual_seed(seed)

    x = torch.randn(batch_size, dim, dtype=DTYPE)
    x = x / x.norm(dim=-1, keepdim=True)
    # Create a random vector
    random_vec = torch.randn(batch_size, dim, dtype=DTYPE)
    # Project out the component parallel to x to get an orthogonal vector
    # For batched vectors, we need to use einsum for proper batch-wise dot products
    # random_vec @ x computes dot product along last dimension for each batch element
    dot_products = torch.einsum('bi,bi->b', random_vec, x).unsqueeze(-1)
    x_squared = torch.einsum('bi,bi->b', x, x).unsqueeze(-1)
    orthogonal_to_x = random_vec - (dot_products / x_squared) * x
    # Normalize the orthogonal vector
    orthogonal_to_x = orthogonal_to_x / orthogonal_to_x.norm(dim=-1,
                                                             keepdim=True)

    assert torch.allclose(torch.einsum('bi,bi->b', orthogonal_to_x, x),
                          torch.zeros(batch_size, dtype=DTYPE))

    y = orthogonal_to_x

    return x, y


def random_vector(seed, batch_size, dim):
    torch.manual_seed(seed)

    x = torch.randn(batch_size, dim, dtype=DTYPE)
    x = x / x.norm(dim=-1, keepdim=True)

    y = torch.randn(batch_size, dim, dtype=DTYPE)
    y = y / y.norm(dim=-1, keepdim=True)

    return x, y


@pytest.mark.parametrize("seed", [0])
@pytest.mark.parametrize("batch_size", [10])
@pytest.mark.parametrize("dim", [2])
@pytest.mark.parametrize("angle", [10, 20, 40, 90, 120])
@pytest.mark.parametrize("i", [0, 1, 2])
@pytest.mark.parametrize("j", [0, 1, 2])
def test_groupcl_similarity(
    seed,
    batch_size,
    dim,
    angle,
    i,
    j,
):
    # skip tests if i == j
    if i == j:
        pytest.skip("Skipping tests where i == j")
    if i >= dim or j >= dim:
        pytest.skip(
            "Skipping tests where i or j is greater than or equal to dim")

    Q = simple_rotation(angle, dim, i, j)

    x, y = orthogonal_pair(seed, batch_size, dim)

    x_prime = torch.einsum("...ij,...j->...i", Q, x)
    y_prime = torch.einsum("...ij,...j->...i", Q, y)

    s, diff = _groupcl_similarity(x, x_prime, y, y_prime)

    assert torch.allclose(s, torch.ones_like(
        s)), f"dot product not all =1, min={s.min()}, max={s.max()}"
    assert torch.allclose(diff, torch.zeros_like(
        diff)), f"difference not all =0, min={diff.min()}, max={diff.max()}"


@pytest.mark.parametrize("seed", [0])
@pytest.mark.parametrize("batch_size", [100])
@pytest.mark.parametrize("dim", [2])
@pytest.mark.parametrize("angle", [10, 20, 40, 90, 120])
@pytest.mark.parametrize("i", [0, 1, 2])
@pytest.mark.parametrize("j", [0, 1, 2])
def test_groupcl_similarity_random_vectors(
    seed,
    batch_size,
    dim,
    angle,
    i,
    j,
):
    # skip tests if i == j
    if i == j:
        pytest.skip("Skipping tests where i == j")
    if i >= dim or j >= dim:
        pytest.skip(
            "Skipping tests where i or j is greater than or equal to dim")

    Q = simple_rotation(angle, dim, i, j)

    x, y = random_vector(seed, batch_size, dim)

    x_prime = torch.einsum("...ij,...j->...i", Q, x)
    y_prime = torch.einsum("...ij,...j->...i", Q, y)

    s, diff = _groupcl_similarity(x, x_prime, y, y_prime)

    assert torch.allclose(s, torch.ones_like(
        s)), f"dot product not all =1, min={s.min()}, max={s.max()}"
    assert torch.allclose(diff, torch.zeros_like(
        diff)), f"difference not all =0, min={diff.min()}, max={diff.max()}"


@pytest.mark.parametrize("seed", [0])
@pytest.mark.parametrize("batch_size", [100])
@pytest.mark.parametrize("dim", [2])
@pytest.mark.parametrize("angle", [10, 20, 40, 90, 120])
@pytest.mark.parametrize("i", [0, 1, 2])
@pytest.mark.parametrize("j", [0, 1, 2])
def test_bivector_model_random_vectors(
    seed,
    batch_size,
    dim,
    angle,
    i,
    j,
):
    # skip tests if i == j
    if i == j:
        pytest.skip("Skipping tests where i == j")
    if i >= dim or j >= dim:
        pytest.skip(
            "Skipping tests where i or j is greater than or equal to dim")

    Q = simple_rotation(angle, dim, i, j)

    x, y = random_vector(seed, batch_size, dim)

    x_prime = torch.einsum("...ij,...j->...i", Q, x)
    y_prime = torch.einsum("...ij,...j->...i", Q, y)

    bivector_model = dynamics.BivectorDynamicsModel(
        dtype=str(DTYPE),
        eps=1e-8,
    )
    (yp, yp_prime, y_prime_pred) = bivector_model(
        x=x,
        y=y,
        x_prime=x_prime,
        y_prime=y_prime,
    )

    assert torch.allclose(
        yp_prime, y_prime_pred
    ), f"yp_prime not allclose to y_prime_pred, min={yp_prime.min()}, max={yp_prime.max()}, min_diff={torch.norm(yp_prime - y_prime_pred, dim=-1).min()}, max_diff={torch.norm(yp_prime - y_prime_pred, dim=-1).max()}"


if __name__ == "__main__":
    pytest.main([__file__])
