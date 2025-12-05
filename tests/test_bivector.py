import pytest

import torch
import numpy as np
from groupcl.utils.bivector import bivector_rotation, subspace_projection_matrix, subspace_projection_matrix, are_independent

ATOL = 1e-3
RTOL = 1e-3
NUM_SEEDS = 10
DTYPE = torch.float64


def ref_impl_bivector_rotation_normalized(origin, target, eps=1e-8):
    """
    Same as bivector_rotation, but with the additional constraint that the origin and target
    are already normalized.
    """

    if not torch.allclose(
            origin.norm(dim=-1, keepdim=True),
            torch.tensor(1., device=origin.device, dtype=origin.dtype)):
        raise ValueError("origin not normalized")
    if not torch.allclose(
            target.norm(dim=-1, keepdim=True),
            torch.tensor(1., device=origin.device, dtype=origin.dtype)):
        raise ValueError("target not normalized")

    n = origin.shape[-1]

    assert origin.shape == target.shape

    batch_shape = origin.shape[:-1]
    I = torch.eye(n, device=origin.device,
                  dtype=origin.dtype).expand(*batch_shape, n, n)

    sum_of_vectors = (origin + target)
    outer_product = torch.einsum("...m,...n->...mn", target, origin)
    nominator = torch.einsum("...m,...n->...mn", sum_of_vectors, sum_of_vectors)
    denominator = 1 + torch.einsum("...n,...n->...", origin, target)

    Q = I + 2 * outer_product

    nonzero_denominator = torch.abs(denominator) > eps
    Q[nonzero_denominator] = Q[nonzero_denominator] - nominator[
        nonzero_denominator] / denominator[nonzero_denominator].unsqueeze(
            -1).unsqueeze(-1)
    return Q


def _assert_allclose(a, b, atol=ATOL, rtol=RTOL, err_msg=""):
    assert torch.allclose(
        a, b, atol=atol,
        rtol=rtol), f"{err_msg}: max diff: {torch.max(torch.abs(a - b))}"


def _assert_allclose_strict(a, b, atol=1e-8, rtol=1e-8, err_msg=""):
    np.testing.assert_allclose(a.detach().numpy(),
                               b.detach().numpy(),
                               atol=atol,
                               rtol=rtol,
                               err_msg=err_msg)


def _make_random_vectors(batch_size, dim, seed=None):
    """Helper function to create random normalized vectors for testing"""
    if seed is not None:
        torch.manual_seed(seed)

    # Create random vectors
    origin = torch.randn(batch_size, dim, dtype=DTYPE)
    target = torch.randn(batch_size, dim, dtype=DTYPE)

    # Normalize
    origin = origin / origin.norm(dim=-1, keepdim=True)
    target = target / target.norm(dim=-1, keepdim=True)

    return origin, target


@pytest.mark.parametrize("batch_size", [1, 2, 10])
@pytest.mark.parametrize("dim", [2, 3, 4, 10])
@pytest.mark.parametrize("seed", np.arange(NUM_SEEDS))
def test_bivector_rotation_random(batch_size, dim, seed):
    origin, target = _make_random_vectors(batch_size, dim, seed)
    Q = bivector_rotation(origin, target)

    assert Q.shape == (batch_size, dim, dim)

    rotated = torch.einsum("...ij,...j->...i", Q, origin)
    _assert_allclose(rotated, target)

    QQt = torch.einsum("...ij,...kj->...ik", Q, Q)
    I = torch.eye(dim, dtype=DTYPE).expand(batch_size, -1, -1)
    _assert_allclose_strict(QQt, I, atol=1e-3, rtol=1e-3)


@pytest.mark.parametrize("seed", np.arange(NUM_SEEDS))
def test_bivector_rotation_2d(seed):
    torch.manual_seed(seed)

    # Test with specific 2D vectors
    origin = torch.tensor([[0.2, .5]], dtype=DTYPE)
    target = torch.tensor([[.5, 0.]], dtype=DTYPE)
    origin = origin / origin.norm(dim=-1, keepdim=True)
    target = target / target.norm(dim=-1, keepdim=True)

    Q = bivector_rotation(origin, target)
    Q_ref = ref_impl_bivector_rotation_normalized(origin, target)

    _assert_allclose(Q, Q_ref)

    # Check output shape
    assert Q.shape == (1, 2, 2)

    # Check that Q rotates origin to target
    rotated = torch.einsum("...ij,...j->...i", Q, origin)
    _assert_allclose_strict(rotated, target, atol=1e-6)


@pytest.mark.parametrize("seed", np.arange(NUM_SEEDS))
@pytest.mark.parametrize("dim", [10, 20])
def test_bivector_rotation_2d_large(seed, dim):
    size = 1000
    origin, target = _make_random_vectors(size, dim, seed)
    Q = bivector_rotation(origin, target)
    Q_ref = ref_impl_bivector_rotation_normalized(origin, target)

    _assert_allclose(Q, Q_ref)

    # Check output shape
    assert Q.shape == (size, dim, dim)

    # Check that Q is orthogonal
    QQt = torch.einsum("...ij,...kj->...ik", Q, Q)
    I = torch.eye(dim, dtype=DTYPE).expand(size, -1, -1)
    _assert_allclose(QQt, I)


@pytest.mark.parametrize("eps", [1e-6, 1e-8, 1e-10, 1e-12])
@pytest.mark.parametrize("seed", np.arange(NUM_SEEDS))
def test_edge_cases(eps, seed):
    """
    When y = -x, the denominator in the equation is 0. This test
    checks if the implementation handles this case correctly.
    """

    torch.manual_seed(seed)
    if seed == 0:
        origin = torch.tensor([[1., 0.]], dtype=DTYPE)
    else:
        origin = torch.randn(1, 2, dtype=DTYPE)

    target = -origin
    target[0, 1] = eps

    origin = origin / origin.norm(dim=-1, keepdim=True)
    target = target / target.norm(dim=-1, keepdim=True)

    Q = bivector_rotation(origin, target)
    Q_ref = ref_impl_bivector_rotation_normalized(origin, target)

    _assert_allclose(Q, Q_ref)

    rotated = torch.einsum("...ij,...j->...i", Q, origin)
    _assert_allclose(rotated, target, atol=1e-6, rtol=1e-6)

    QQt = torch.einsum("...ij,...kj->...ik", Q, Q)
    I = torch.eye(2, dtype=DTYPE).expand(1, -1, -1)
    _assert_allclose(QQt, I, atol=1e-5, rtol=1e-5)


def test_larger_tensors():

    origin = torch.randn(3, 4, 5, 6, 3, dtype=DTYPE)
    target = torch.randn(3, 4, 5, 6, 3, dtype=DTYPE)

    origin = origin / origin.norm(dim=-1, keepdim=True)
    target = target / target.norm(dim=-1, keepdim=True)

    Q = bivector_rotation(origin, target)
    Q_ref = ref_impl_bivector_rotation_normalized(origin, target)

    _assert_allclose(Q, Q_ref)

    rotated = torch.einsum("...ij,...j->...i", Q, origin)
    _assert_allclose(rotated, target, atol=1e-4, rtol=1e-4)

    QQt = torch.einsum("...ij,...kj->...ik", Q, Q)
    I = torch.eye(3, dtype=DTYPE).expand(3, 4, 5, 6, -1, -1)
    _assert_allclose(QQt, I, atol=1e-4, rtol=1e-4)


def test_not_normalized():
    origin = torch.randn(1, 2, dtype=DTYPE)
    target = torch.randn(1, 2, dtype=DTYPE)

    origin_norm = origin.norm(dim=-1)
    target_norm = target.norm(dim=-1)

    Q = bivector_rotation(origin, target)

    # note: equivalent to
    # rotated = (Q @ origin.unsqueeze(-1)).squeeze(-1)
    rotated = torch.einsum("...ij,...j->...i", Q, origin)

    _assert_allclose(rotated, target, atol=1e-6, rtol=1e-6)

    QQt = torch.einsum("...ij,...kj->...ik", Q, Q)
    I = torch.eye(2, dtype=DTYPE).expand(1, -1,
                                         -1) * (target_norm / origin_norm)**2
    _assert_allclose(QQt, I, atol=1e-5, rtol=1e-5)


def _test_paired_transform(Q_true,
                           batch_size=None,
                           seed=None,
                           x=None,
                           y=None,
                           x_prime=None,
                           y_prime=None):

    if x is None or y is None or x_prime is None or y_prime is None:
        # get x, y
        x, y = _make_random_vectors(
            batch_size,
            dim=Q_true.shape[-1],
            seed=seed,
        )
        x_prime = torch.einsum("...ij,...j->...i", Q_true, x)
        y_prime = torch.einsum("...ij,...j->...i", Q_true, y)

        assert are_independent([
            x, x_prime
        ]), f"x and x_prime are not linearly independent for Q={Q_true}"
        assert are_independent([
            y, y_prime
        ]), f"y and y_prime are not linearly independent for Q={Q_true}"

    Qx = bivector_rotation(origin=x, target=x_prime)
    Qy = bivector_rotation(origin=y, target=y_prime)

    # compute determinant of Qx, Qy
    det_Qx = torch.det(Qx)
    det_Qy = torch.det(Qy)
    print(f"\n\ndet_Qx: {det_Qx}, \ndet_Qy: {det_Qy}\n")
    assert torch.allclose(
        det_Qx,
        torch.ones_like(det_Qx)), f"Determinant of Qx is not 1: {det_Qx}"
    assert torch.allclose(
        det_Qy,
        torch.ones_like(det_Qy)), f"Determinant of Qy is not 1: {det_Qy}"

    x_prime_pred = torch.einsum("...ij,...j->...i", Qx, x)
    y_prime_pred = torch.einsum("...ij,...j->...i", Qy, y)

    _assert_allclose(x_prime_pred,
                     x_prime,
                     err_msg=f"x_prime is not equal to Qx @ x")
    _assert_allclose(y_prime_pred,
                     y_prime,
                     err_msg=f"y_prime is not equal to Qy @ y")

    # to check if Qx can be used to predict y_prime we first need to project y into the plane spanned by x, x_prime

    P_xxp = subspace_projection_matrix(x, x_prime)
    y_proj = torch.einsum("...ij,...j->...i", P_xxp, y)
    y_prime_proj = torch.einsum("...ij,...j->...i", P_xxp, y_prime)

    # now we can check if Qx @ y_proj is close to y_prime_proj
    y_prime_pred_Qx = torch.einsum("...ij,...j->...i", Qx, y_proj)
    _assert_allclose(y_prime_pred_Qx,
                     y_prime_proj,
                     err_msg=f"y_prime_pred_Qx is not equal to y_prime_proj")

    # the other way around should also work
    P_yyp = subspace_projection_matrix(y, y_prime)
    x_proj = torch.einsum("...ij,...j->...i", P_yyp, x)
    x_prime_proj = torch.einsum("...ij,...j->...i", P_yyp, x_prime)

    x_prime_pred_Qy = torch.einsum("...ij,...j->...i", Qy, x_proj)
    _assert_allclose(x_prime_pred_Qy,
                     x_prime_proj,
                     err_msg=f"x_prime_pred_Qy is not equal to x_prime_proj")

    # finally we check if computing the bivector in the projected space yields the same results a the bivector matrix of the spanning vectors
    Qy_projx = bivector_rotation(y_proj, y_prime_proj)
    Qx_projy = bivector_rotation(x_proj, x_prime_proj)

    _assert_allclose(Qy_projx, Qy)
    _assert_allclose(Qx_projy, Qx)


# @pytest.mark.parametrize("seed", [0])
# @pytest.mark.parametrize("batch_size", [10])
# @pytest.mark.parametrize("dim", [5])
# def test_paired_transform_identity(seed, batch_size, dim):
#     Q_true = torch.eye(dim, dtype=DTYPE).expand(batch_size, -1, -1)
#     _test_paired_transform(Q_true, batch_size, seed)

# @pytest.mark.parametrize("seed", [0])
# @pytest.mark.parametrize("batch_size", [10])
# @pytest.mark.parametrize("dim", [5])
# def test_paired_transform_permutation(seed, batch_size, dim):

#     torch.manual_seed(seed)
#     Q_perm = torch.eye(dim, dtype=DTYPE).expand(batch_size, -1, -1)
#     # sample a random permutation of the dimensions
#     # Sample batch_size many permutations
#     perms = [torch.randperm(dim) for _ in range(batch_size)]
#     for i, perm in enumerate(perms):
#         Q_perm[i, perm, :] = Q_perm[i, :, perm]

#     # compute det of Q_perm
#     det_Q_perm = torch.det(Q_perm)
#     assert torch.allclose(det_Q_perm, torch.ones_like(det_Q_perm))
#     _test_paired_transform(Q_perm, batch_size, seed)

# @pytest.mark.parametrize("seed", [0])
# @pytest.mark.parametrize("batch_size", [10])
# @pytest.mark.parametrize("dim", [5])
# @pytest.mark.parametrize("angle", [5, 20, 45, 80, 90, 120, 180])
# def test_paired_transform_simple_rotation(seed, batch_size, dim, angle):
#     # create a simple rotation matrix, by sampling a random plane (i, j) and an angle theta
#     torch.manual_seed(seed)
#     Q_rots = []

#     for _ in range(batch_size):
#         # sample a random plane (i, j)
#         # Sample two distinct indices for the rotation plane
#         i = torch.randint(0, dim, (1,)).item()
#         j = torch.randint(0, dim, (1,)).item()
#         while j == i:
#             j = torch.randint(0, dim, (1,)).item()
#         # sample a random angle
#         theta = torch.tensor(angle, dtype=DTYPE)
#         theta_rad = theta * torch.pi / 180.0
#         Qrot = torch.eye(dim, dtype=DTYPE)
#         Qrot[i, i] = torch.cos(theta_rad)
#         Qrot[i, j] = -torch.sin(theta_rad)
#         Qrot[j, i] = torch.sin(theta_rad)
#         Qrot[j, j] = torch.cos(theta_rad)
#         Q_rots.append(Qrot)
#     Q_rots = torch.stack(Q_rots, dim=0)

#     # compute det of Q_perm
#     det_Q_rots = torch.det(Q_rots)
#     assert torch.allclose(det_Q_rots, torch.ones_like(det_Q_rots))
#     _test_paired_transform(Q_rots, batch_size, seed + 1)

from pathlib import Path
from groupcl import datasets
from groupcl.models import mixing
from groupcl.models.dynamics import linear_dynamics
from groupcl.models.dynamics import utils
from groupcl.utils.rotation_matrix import MinMaxRotationSampler

from typing import List


def split_seed(seed: int, num_splits: int) -> List[int]:
    import random
    # Set the random seed for reproducibility
    random_generator = random.Random(seed)
    # Generate num_splits random seeds

    return [random_generator.randint(1, 100000) for _ in range(num_splits)]


def get_syn_dataset_config(
    *,
    obs_dim=10,
    latent_dim=5,
    num_samples=1_000,
    num_systems=10,
    seed=42,
    min_angle=0,
    max_angle=0,
):

    dataset_seed, lds_seed, mixing_seed = split_seed(seed, 3)

    dataset = datasets.SyntheticPairedRotationsDataset(
        lazy=True,
        seed=dataset_seed,
        num_samples=num_samples,
        mixing_model=mixing.IdentityMixingModel(
            seed=mixing_seed,
            output_dim=latent_dim,
            input_dim=latent_dim,
        ),
        dynamics_model=linear_dynamics.LinearDynamicsModel(
            seed=lds_seed,
            dim=latent_dim,
            num_systems=num_systems,
            initializer=utils.RotationLDSParameters(
                rotation_sampler=MinMaxRotationSampler(
                    min_angle=min_angle,
                    max_angle=max_angle,
                ),),
        ),
    )
    return dataset.to_dict()


# @pytest.mark.parametrize("seed", [0])
# @pytest.mark.parametrize("angle", [0, 5, 20, 45, 80, 90, 120, 180])
# def test_group_rotation_reconstruction(tmp_path, seed, angle):
#     min_angle, max_angle = angle, angle
#     tmp_data_root = Path(tmp_path) / "data"
#     tmp_data_root.mkdir()
#     # clean up any existing data
#     if tmp_data_root.exists():
#         import shutil
#         shutil.rmtree(tmp_data_root,)
#     tmp_data_root.mkdir()
#     dataset_config = get_syn_dataset_config(
#         obs_dim=10,
#         latent_dim=5,
#         num_samples=1,
#         num_systems=1,
#         seed=seed,
#         min_angle=angle,
#         max_angle=angle,
#     )
#     dataset = datasets.SyntheticPairedRotationsDataset.from_dict(
#         dataset_config,
#         kwargs={
#             "SyntheticPairedRotationsDataset":
#                 dict(
#                     root=tmp_data_root,
#                     force_regenerate=True,
#                 )
#         },
#     )

#     data = dataset.ground_truth_data
#     system_matrices = dataset.dynamics_model.A
#     # for sample_idx in range(data.num_samples):
#     x = data.latents.x.to(DTYPE)
#     y = data.latents.y.to(DTYPE)
#     x_prime = data.latents.x_prime.to(DTYPE)
#     y_prime = data.latents.y_prime.to(DTYPE)
#     Q_gt = system_matrices[data.system_idx].to(DTYPE)

#     _test_paired_transform(Q_gt, x=x, y=y, x_prime=x_prime, y_prime=y_prime)

if __name__ == "__main__":
    pytest.main(
        ["-xvs", __file__, "-k", "test_paired_transform_simple_rotation"])
