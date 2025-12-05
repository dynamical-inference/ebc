import pytest

import torch
import numpy as np

from groupcl import criterions
from groupcl.models.groups import (GroupDynamicsModel, LeastSquaresSolver,
                                   DirectSolveSystemSolver, ProcustesSVDSolver,
                                   SystemSolver)
from groupcl.utils import datatypes as dt
from typing import List, Tuple, Callable, Optional
import math
from scipy.stats import ortho_group

ATOL = 3e-5
RTOL = 3e-5
NUM_SEEDS = 3
DTYPE = torch.float32


def _assert_allclose(a, b, atol=ATOL, rtol=RTOL):
    if isinstance(a, torch.Tensor):
        a = a.detach().numpy()
    if isinstance(b, torch.Tensor):
        b = b.detach().numpy()
    np.testing.assert_allclose(
        a,
        b,
        atol=atol,
        rtol=rtol,
        verbose=True,
    )


def generate_random_matrix(dim: int,
                           orthogonal: bool = True,
                           seed: Optional[int] = None) -> torch.Tensor:
    if seed is not None:
        torch.manual_seed(seed)

    if orthogonal:
        # Generate a random matrix and use QR decomposition to get an orthogonal matrix

        A = torch.tensor(ortho_group.rvs(dim, random_state=seed), dtype=DTYPE)
        # assert abs(torch.det(Q)) - 1 < 1e-3
        dets = torch.linalg.det(A)
        assert torch.allclose(torch.abs(dets), torch.ones_like(dets))
        return A
    else:
        while True:
            # Generate a random invertible matrix
            A = torch.randn(dim, dim, dtype=DTYPE)
            # Check if it's invertible (non-zero determinant)
            if abs(torch.det(A)) > 1e-1:
                # Check it's not accidentally orthogonal
                is_orthogonal = torch.allclose(A @ A.T,
                                               torch.eye(dim, dtype=DTYPE),
                                               atol=1e-2)
                if not is_orthogonal:
                    return A


def generate_linear_system_data(
    num_systems: int,
    num_samples: int,
    dim: int,
    orthogonal: bool = True,
    seed: Optional[int] = None
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if seed is not None:
        torch.manual_seed(seed)

    # Generate random system matrices
    A = torch.stack([
        generate_random_matrix(dim,
                               orthogonal,
                               seed=seed + i if seed is not None else None)
        for i in range(num_systems)
    ])

    # Generate random input data
    X = torch.randn(num_systems, num_samples, dim, dtype=DTYPE)
    # Generate output data
    X_prime = torch.einsum("...ij,...kj->...ki", A, X)

    return X, X_prime, A


@pytest.mark.parametrize("num_negatives", [2**11])
@pytest.mark.parametrize("dim", [5, 12])
@pytest.mark.parametrize("num_systems", [128])
@pytest.mark.parametrize("num_samples", [10, 20])
@pytest.mark.parametrize("orthogonal_A", [
    True,
    False,
])
@pytest.mark.parametrize(
    "orthogonal_L",
    [
        True,
        # False,
    ])
@pytest.mark.parametrize("system_solver", [
    LeastSquaresSolver(),
    DirectSolveSystemSolver(),
    ProcustesSVDSolver(),
])
@pytest.mark.parametrize("seed", [0])
def test_loss_minimizer(num_negatives, dim, num_systems, num_samples,
                        orthogonal_A, orthogonal_L, system_solver, seed):

    # Skip test if num_samples is smaller than dim
    # This is necessary because solving the linear system requires at least as many samples as dimensions
    if num_samples < dim:
        pytest.skip(f"Skipping test: num_samples ({num_samples}) < dim ({dim})")
    seed += 1
    # for any given x, x_prime that is generated via x_prime = A @ x,
    # the loss should be minimized when we pass L @ x, L @ x_prime to the loss function instead of x, x_prime
    x, x_prime, A = generate_linear_system_data(
        num_systems=num_systems,
        num_samples=num_samples,
        dim=dim,
        orthogonal=orthogonal_A,
        seed=seed,
    )
    negatives = torch.randn(
        num_negatives,
        dim,
        dtype=DTYPE,
    )

    # sample and apply indeterminacy matrix L
    L = generate_random_matrix(dim=dim, orthogonal=orthogonal_L, seed=seed * 13)
    x_perturbed = torch.einsum("...ij,...kj->...ki", L, x)
    x_prime_perturbed = torch.einsum("...ij,...kj->...ki", L, x_prime)
    negatives_perturbed = torch.einsum("...ij,...kj->...ki", L, negatives)

    # compute loss
    loss_fn = criterions.GCLLoss(
        group_model=GroupDynamicsModel(system_solver=system_solver),
        criterion=criterions.MseInfoNCE(),
    )

    gt_loss, gt_metrics = loss_fn(embeddings=dt.ContrastiveGroupBatch(
        positives=dt.SinglePairedGroupData(
            x=x,
            x_prime=x_prime,
        ),
        negatives=dt.SingleData(x=negatives),
    ))

    # compute loss with perturbed data
    loss, metrics = loss_fn(embeddings=dt.ContrastiveGroupBatch(
        positives=dt.SinglePairedGroupData(
            x=x_perturbed,
            x_prime=x_prime_perturbed,
        ),
        negatives=dt.SingleData(x=negatives_perturbed),
    ))

    # in both cases we should get equally good approximations of the system matrix
    # 1. check mse_x_prime_pred
    _assert_allclose(metrics["mse_x_prime_pred"],
                     gt_metrics["mse_x_prime_pred"],
                     atol=ATOL)
    # 2. check loss_align
    _assert_allclose(metrics["loss_align"], gt_metrics["loss_align"], atol=ATOL)

    # 3. the uniformity loss should also be the same, given enough negatives
    _assert_allclose(metrics["loss_uniformity"],
                     gt_metrics["loss_uniformity"],
                     atol=ATOL)

    # finally, this should also mean the losses are the same
    _assert_allclose(loss, gt_loss, atol=ATOL)
