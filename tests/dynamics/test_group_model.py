import pytest
import torch
import numpy as np
from typing import List, Tuple, Callable, Optional
import math
from groupcl.models.groups import (GroupDynamicsModel, LeastSquaresSolver,
                                   DirectSolveSystemSolver, ProcustesSVDSolver,
                                   SystemSolver)
from scipy.stats import ortho_group

ATOL = 3e-5
RTOL = 3e-5
NUM_SEEDS = 3
DTYPE = torch.float32


def _assert_allclose(a, b, atol=ATOL, rtol=RTOL):
    np.testing.assert_allclose(
        a.detach().numpy(),
        b.detach().numpy(),
        atol=atol,
        rtol=rtol,
        verbose=True,
    )


def generate_random_matrix(dim: int,
                           orthogonal: bool = True,
                           seed: Optional[int] = None) -> torch.Tensor:
    """Generate a random matrix that is either orthogonal or just invertible."""
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
    """
    Generate data for a linear dynamical system X_prime = A @ X.
    
    Args:
        num_systems: Number of systems to generate
        num_samples: Number of samples per system
        dim: Dimensionality of the data
        orthogonal: Whether to generate orthogonal transformation matrices
        seed: Random seed
        
    Returns:
        X: Input data of shape (num_systems, num_samples, dim)
        X_prime: Output data of shape (num_systems, num_samples, dim)
        A: System matrices of shape (num_systems, dim, dim)
    """
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


def _test_solver_with_data(solver: SystemSolver, X: torch.Tensor,
                           X_prime: torch.Tensor, A_true: torch.Tensor,
                           orthogonal: bool):
    """Test a solver with the given data."""
    # Solve the system
    A_pred = solver.solve(X, X_prime)

    # Check that the solution is correct
    _assert_allclose(A_pred, A_true)

    # Check that X_prime = A_pred @ X
    X_prime_pred = torch.einsum("...ij,...kj->...ki", A_pred, X)
    _assert_allclose(X_prime_pred, X_prime)

    # Check orthogonality if required
    if orthogonal or solver.orthogonal_solution:
        # Check determinants are close to 1 or -1
        dets = torch.linalg.det(A_pred)
        assert torch.all(
            torch.abs(torch.abs(dets) -
                      1.0) < 1e-3), f"Determinants not close to 1: {dets}"

        # Check A @ A.T = I for all matrices
        should_be_eye = A_pred @ A_pred.transpose(-1, -2)
        _assert_allclose(
            should_be_eye,
            torch.eye(A_pred.shape[-1], dtype=DTYPE).expand_as(should_be_eye))


def _test_batched_vs_nonbatched(solver: SystemSolver, X: torch.Tensor,
                                X_prime: torch.Tensor):
    """Test that batched and non-batched versions give the same results."""
    # Skip if _solve_systems is not implemented
    try:
        # Try to call _solve_systems directly to check if it's implemented
        solver._solve_systems(X[:1], X_prime[:1])
        batched_implemented = True
    except NotImplementedError:
        batched_implemented = False

    if batched_implemented:

        # Solve using batched version
        A_batched = solver._solve_systems(X, X_prime)

        # Solve using non-batched version
        A_nonbatched = torch.stack(
            [solver._solve_system(X[i], X_prime[i]) for i in range(X.shape[0])])

        # Check that results are the same
        _assert_allclose(A_batched, A_nonbatched)


def _test_sample_order_invariance(solver: SystemSolver, X: torch.Tensor,
                                  X_prime: torch.Tensor):
    """Test that shuffling the order of samples doesn't affect the result."""
    # Get original solution
    A_orig = solver.solve(X, X_prime)

    # For each system, shuffle the samples
    A_shuffled_list = []
    for i in range(X.shape[0]):
        # Generate random permutation
        perm = torch.randperm(X.shape[1])

        # Shuffle the samples
        X_shuffled = X[i, perm]
        X_prime_shuffled = X_prime[i, perm]

        # Solve the shuffled system
        A_shuffled = solver._solve_system(X_shuffled, X_prime_shuffled)
        A_shuffled_list.append(A_shuffled)

    A_shuffled = torch.stack(A_shuffled_list)

    # Check that results are the same
    _assert_allclose(A_shuffled, A_orig)


@pytest.mark.parametrize(
    "solver_class",
    [LeastSquaresSolver, DirectSolveSystemSolver, ProcustesSVDSolver])
@pytest.mark.parametrize("orthogonal", [True, False])
@pytest.mark.parametrize("num_systems", [1, 3])
@pytest.mark.parametrize("num_samples", [None, 100])
@pytest.mark.parametrize("dim", [2, 5, 10])
@pytest.mark.parametrize("seed", range(NUM_SEEDS))
def test_system_solver(solver_class, orthogonal, num_systems, num_samples, dim,
                       seed):
    """Test a system solver with various configurations."""
    # Skip incompatible configurations
    if not orthogonal and solver_class == ProcustesSVDSolver:
        pytest.skip("ProcustesSVDSolver only works with orthogonal matrices")

    if num_samples is None:
        num_samples = math.ceil(dim * 1.4)

    # Skip if num_samples < dim (underdetermined system)
    if num_samples < dim:
        pytest.skip(
            f"Skipping underdetermined system with {num_samples} < {dim}")

    # Generate data
    X, X_prime, A_true = generate_linear_system_data(num_systems=num_systems,
                                                     num_samples=num_samples,
                                                     dim=dim,
                                                     orthogonal=orthogonal,
                                                     seed=seed)

    # Create solver
    if solver_class == DirectSolveSystemSolver:
        # DirectSolveSystemSolver always has orthogonal_solution=False
        solver = solver_class()
    else:
        # For other solvers, set orthogonal_solution based on the test parameter
        solver = solver_class(orthogonal_solution=orthogonal)

    # Run tests
    _test_solver_with_data(solver, X, X_prime, A_true, orthogonal)
    _test_batched_vs_nonbatched(solver, X, X_prime)
    _test_sample_order_invariance(solver, X, X_prime)


@pytest.mark.parametrize(
    "solver_class",
    [LeastSquaresSolver, DirectSolveSystemSolver, ProcustesSVDSolver])
@pytest.mark.parametrize("no_grad_solve", [True, False])
@pytest.mark.parametrize("orthogonal", [
    True,
    False,
])
@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("seed", [0])
def test_group_dynamics_model(solver_class, no_grad_solve, orthogonal, dim,
                              seed):
    """Test the GroupDynamicsModel with various solvers."""
    # Skip incompatible configurations
    if not orthogonal and solver_class == ProcustesSVDSolver:
        pytest.skip("ProcustesSVDSolver only works with orthogonal matrices")

    # Generate data
    num_systems = 7
    num_samples = 10

    X, X_prime, A_true = generate_linear_system_data(num_systems=num_systems,
                                                     num_samples=num_samples,
                                                     dim=dim,
                                                     orthogonal=orthogonal,
                                                     seed=seed)

    # Generate random targets
    torch.manual_seed(seed)
    targets = torch.randn(num_systems, dim, dtype=DTYPE)

    # Create solver
    if solver_class == DirectSolveSystemSolver:
        solver = solver_class()
    else:
        solver = solver_class(orthogonal_solution=orthogonal)

    # Create model
    model = GroupDynamicsModel(no_grad_solve=no_grad_solve,
                               system_solver=solver)

    # Compute expected targets
    expected_targets = torch.einsum("...ij,...j->...i", A_true, targets)

    # Compute predicted targets and Q matrices
    with torch.set_grad_enabled(not no_grad_solve):
        predicted_targets, Q = model(ref=X, ref_prime=X_prime, target=targets)

    # Check that predictions match expectations
    _assert_allclose(predicted_targets, expected_targets)

    # Check that Q matrices match the true matrices
    _assert_allclose(Q, A_true)


from groupcl.models import mixing
from groupcl.models import encoder
from groupcl.models import dynamics
from groupcl.models import groups
from groupcl import criterions
from groupcl import solver as solvers
from groupcl import datasets
from groupcl import loader
from groupcl.experiments import Experiment
from groupcl.utils import temperature_scheduler as scheduler
from groupcl.utils import rotation_matrix as rotation_sampler


def get_dataset_config(
    latent_dim=5,
    content_dim=2,
    num_classes=10,
    num_samples=1000,
    num_systems=10,
    seed=123,
):

    lds = dynamics.LinearDynamicsModel(
        seed=seed,
        dim=latent_dim - content_dim,
        num_systems=num_systems,
        initializer=dynamics.RotationLDSParameters(
            rotation_sampler=rotation_sampler.MinMaxRotationSampler(
                min_angle=0,
                max_angle=360,
            ),),
    )

    content_embedding = datasets.HypersphereContentEmbedding(
        num_classes=num_classes,
        dim=content_dim,
        seed=seed,
    )

    dataset = datasets.SyntheticContentDataset(
        content_embedding=content_embedding,
        num_samples=num_samples,
        mixing_model=mixing.IdentityMixingModel(
            output_dim=latent_dim,
            input_dim=latent_dim,
            seed=seed,
        ),
        dynamics_model=lds,
        seed=seed,
    )

    return dataset


@pytest.mark.parametrize("force_content_dim", [True, False])
def test_group_dynamics_model_with_content(force_content_dim):

    latent_dim = 5
    content_dim = 2

    model_dynamics_dim = latent_dim - content_dim if force_content_dim else None
    group_model = groups.GroupDynamicsModel(
        system_solver=groups.LeastSquaresSolver(orthogonal_solution=False),
        dynamics_dim=model_dynamics_dim,
    )

    dataset = get_dataset_config(
        latent_dim=latent_dim,
        content_dim=content_dim,
        num_classes=10,
        num_samples=1000,
        num_systems=10,
        seed=123,
    )

    data_loader = loader.GCLDataLoader(
        batch_size=512,
        samples_per_action=8,
    )

    data_loader.lazy_init(dataset)
    for embeddings in data_loader:
        target = embeddings.positives[:, 0]
        # check correct class embeddings across x and x_prime
        assert torch.allclose(target.x_prime[..., -content_dim:],
                              target.x[..., -content_dim:]), \
            "target and target_prime must have same content embeddings"

        reference = embeddings.positives[:, 1:]

        # check action indices matches for target and reference
        if embeddings.positives.actions_idx is not None:
            target_actions_idx = embeddings.positives.actions_idx[:, :1]
            reference_actions_idx = embeddings.positives.actions_idx[:, 1:]
            assert torch.all(target_actions_idx == reference_actions_idx
                            ), "target and reference action indices must match"

        targets_x_prime_pred, Q = group_model(
            ref=reference.x,
            ref_prime=reference.x_prime,
            target=target.x,
        )

        _assert_allclose(targets_x_prime_pred.cpu(), target.x_prime.cpu())

        # Q should be a matrix composed of two blocks on the diagonal
        # R and I, where
        # Q = [
        #     [R, 0],
        #     [0, I],
        # ]
        # Extract the four blocks of Q
        dynamics_dim = target.x.shape[-1] - content_dim

        # Top-left block (R): dynamics part
        R = Q[:, :dynamics_dim, :dynamics_dim]

        # Top-right block: should be close to zero
        top_right = Q[:, :dynamics_dim, dynamics_dim:]

        # Bottom-left block: should be close to zero
        bottom_left = Q[:, dynamics_dim:, :dynamics_dim]

        # Bottom-right block (I): content part, should be identity
        I = Q[:, dynamics_dim:, dynamics_dim:]

        # Check that off-diagonal blocks are close to zero
        assert torch.allclose(top_right, torch.zeros_like(top_right), atol=1e-5), \
            "Top-right block should be close to zero"
        assert torch.allclose(bottom_left, torch.zeros_like(bottom_left), atol=1e-5), \
            "Bottom-left block should be close to zero"

        # Check that bottom-right block is close to identity
        identity = torch.eye(content_dim,
                             device=I.device).expand(I.shape[0], -1, -1)
        assert torch.allclose(I, identity, atol=1e-5), \
            "Bottom-right block should be close to identity matrix"


if __name__ == "__main__":
    pytest.main([__file__])
