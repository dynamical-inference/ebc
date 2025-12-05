from abc import ABC, abstractmethod
from contextlib import contextmanager
from typing import Optional, Tuple

import torch
from config_dataclass import Configurable, config_dataclass, config_field
from jaxtyping import Float, jaxtyped
from jaxtyping._typeguard import typechecked
from torch import Tensor

from groupcl.utils.complex import (
    complex_matrix_to_real,
    real_data_to_complex,
    solve_complex_matrix_system,
)


@contextmanager
def maybe_nograd(condition: bool):
    """Context manager that disables gradient calculation when condition is True."""
    if condition:
        with torch.no_grad():
            yield
    else:
        yield


@jaxtyped(typechecker=typechecked)
@config_dataclass
class SystemSolver(Configurable, ABC):

    orthogonal_solution: bool = config_field(default=True)

    def solve(
        self,
        X: Float[Tensor, "num_systems num_samples dim"],
        X_prime: Float[Tensor, "num_systems num_samples dim"],
    ) -> Float[Tensor, "num_systems dim dim"]:

        systems = None
        # try batched implementation solver_systems first
        # if that fails, try for loop over solve_system
        try:
            systems = self._solve_systems(X, X_prime)
        except NotImplementedError:
            systems = torch.stack(
                [
                    self._solve_system(X[i], X_prime[i])
                    for i in range(X.shape[0])
                ],
                dim=0,
            )

        if self.orthogonal_solution:
            # assert all systems are orthogonal
            # check abs(det(S)) = 1 for all systems
            dets = torch.linalg.det(systems)
            assert torch.allclose(
                torch.abs(dets), torch.ones_like(dets), atol=1e-3
            ), f"det(S) is not 1 for all systems, dets: {dets.mean()}, dets.max(): {dets.max()}, dets.min(): {dets.min()}"

            # check if S@S.T is the identity matrix
            should_be_eye = systems @ systems.transpose(-1, -2)
            assert torch.allclose(
                should_be_eye,
                torch.eye(X.shape[-1], device=X.device).unsqueeze(0),
                atol=1e-3,
            ), "S@S.T is not the identity matrix"

        return systems

    @abstractmethod
    def _solve_system(
        self,
        X: Float[Tensor, "num_samples dim"],
        X_prime: Float[Tensor, "num_samples dim"],
    ) -> Float[Tensor, "dim dim"]:
        pass

    @abstractmethod
    def _solve_systems(
        self,
        X: Float[Tensor, "num_systems num_samples dim"],
        X_prime: Float[Tensor, "num_systems num_samples dim"],
    ) -> Float[Tensor, "num_systems dim dim"]:
        pass


@jaxtyped(typechecker=typechecked)
@config_dataclass
class ProcustesSVDSolver(SystemSolver):
    """Solves for the orthogonal transformation matrix using Procrustes analysis with SVD."""

    def __post_init__(self):
        # always set orthogonal_solution to True
        self.orthogonal_solution = True

    def _solve_system(
        self,
        X: Float[Tensor, "num_samples dim"],
        X_prime: Float[Tensor, "num_samples dim"],
    ) -> Float[Tensor, "dim dim"]:
        """Compute the orthogonal transformation matrix using SVD."""
        X_primeTX = X_prime.T @ X
        U, _, Vh = torch.linalg.svd(X_primeTX)
        R = U @ Vh

        return R

    def _solve_systems(
        self,
        X: Float[Tensor, "num_systems num_samples dim"],
        X_prime: Float[Tensor, "num_systems num_samples dim"],
    ) -> Float[Tensor, "num_systems dim dim"]:
        """Solve for multiple systems in parallel using SVD."""
        X_primeTX = X_prime.transpose(-1, -2) @ X
        U, _, Vh = torch.linalg.svd(X_primeTX)
        R = U @ Vh

        return R


@jaxtyped(typechecker=typechecked)
@config_dataclass
class LeastSquaresSolver(SystemSolver):
    """Solves for the transformation matrix using least squares method."""

    def _solve_system(
        self,
        X: Float[Tensor, "num_samples dim"],
        X_prime: Float[Tensor, "num_samples dim"],
    ) -> Float[Tensor, "dim dim"]:
        """Compute the transformation matrix using least squares."""
        # Solve for R in X_prime = X @ R.T
        R_t = torch.linalg.lstsq(X, X_prime).solution
        R = R_t.T

        if self.orthogonal_solution:
            # project R onto the space of orthogonal matrices
            U, _, Vh = torch.linalg.svd(R)
            R = U @ Vh

        return R

    def _solve_systems(
        self,
        X: Float[Tensor, "num_systems num_samples dim"],
        X_prime: Float[Tensor, "num_systems num_samples dim"],
    ) -> Float[Tensor, "num_systems dim dim"]:
        """Solve for multiple systems in parallel."""
        # Solve for R in X_prime = X @ R.T
        R_t = torch.linalg.lstsq(X, X_prime).solution
        R = R_t.transpose(-1, -2)

        if self.orthogonal_solution:
            # project R onto the space of orthogonal matrices
            U, _, Vh = torch.linalg.svd(R)
            R = U @ Vh
        return R


@jaxtyped(typechecker=typechecked)
@config_dataclass
class DirectSolveSystemSolver(SystemSolver):
    """Solves for the transformation matrix using direct linear system solving."""

    def __post_init__(self):
        # always set orthogonal_solution to False
        self.orthogonal_solution = False

    def _solve_system(
        self,
        X: Float[Tensor, "num_samples dim"],
        X_prime: Float[Tensor, "num_samples dim"],
    ) -> Float[Tensor, "dim dim"]:
        """Compute the transformation matrix using direct solve."""
        # Compute X.T @ X
        XtX = X.T @ X

        # Compute X.T @ X_prime
        XtX_prime = X.T @ X_prime

        # Solve for R in (X.T @ X) @ R.T = X.T @ X_prime
        R_t = torch.linalg.solve(XtX, XtX_prime)
        R = R_t.T

        return R

    def _solve_systems(
        self,
        X: Float[Tensor, "num_systems num_samples dim"],
        X_prime: Float[Tensor, "num_systems num_samples dim"],
    ) -> Float[Tensor, "num_systems dim dim"]:
        """Solve for multiple systems in parallel."""
        raise NotImplementedError(
            "Batch solving not implemented for direct solve method")


@jaxtyped(typechecker=typechecked)
@config_dataclass
class ComplexLeastSquaresSolver(LeastSquaresSolver):
    """
    Solves for the transformation matrix using least squares in the complex domain.
    
    This solver converts real-valued input data to complex representation, solves
    the linear system using complex arithmetic, and converts the result back to 
    real representation. This can be beneficial for certain types of transformations
    that are more naturally expressed in the complex domain.
    
    Requirements:
    - Input dimension must be even (since real data is converted to complex)
    - The resulting transformation matrix maintains the same API as LeastSquaresSolver
    """

    def __post_init__(self):
        # Complex solver doesn't support orthogonal projection in the current implementation
        # since we're focused on the pure complex arithmetic approach
        self.orthogonal_solution = False

    def _solve_system(
        self,
        X: Float[Tensor, "num_samples dim"],
        X_prime: Float[Tensor, "num_samples dim"],
    ) -> Float[Tensor, "dim dim"]:
        """Compute the transformation matrix using complex least squares."""
        # Validate that dimension is even for complex conversion
        dim = X.shape[-1]
        if dim % 2 != 0:
            raise ValueError(
                f"Input dimension {dim} must be even for complex conversion. "
                f"Consider padding or using LeastSquaresSolver for odd dimensions."
            )

        return self._solve_system(X.unsqueeze(0),
                                  X_prime.unsqueeze(0)).squeeze(0)

    def _solve_systems(
        self,
        X: Float[Tensor, "num_systems num_samples dim"],
        X_prime: Float[Tensor, "num_systems num_samples dim"],
    ) -> Float[Tensor, "num_systems dim dim"]:
        """Solve for multiple systems in parallel using complex arithmetic."""
        # Validate that dimension is even for complex conversion
        dim = X.shape[-1]
        if dim % 2 != 0:
            raise ValueError(
                f"Input dimension {dim} must be even for complex conversion. "
                f"Consider padding or using LeastSquaresSolver for odd dimensions."
            )

        # Convert real data to complex representation
        X_complex = real_data_to_complex(
            X)  # (num_systems, num_samples, dim//2)
        X_prime_complex = real_data_to_complex(
            X_prime)  # (num_systems, num_samples, dim//2)

        # Solve the linear system in complex domain: X_prime = X @ R.T
        R_complex = solve_complex_matrix_system(
            X_complex, X_prime_complex)  # (num_systems, dim//2, dim//2)

        # Convert complex result back to real representation
        R_real = complex_matrix_to_real(R_complex)  # (num_systems, dim, dim)

        return R_real


@jaxtyped(typechecker=typechecked)
@config_dataclass
class GroupDynamicsModel(Configurable):

    no_grad_solve: bool = config_field(default=True)
    system_solver: SystemSolver = config_field(
        default_factory=LeastSquaresSolver)
    dynamics_dim: Optional[int] = config_field(default=None, skip_default=True)

    @jaxtyped(typechecker=typechecked)
    def __call__(
        self,
        ref: Float[Tensor, "batch num_samples dim"],
        ref_prime: Float[Tensor, "batch num_samples dim"],
        target: Float[Tensor, "batch dim"],
        **kwargs,
    ) -> Tuple[Float[Tensor, "batch dim"], Float[Tensor, "batch dim dim"]]:
        """
        Estimates a matrix Q from the references by solving ref_prime = Q @ ref.
        Then applies Q to the target to get the predicted target, returning Q@target.
        """
        with maybe_nograd(self.no_grad_solve):
            if self.dynamics_dim is not None:
                latent_dim = ref.shape[-1]
                ref = ref[:, :, :self.dynamics_dim]
                ref_prime = ref_prime[:, :, :self.dynamics_dim]

            Q = self.system_solver.solve(X=ref, X_prime=ref_prime)

            if self.dynamics_dim is not None:
                # append Block Identity to extend Q to shape latent_dim x latent_dim
                content_dim = latent_dim - self.dynamics_dim
                num_systems = Q.shape[0]
                identity_matrix = torch.eye(content_dim, device=Q.device)
                Q_full = torch.zeros(num_systems,
                                     latent_dim,
                                     latent_dim,
                                     device=Q.device)
                Q_full[:, :self.dynamics_dim, :self.dynamics_dim] = Q
                Q_full[:, self.dynamics_dim:,
                       self.dynamics_dim:] = identity_matrix
                Q = Q_full

        target_pred = torch.einsum("...ij,...j->...i", Q, target)
        return target_pred, Q
