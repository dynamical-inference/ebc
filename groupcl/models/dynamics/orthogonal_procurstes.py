from config_dataclass import torch_dataclass, config_field
from typing import Tuple, Literal, Optional

import torch
from jaxtyping import Float
from jaxtyping import Integer
from jaxtyping import jaxtyped
from torch import Tensor
from jaxtyping._typeguard import typechecked

from groupcl.models.dynamics.base import BaseDynamicsModel

from groupcl.utils.bivector import bivector_rotation, subspace_projection_matrix, check_independent

from contextlib import contextmanager


@contextmanager
def maybe_nograd(condition: bool):
    """Context manager that disables gradient calculation when condition is True."""
    if condition:
        with torch.no_grad():
            yield
    else:
        yield


@jaxtyped(typechecker=typechecked)
def orthogonal_procrustes_svd(
    X: Float[Tensor, "num_samples dim"],
    X_prime: Float[Tensor, "num_samples dim"],
) -> Float[Tensor, "dim dim"]:
    """
    Computes the rotation matrix that best aligns X with X_prime using SVD.
    i.e. computes R such that x_prime = R @ x
    """
    XtX_prime = X.T @ X_prime
    U, _, Vh = torch.linalg.svd(XtX_prime)
    R = U @ Vh
    if torch.det(R) < 0:
        Vh[-1] = -Vh[-1]
        R = U @ Vh
    return R


@jaxtyped(typechecker=typechecked)
def solve_linear_lstsq(
    X: Float[Tensor, "num_samples dim"],
    X_prime: Float[Tensor, "num_samples dim"],
) -> Float[Tensor, "dim dim"]:
    """
    Solves the linear equation x_prime = R @ x for R using least squares.
    
    This method uses torch.linalg.lstsq to find the solution without
    enforcing orthogonality constraints.
    """
    # Solve for R in X_prime = X @ R.T
    R_t = torch.linalg.lstsq(X, X_prime).solution
    R = R_t.T

    U, _, Vh = torch.linalg.svd(R)
    R = U @ Vh
    if torch.det(R) < 0:
        Vh[-1] = -Vh[-1]
        R = U @ Vh

    return R


@jaxtyped(typechecker=typechecked)
def solve_linear_direct(
    X: Float[Tensor, "num_samples dim"],
    X_prime: Float[Tensor, "num_samples dim"],
) -> Float[Tensor, "dim dim"]:
    """
    Solves the linear equation x_prime = R @ x for R using direct solve.
    
    This method uses torch.linalg.solve and requires X.T @ X to be invertible.
    No orthogonality constraints are enforced.
    """
    # Compute X.T @ X
    XtX = X.T @ X

    # Compute X.T @ X_prime
    XtX_prime = X.T @ X_prime

    # Solve for R in (X.T @ X) @ R.T = X.T @ X_prime
    R_t = torch.linalg.solve(XtX, XtX_prime)
    R = R_t.T

    U, _, Vh = torch.linalg.svd(R)
    R = U @ Vh
    if torch.det(R) < 0:
        Vh[-1] = -Vh[-1]
        R = U @ Vh

    return R


@jaxtyped(typechecker=typechecked)
@torch_dataclass
class OrthogonalProcrustesModel(BaseDynamicsModel):

    dim: int = config_field(default=2)
    num_systems: int = config_field(default=1)
    no_grad_fit: bool = config_field(default=True)
    fit_method: Literal["svd", "lstsq", "direct"] = config_field(default="svd")

    def __lazy_post_init__(self):
        super().__lazy_post_init__()
        self.R = torch.nn.Parameter(
            torch.eye(self.dim).unsqueeze(0).repeat(self.num_systems, 1, 1))

    @jaxtyped(typechecker=typechecked)
    def fit_parameters(self,
                       x: Float[Tensor, "batch dim"],
                       x_prime: Float[Tensor, "batch dim"],
                       system_idx: Integer[Tensor, "batch"],
                       no_grad: Optional[bool] = None):

        if no_grad is None:
            no_grad = self.no_grad_fit

        with maybe_nograd(no_grad):
            unique_system_idx = torch.unique(system_idx)
            for system_id in unique_system_idx:
                system_mask = system_idx == system_id
                system_X = x[system_mask]
                system_X_prime = x_prime[system_mask]

                # check if we have enough data to fit the model
                if system_X.shape[0] < self.dim:
                    raise ValueError(
                        f"Not enough data to fit the model for system {system_id}, need at least {self.dim} samples but got {system_X.shape[0]}"
                    )

                if self.fit_method == "svd":
                    self.R[system_id] = orthogonal_procrustes_svd(
                        system_X,
                        system_X_prime,
                    )
                elif self.fit_method == "lstsq":
                    self.R[system_id] = solve_linear_lstsq(
                        system_X,
                        system_X_prime,
                    )
                elif self.fit_method == "direct":
                    self.R[system_id] = solve_linear_direct(
                        system_X,
                        system_X_prime,
                    )

    @jaxtyped(typechecker=typechecked)
    def forward(
        self,
        x: Float[Tensor, "batch {self.dim}"],
        system_idx: Optional[Integer[Tensor, "batch"]] = None,
        **kwargs,
    ) -> Float[Tensor, "batch {self.dim}"]:
        """
        Computes the orthogonal procrustes transformation of y.
        """

        if system_idx is None:
            assert self.num_systems == 1, "system_idx must be provided when num_systems > 1"
            system_idx = torch.zeros(x.shape[0], dtype=torch.long)

        R = self.R[system_idx]
        return torch.einsum("...ij,...j->...i", R, x)
