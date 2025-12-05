import torch
import numpy as np
from typing import Union, Tuple, List

from jaxtyping import Float
from jaxtyping import Bool
from jaxtyping import jaxtyped
from torch import Tensor
from jaxtyping._typeguard import typechecked


@jaxtyped(typechecker=typechecked)
def bivector_rotation(
    origin: Float[Tensor, "... n"],
    target: Float[Tensor, "... n"],
    eps: float = 1e-8,
) -> Float[Tensor, "... n n"]:
    """Contruct a rotation matrix for rotating origin onto target

    Given an origin vector ``x`` and a target vector ``y``, this function returns an orthogonal
    matrix ``Q`` such that ``Qx = y``. The matrix is computed as:

    .. math::

        Q = I - (x + y)(x + y)^T / (1 + x^T y) + 2 yx^T 

    This function supports batched computation, given vectors of size ``(..., n)`` and ``(..., n)``,
    the return is ``(..., n, n)``.

    Args:
        origin: (..., n) tensor of vectors
        target: (..., n) tensor of vectors

    Returns:
        (..., n, n) tensor of rotation matrices
    """

    origin_norm = origin.norm(dim=-1, keepdim=True)
    target_norm = target.norm(dim=-1, keepdim=True)

    origin = origin / origin_norm
    target = target / target_norm

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
    return Q * (target_norm / origin_norm).unsqueeze(-1)


@jaxtyped(typechecker=typechecked)
def subspace_projection_matrix(
    basis1: Float[Tensor, "... dim"],
    basis2: Float[Tensor, "... dim"],
) -> Float[Tensor, "... dim dim"]:
    """
    Computes the orthogonal projection matrix onto the subspace spanned by two vectors.

    Args:
        basis1: A 1DTensor representing the first basis vector.
        basis2: A 1DTensor representing the second basis vector.
                Must have the same dimension and dtype as basis1.

    Returns:
        A 2DTensor representing the projection matrix P.

    """
    if not are_independent([basis1, basis2]):
        raise ValueError("Basis vectors must be linearly independent")

    # --- Construct Matrix A ---
    # Ensure vectors are column vectors for matrix construction
    # Stack them column-wise to form matrix A
    # Use .unsqueeze(1) instead of .view(-1, 1) for clarity with 1D input
    A = torch.stack((basis1, basis2), dim=-1)  # Shape becomes [dimension, 2]

    # --- Compute Projection Matrix P = A @ inv(A^T @ A) @ A^T ---
    At = A.transpose(-1, -2)  # Shape [2, dimension]
    AtA = At @ A  # Shape [2, 2]

    # Check for near-zero determinant before inverting (optional but good practice)
    # det = torch.linalg.det(AtA)
    # if torch.isclose(det,Tensor(0.0, dtype=AtA.dtype)):
    #     raise torch.linalg.LinAlgError("Matrix A^T A is singular, vectors might be linearly dependent.")

    # Compute the inverse of A^T A.
    # torch.linalg.inv is generally preferred over torch.inverse
    # This will raise LinAlgError if AtA is singular
    try:
        AtA_inv = torch.linalg.inv(AtA)  # Shape [2, 2]
    except torch.linalg.LinAlgError as e:
        raise torch.linalg.LinAlgError(
            f"Failed to compute inverse of A^T A. Basis vectors might be linearly dependent. Original error: {e}"
        ) from e

    # Compute the projection matrix P
    P = A @ AtA_inv @ At  # Shape [dimension, dimension]

    return P


@jaxtyped(typechecker=typechecked)
def check_independent(
    vectors: Union[
        List[Float[Tensor, "... dim"]],
        Float[Tensor, "... num_vectors dim"],
    ]
) -> Bool[Tensor, "... num_vectors"]:
    """
    Check if a set of vectors are linearly independent.
    
    Args:
        vectors (list or Tensor): List of 1D tensors or a 2D tensor of shape (num_vectors, dimension)
        
    Returns:
        bool: True if vectors are independent, False otherwise
    """
    # Stack vectors into a matrix if it's a list
    if isinstance(vectors, list):
        # shape should be (..., num_vectors, dimension)
        matrix = torch.stack(vectors, dim=-2)
    else:
        matrix = vectors

    # Compute the rank of the matrix
    ranks = torch.linalg.matrix_rank(matrix.float())

    # Number of vectors
    num_vectors = matrix.shape[-2]

    # If rank equals number of vectors, they're independent
    return ranks == num_vectors


@jaxtyped(typechecker=typechecked)
def are_independent(
    vectors: Union[
        List[Float[Tensor, "... dim"]],
        Float[Tensor, "... num_vectors dim"],
    ]
) -> bool:
    """
    Check if a set of vectors are linearly independent.
    
    Args:
        vectors (list or Tensor): List of 1D tensors or a 2D tensor of shape (num_vectors, dimension)
        
    Returns:
        bool: True if vectors are independent, False otherwise
    """
    # If rank equals number of vectors, they're independent
    return torch.all(check_independent(vectors)).item()


@jaxtyped(typechecker=typechecked)
def in_subspace(
    vector: Float[torch.Tensor, "... n"],
    span_a: Float[torch.Tensor, "... n"],
    span_b: Float[torch.Tensor, "... n"],
) -> bool:
    """
    Check if a vector is in the span of two basis vectors.
    """
    assert are_independent([span_a, span_b
                           ]), "Basis vectors must be linearly independent"
    # Solve the linear system to find coefficients
    # We want to find coefficients a, b such that vector = a*span_a + b*span_b
    # This is equivalent to solving the system:
    # [span_a_x span_b_x] [a] = [vector_x]
    # [span_a_y span_b_y] [b]   [vector_y]
    # ... and so on for all dimensions

    # Set up the system as a least squares problem
    A = torch.stack([span_a, span_b], dim=1)  # Shape: [dim, 2]

    # Solve the least squares problem
    try:
        # Use torch.linalg.lstsq for a more stable solution
        solution, residuals, rank, singular_values = torch.linalg.lstsq(
            A,
            vector.unsqueeze(1),
            driver='gelsd',
        )
        solution = solution.squeeze()

        # Check if the residual is close to zero
        # If the vector is in the span, the residual should be very small
        reconstructed = span_a * solution[0] + span_b * solution[1]
    except Exception:
        # Fall back to a simpler approach if lstsq fails
        # Project the vector onto the subspace and check if it's preserved
        A_pseudo_inv = torch.linalg.pinv(A)  # Shape: [2, dim]
        coefficients = A_pseudo_inv @ vector  # Shape: [2]
        reconstructed = A @ coefficients  # Shape: [dim]

    # Check if the reconstruction error is small
    error = torch.norm(vector - reconstructed)
    return error < 1e-6 * torch.norm(vector)
