import math
import random
from abc import ABC
from abc import abstractmethod
from dataclasses import dataclass
from dataclasses import field
from typing import List, Tuple, Union, Optional

import numpy as np
import torch
from jaxtyping import Float
from jaxtyping import jaxtyped
from torch import Tensor
from jaxtyping._typeguard import typechecked

from config_dataclass import Configurable, torch_dataclass, config_field, check_initialized


@jaxtyped(typechecker=typechecked)
@torch_dataclass
class RotationSampler(Configurable, ABC):
    """Base class for sampling rotation matrices."""

    verbose: bool = field(default=False)

    def sample(self, dim: int) -> Float[Tensor, "dim dim"]:
        """Sample a random rotation matrix."""
        assert dim >= 2, "Dimension must be at least 2"
        rotation_matrix = self._sample(dim)
        self._verify_rotation_matrix(rotation_matrix)
        return rotation_matrix

    @abstractmethod
    def _sample(self, dim: int) -> Float[Tensor, "dim dim"]:
        """Sample a random rotation matrix."""

    def _verify_rotation_matrix(self, matrix: Tensor) -> None:
        """Verify that the matrix is indeed a rotation matrix."""
        dim = matrix.shape[-1]
        # Check orthogonality: R^T R = I
        identity = torch.eye(dim, device=matrix.device)
        is_orthogonal = torch.allclose(matrix @ matrix.T, identity, atol=1e-6)
        assert is_orthogonal, "M@M.T is not the identity matrix, matrix is not orthogonal"

        # Check determinant is 1 (proper rotation)
        det = torch.det(matrix)
        is_proper = torch.allclose(det, torch.tensor(1.0), atol=1e-6)
        assert is_proper, f"Matrix determinant is not 1 (got {det}), matrix is not a proper rotation"


@jaxtyped(typechecker=typechecked)
@torch_dataclass
class MinMaxRotationSampler(RotationSampler):
    """
    Samples rotation matrices by applying random rotations to all pairs of dimensions.
    For each dimension pair, samples an angle between min_angle and max_angle degrees
    and creates a rotation matrix. The final matrix is the product of all indivudal 2d-rotations.
    """

    min_angle: Union[int, float] = config_field(
        default=0, help="Minimum rotation angle in degrees")
    max_angle: Union[int, float] = config_field(
        default=5, help="Maximum rotation angle in degrees")

    def __post_init__(self):
        super().__post_init__()
        assert self.max_angle >= self.min_angle, "max_angle must be >= min_angle"

    @jaxtyped(typechecker=typechecked)
    def _sample(self, dim: int) -> Float[Tensor, "dim dim"]:
        """Sample a random rotation matrix."""
        W = torch.eye(dim)
        thetas = []

        for d1 in range(dim):
            for d2 in range(d1 + 1, dim):
                # Random angle between min_angle and max_angle
                theta_degree = random.uniform(self.min_angle, self.max_angle)
                # Random direction of rotation
                if random.choice([True, False]):
                    theta_degree = -theta_degree

                if self.verbose:
                    print(f"rotating {d1} and {d2} by {theta_degree} degrees")

                # Convert to radians and create rotation matrix
                theta = math.radians(theta_degree)
                Wrot = torch.eye(dim)
                Wrot[d1, d1] = math.cos(theta)
                Wrot[d1, d2] = -math.sin(theta)
                Wrot[d2, d1] = math.sin(theta)
                Wrot[d2, d2] = math.cos(theta)

                W = W @ Wrot
                thetas.append(theta_degree)

        if self.verbose:
            print(
                f"Created random matrix with mean absolute angle of {np.mean(np.abs(thetas)):.2f} degrees"
            )
        return W


def create_rotation_matrix(dimensions, rotations):
    """
    Create an N-dimensional rotation matrix.
    Args:
        dimensions (int): 
            The dimensionality of the space.
        rotations (List[Tuple[int, int, float]] or List[int]): 
            Either a list of tuples, each containing two dimensions (0-indexed) and an angle in degrees. 
            Or a list of integers, specifying the degrees of rotation around each axis (in order). 
    
    Returns:
        np.ndarray: N-dimensional rotation matrix.
    Examples:
    ----- Specify rotations as a list of tuples -----
    rotations_5d = [(0, 1, 45), (1, 2, 30), (2, 3, 30), (3, 4, 60), (4, 0, 60)]
    rotation_matrix_5d = create_rotation_matrix(5, rotations_5d)
    
    ----- Specify rotations as a list of ints -----
    rotations_5d = [45, 30, 30, 60, 60]
    rotation_matrix_5d = create_rotation_matrix(5, rotations_5d)
    """
    # Initialize the rotation matrix as an identity matrix
    rotation_matrix = np.identity(dimensions)

    if isinstance(rotations[0], float) or isinstance(rotations[0], int):
        tmp_rotations = []
        for i, degree in enumerate(rotations):
            assert isinstance(degree, float) or isinstance(
                degree, int
            ), "Must specify a list of numbers or list of tuples, but not both."
            tmp_rotations.append((i, (i + 1) % dimensions, degree))
        rotations = tmp_rotations
    else:
        assert all([
            isinstance(rotation, tuple) for rotation in rotations
        ]), "Must specify a list of numbers or list of tuples, but not both."

    print("using rotations", rotations)
    # Convert angles from degrees to radians
    rotations = [(a, b, np.radians(angle)) for a, b, angle in rotations]
    for a, b, angle in rotations:
        # Validate the dimension indices
        if a >= dimensions or b >= dimensions or a == b:
            raise ValueError("Invalid rotation plane dimensions specified.")

        # Create a 2D rotation matrix for the current plane
        sin, cos = np.sin(angle), np.cos(angle)
        plane_rotation = np.identity(dimensions)
        plane_rotation[[a, a, b, b], [a, b, a, b]] = [cos, -sin, sin, cos]

        # Combine with the overall rotation matrix
        rotation_matrix = np.dot(rotation_matrix, plane_rotation)

    return rotation_matrix


def _get_random_coprime(n):
    """Finds a random integer k such that 1 <= k < n and gcd(k, n) = 1."""
    if n <= 1:
        raise ValueError("Cannot find coprime for n <= 1")
    if n == 2:
        # The only candidate k=1 is coprime to 2.
        return 1
    # Loop until a coprime k is found
    while True:
        # Generate k in the range [1, n-1]
        k = random.randint(1, n - 1)
        if math.gcd(k, n) == 1:
            return k


def generate_high_dim_rotation_exact_order_n(
    d: int,
    n: int,
    num_rotating_planes: Optional[int] = None,
):
    """
    Generates a random d-dimensional rotation matrix R such that R^n = I,
    and n is the *smallest* positive integer for which this holds (exact order n).

    The rotation belongs to the Special Orthogonal group SO(d) (orthogonal
    matrix with determinant +1). The condition R^n = I means applying the
    rotation n times returns to the identity. The additional constraint ensures
    that R^m is not the identity for any 1 <= m < n.

    Algorithm Overview:
    Builds upon the method for generating R with R^n = I, adding a constraint
    to ensure n is the minimal such power (i.e., R has exact order n).

    1.  Canonical Form Construction (D):
        - As before, R = P @ D @ P.T, where D is block-diagonal containing
          2x2 rotation blocks with angles theta_j = 2 * pi * k_j / n.
        - The order of R is the least common multiple (LCM) of the orders of
          the individual planar rotations. The order of a single planar
          rotation with angle theta_j = 2*pi*k_j/n is n_j = n / gcd(k_j, n).
        - To ensure LCM(n_1, ..., n_m) = n, it is sufficient to guarantee
          that *at least one* chosen k_j value is coprime to n
          (i.e., gcd(k_j, n) = 1). If gcd(k_j, n) = 1, then n_j = n, forcing
          the LCM of all orders to be exactly n.
        - This function constructs D by:
            a. Determining the number of 2x2 rotation blocks 'm'. Requires m>=1
               if n>1.
            b. Generating m candidate integers k_j randomly from {1, ..., n-1}.
            c. Checking if any generated k_j is coprime to n (using math.gcd).
            d. *If not*, replacing one candidate k_j with a new value k_coprime
               that is randomly sampled from the set
               {k | 1 <= k < n, gcd(k, n) = 1}. This guarantees the condition.
            e. Placing the m 2x2 rotation blocks corresponding to the final set
               of k_j values onto the diagonal of D.
            f. Filling the remaining diagonal entries with 1s.

    2.  Random Orientation (P):
        - Generated as before: a random rotation matrix P in SO(d) using QR
          decomposition of a Gaussian matrix.

    3.  Final Matrix (R):
        - Computed as R = P @ D @ P.T.

    This ensures R is in SO(d), R^n = I, and R^m != I for 1 <= m < n.

    Args:
        d (int): The dimension of the space (must be >= 2 for n>1).
        n (int): The exact desired order of the rotation (R^n = I, R^m != I for m<n).
                 Must be >= 1.
        num_rotating_planes (int, optional): The number of 2x2 rotation blocks
            (planes with non-trivial rotation) to include in the canonical form D.
            If n > 1, must be between 1 and floor(d/2). Using fewer planes is
            possible but increases the chance the initial random k's are not
            coprime, requiring adjustment.
            If None (default), sets it to max(1, floor(d/2)) if n>1.
            If n=1, this is ignored (result is Identity).

    Returns:
        np.ndarray: The d x d rotation matrix R in SO(d) with exact order n.

    Raises:
        ValueError: If inputs are invalid, or if parameters conflict (e.g., d=1
                    and n>1, or num_rotating_planes=0 and n>1).
    """

    # --- Input Validation ---
    if not isinstance(d, int) or d < 1:
        raise ValueError("Dimension d must be a positive integer.")
    if not isinstance(n, int) or n < 1:
        raise ValueError("Order n must be a positive integer.")

    # Handle d=1 case
    if d == 1:
        if n == 1:
            return np.array([[1.0]])  # Identity has order 1
        else:
            # Cannot have order > 1 in 1D
            raise ValueError("Cannot generate rotation of order n > 1 in d=1.")

    # Handle n=1 case (must be identity)
    if n == 1:
        # Identity has order 1
        return np.identity(d)

    # --- Determine number of rotating planes (m) ---
    # For exact order n > 1, we need at least one rotating plane.
    max_planes = d // 2
    if max_planes < 1:
        # Should not happen if d >= 2, already handled d=1 case
        raise ValueError(
            f"Dimension d={d} is too small to support rotating planes.")

    if num_rotating_planes is None:
        # Default to max planes, ensuring at least 1.
        m = max(1, max_planes)
    else:
        if not isinstance(num_rotating_planes,
                          int) or not (1 <= num_rotating_planes <= max_planes):
            raise ValueError(
                f"If n>1, num_rotating_planes must be an integer between 1 and {max_planes}."
            )
        m = num_rotating_planes

    # --- Generate k values ensuring the overall order is exactly n ---
    # (See Algorithm Overview, point 1c-d in docstring)

    # Generate m candidate integers k_j from {1, ..., n-1}
    # np.random.randint samples from [low, high)
    k_values = np.random.randint(1, n, size=m)

    # Check if at least one k_j is coprime to n
    is_any_coprime = any(math.gcd(k, n) == 1 for k in k_values)

    if not is_any_coprime:
        # If none are coprime, force one to be coprime to ensure exact order n.
        # Find a random k such that gcd(k, n) = 1
        k_coprime = _get_random_coprime(n)
        # Replace the first generated k value (or choose a random index).
        k_values[0] = k_coprime
        # Verify the replacement worked (optional sanity check)
        # assert math.gcd(k_values[0], n) == 1

    # --- Step 1: Construct the canonical block-diagonal matrix D ---
    # (See Algorithm Overview, point 1e-f in docstring)
    D = np.identity(d)
    for j in range(m):
        # Use the final k_values (guaranteed to have at least one coprime to n)
        k = k_values[j]
        theta = 2.0 * np.pi * k / n
        c, s = np.cos(theta), np.sin(theta)
        idx1, idx2 = 2 * j, 2 * j + 1
        D[idx1, idx1], D[idx1, idx2] = c, -s
        D[idx2, idx1], D[idx2, idx2] = s, c

    # --- Step 2: Generate a random orientation matrix P from SO(d) ---
    # (See Algorithm Overview, point 2 in docstring)
    A = np.random.randn(d, d)
    P, _ = np.linalg.qr(A)
    if np.linalg.det(P) < 0:
        P[:, 0] *= -1

    # --- Step 3: Combine D and P to get the final rotation R ---
    # (See Algorithm Overview, point 3 in docstring)
    R = P @ D @ P.T

    # The resulting R is in SO(d) and has exact order n by construction.
    assert np.allclose(np.linalg.matrix_power(R, n),
                       np.identity(d)), "R^n is not Identity"
    assert np.allclose(R @ R.T, np.identity(d)), "R is not orthogonal"
    assert np.isclose(np.linalg.det(R), 1.0), "R has determinant not 1"
    return R


@jaxtyped(typechecker=typechecked)
def generate_high_dim_rotation_single_turn_order_n(
    d: int,
    n: int,
    num_rotating_planes: Optional[int] = None,
) -> Float[np.ndarray, "d d"]:
    """
    Generates a random d-dimensional rotation matrix R such that:
    1. R^n = I (returns to identity after n steps).
    2. n is the *smallest* positive integer for which R^n = I (exact order n).
    3. The total rotation accumulated over n steps corresponds to exactly one
       full circle (2*pi radians), not multiple full circles. The cumulative
       angle for R^m (m<n) does not exceed 2*pi.

    This ensures R^m != I for 1 <= m < n, and avoids excessive "wrapping"
    where R^n might represent 4*pi, 6*pi, etc., or where intermediate
    powers R^m already exceed 2*pi cumulative rotation.

    The rotation belongs to the Special Orthogonal group SO(d).

    Algorithm Overview:
    The combined conditions imply that the fundamental rotation angle in the
    canonical form must be exactly theta = 2*pi / n for all participating planes.

    1.  Canonical Form Construction (D):
        - R = P @ D @ P.T, where D is block-diagonal.
        - To satisfy all conditions (R^n=I, exact order n, single turn),
          the 2x2 rotation blocks in D must *all* use the specific angle
          theta = 2 * pi / n. This corresponds to setting k=1 in the
          general angle formula theta = 2*pi*k/n.
        - Using theta = 2*pi/n ensures:
            a. R^n = I (since n * theta = 2*pi).
            b. Exact order n (since k=1 is coprime to n).
            c. Total rotation of exactly 2*pi over n steps, with cumulative
               angles m*theta <= 2*pi for m <= n.
        - The function constructs D by:
            a. Determining the number of 2x2 blocks 'm' (num_rotating_planes).
               Requires m>=1 if n>1.
            b. Calculating the single angle theta = 2*pi / n.
            c. Placing m identical 2x2 rotation blocks for angle theta onto the
               diagonal of an identity matrix.
            d. Leaving remaining diagonal entries as 1.

    2.  Random Orientation (P):
        - A random rotation matrix P in SO(d) is generated using QR decomposition,
          providing a random orientation for the rotation planes.

    3.  Final Matrix (R):
        - Computed as R = P @ D @ P.T.

    Args:
        d (int): The dimension of the space (must be >= 2 for n>1).
        n (int): The exact desired order of the rotation, corresponding to a
                 single full turn (must be >= 1).
        num_rotating_planes (int, optional): The number of 2x2 rotation blocks
            (planes with non-trivial rotation) to include in the canonical form D.
            If n > 1, must be between 1 and floor(d/2).
            If None (default), sets it to max(1, floor(d/2)) if n>1.
            If n=1, this is ignored.

    Returns:
        np.ndarray: The d x d rotation matrix R in SO(d) with exact order n,
                    completing exactly one turn in n steps.

    Raises:
        ValueError: If inputs are invalid or parameters conflict (e.g., d=1 & n>1).
    """

    # --- Input Validation ---
    if not isinstance(d, int) or d < 1:
        raise ValueError("Dimension d must be a positive integer.")
    if not isinstance(n, int) or n < 1:
        raise ValueError("Order n must be a positive integer.")

    # Handle d=1 case
    if d == 1:
        if n == 1:
            return np.array([[1.0]])  # Identity has order 1
        else:
            # Cannot have order > 1 in 1D
            raise ValueError("Cannot generate rotation of order n > 1 in d=1.")

    # Handle n=1 case (must be identity)
    if n == 1:
        # Identity has order 1
        return np.identity(d)

    # --- Determine number of rotating planes (m) ---
    # For exact order n > 1, we need at least one rotating plane.
    max_planes = d // 2
    if max_planes < 1:
        # Should only happen if d=1, which was handled.
        raise ValueError(f"Dimension d={d} is too small for rotating planes.")

    if num_rotating_planes is None:
        # Default to max planes, ensuring at least 1.
        m = max(1, max_planes)
    else:
        # If n>1, user must specify at least 1 plane.
        if not isinstance(num_rotating_planes,
                          int) or not (1 <= num_rotating_planes <= max_planes):
            raise ValueError(
                f"If n>1, num_rotating_planes must be an integer between 1 and {max_planes}."
            )
        m = num_rotating_planes

    # --- Define the single rotation angle per step ---
    # theta = 2*pi / n is uniquely determined by the constraints.
    theta = 2.0 * np.pi / n

    # --- Step 1: Construct the canonical block-diagonal matrix D ---
    # All rotation blocks use the same angle theta = 2*pi / n.
    D = np.identity(d)
    c, s = np.cos(theta), np.sin(theta)  # Calculate trig values once
    for j in range(m):
        idx1, idx2 = 2 * j, 2 * j + 1
        # Place the 2x2 rotation block into D
        D[idx1, idx1], D[idx1, idx2] = c, -s
        D[idx2, idx1], D[idx2, idx2] = s, c
    # The remaining d - 2m diagonal elements are already 1.

    # --- Step 2: Generate a random orientation matrix P from SO(d) ---
    # This provides the random orientation of the rotation planes.
    A = np.random.randn(d, d)
    P, _ = np.linalg.qr(A)
    # Ensure P is in SO(d) (determinant +1)
    if np.linalg.det(P) < 0:
        P[:, 0] *= -1

    # --- Step 3: Combine D and P to get the final rotation R ---
    # R = P D P^T applies the canonical rotation D in the basis defined by P.
    R = P @ D @ P.T

    # This R satisfies all conditions: SO(d), exact order n, single turn.
    assert np.allclose(np.linalg.matrix_power(R, n),
                       np.identity(d)), "R^n is not Identity"
    assert np.allclose(R @ R.T, np.identity(d)), "R is not orthogonal"
    assert np.isclose(np.linalg.det(R), 1.0), "R has determinant not 1"
    return R


def create_rotation_group(dim: int, group_size: int):
    """
    We define a rotation group by a rotation axis and the number of elements in that group.
    """
    # base_rotation = generate_high_dim_rotation_exact_order_n(d=dim,
    #                                                          n=group_size)
    base_rotation = generate_high_dim_rotation_single_turn_order_n(
        d=dim, n=group_size + 1)
    rotation_groups = [
        np.linalg.matrix_power(base_rotation, i)
        for i in range(1, group_size + 1)
    ]

    # to check whether this is a valid group, we take the last rotation matrix, perform one more rotation,
    # and check whether it is close to the identity matrix
    hopefully_identity = np.dot(rotation_groups[-1], base_rotation)
    assert np.allclose(hopefully_identity, np.identity(
        dim)), f"This is not a valid rotation group, {hopefully_identity}"

    # assert that none of the matrices are the identity matrix
    # Check that none of the matrices are the identity matrix
    for i, matrix in enumerate(rotation_groups):
        assert not np.allclose(
            matrix,
            np.identity(dim)), f"Matrix at index {i} is the identity matrix"

    return rotation_groups


def get_rotation_axes(dim: int):
    axis = torch.arange(dim)
    # every possible combination of two axis
    combinations = torch.combinations(axis, 2)
    # we also want to include the mirror image of each combination
    # so if we have (1, 2) we also want (2, 1), since this is the
    # rotation around the same axis, but in the opposite direction
    combinations = torch.cat([combinations, combinations[:, [1, 0]]], dim=0)
    return combinations


def sample_group_sizes(min_group_size: int, max_group_size: int,
                       total_group_elements: int, num_groups: int):
    """
    Sample a list of group sizes that sum to a target total while satisfying size constraints.

    Args:
        min_group_size (int): Minimum size allowed for each group
        max_group_size (int): Maximum size allowed for each group  
        total_group_elements (int): Target sum of all group sizes
        num_groups (int): Number of groups to generate sizes for

    Returns:
        torch.Tensor: A randomly shuffled tensor of group sizes that sum to total_group_elements

    Raises:
        ValueError: If constraints cannot be satisfied given the input parameters
    """
    # First check if it's possible to satisfy the constraints
    if min_group_size * num_groups > total_group_elements:
        raise ValueError(
            f"Total elements {total_group_elements} is too small - need at least {min_group_size * num_groups} elements for {num_groups} groups of minimum size {min_group_size}"
        )

    if max_group_size * num_groups < total_group_elements:
        raise ValueError(
            f"Total elements {total_group_elements} is too large - can have at most {max_group_size * num_groups} elements with {num_groups} groups of maximum size {max_group_size}"
        )

    result = []
    remaining_sum = total_group_elements
    for i in range(num_groups - 1):
        max_possible = min(
            max_group_size,
            remaining_sum - (num_groups - i - 1) * min_group_size)
        value = torch.randint(min_group_size, max_possible + 1, (1,)).item()
        result.append(value)
        remaining_sum -= value
    result.append(remaining_sum)
    # shuffle using torch
    sizes = torch.tensor(result)[torch.randperm(len(result))]

    assert sizes.sum(
    ) == total_group_elements, "The sum of the group sizes does not match the total group elements"

    return sizes


def sample_rotation_groups(
    dim: int,
    num_groups: int,
    min_group_size: int,
    max_group_size: int,
    total_group_elements: int,
):
    """
    Generate random rotation groups in a specified dimension with constrained group sizes.

    Args:
        dim (int): Dimension of the space to generate rotations in
        num_groups (int): Number of rotation groups to generate
        min_group_size (int): Minimum size for each rotation group
        max_group_size (int): Maximum size for each rotation group
        total_group_elements (int): Total number of rotation elements across all groups

    Returns:
        Tuple containing:
            - List[List[np.ndarray]]: List of rotation groups, where each group contains rotation matrices
            - torch.Tensor: Selected rotation axes for each group
            - torch.Tensor: Size of each rotation group

    Raises:
        AssertionError: If input constraints are invalid
    """

    assert min_group_size >= 1, "min_group_size must be greater or equal to 1"
    assert min_group_size <= max_group_size, "min_group_size must be less than max_group_size"
    group_sizes = sample_group_sizes(min_group_size, max_group_size,
                                     total_group_elements, num_groups)

    rotation_groups = []
    for i in range(num_groups):
        rotation_group = create_rotation_group(
            dim,
            group_size=group_sizes[i].item(),
        )
        rotation_groups.append(rotation_group)

    return rotation_groups, None, group_sizes


def sample_groups_with_permutations(
    dim: int,
    num_groups: int,
    min_group_size: int,
    max_group_size: int,
    num_matrices: int,
    num_groups_to_combine: int,
):
    """
    Sample rotation matrices by combinining elements of multiple groups.
    1. Sample random group sizes
    2. Sample base transformation matrix from each group
    3. Sample num_groups_to_combine groups
    4. Sample random repetition of group element (<= group_size)
    5. Combine group elements by multiplying them together
    """
    group_sizes = np.random.randint(min_group_size, max_group_size,
                                    (num_groups,))

    # identity matrix is always the first group
    base_rotation_matrices = []
    for i in range(num_groups):
        n = group_sizes[i].item() + 1
        base_rotation = generate_high_dim_rotation_single_turn_order_n(d=dim,
                                                                       n=n)
        # check that repeating the base rotation the correct number of times yields the identity matrix
        assert np.allclose(
            np.linalg.matrix_power(base_rotation, n), np.eye(dim)
        ), f"Base rotation matrix does not yield identity matrix when repeated {n} times"

        base_rotation_matrices.append(base_rotation)

    rotation_matrices = []
    group_sequences = []
    group_sequences_repetitions = []
    for i in range(num_matrices):
        # sample num_groups_to_combine groups
        group_sequence = np.random.choice(np.arange(num_groups),
                                          num_groups_to_combine)

        # for each group, sample between 0 and group_size - 1 repetitions
        group_repetitions = [
            np.random.randint(0, group_sizes[j].item() + 1)
            for j in group_sequence
        ]

        group_sequences.append(group_sequence)
        group_sequences_repetitions.append(group_repetitions)

        # combine the groups by multiplying them together
        combined_rotation_matrices = np.eye(dim)
        for i in range(num_groups_to_combine):
            new_group_element = np.linalg.matrix_power(
                base_rotation_matrices[group_sequence[i]], group_repetitions[i])
            combined_rotation_matrices = np.dot(combined_rotation_matrices,
                                                new_group_element)

        # check the combined_rotation_matrices is a valid rotation matrix
        RRT = combined_rotation_matrices @ combined_rotation_matrices.T
        assert np.allclose(
            RRT, np.eye(dim)), "Combined rotation matrix is not orthogonal"
        rotation_matrices.append(combined_rotation_matrices)

    return rotation_matrices, group_sequences, group_sequences_repetitions, base_rotation_matrices, group_sizes
