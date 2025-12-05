from abc import ABC
from abc import abstractmethod
from dataclasses import dataclass
import random

import numpy as np
import torch

from jaxtyping import Float
from jaxtyping import jaxtyped
from torch import Tensor
from jaxtyping._typeguard import typechecked
from scipy.stats import ortho_group, special_ortho_group

from groupcl.models.dynamics.base import BaseDynamicsModel
from config_dataclass import Configurable, config_dataclass, torch_dataclass, config_field, check_initialized

from groupcl.utils.rotation_matrix import MinMaxRotationSampler
from groupcl.utils.rotation_matrix import RotationSampler
from groupcl.utils.rotation_matrix import sample_rotation_groups
from groupcl.utils.rotation_matrix import sample_groups_with_permutations


@jaxtyped(typechecker=typechecked)
@config_dataclass
class LDSParameterInitializer(Configurable, ABC):
    """
    A class for initializing the parameters of a LinearDynamicsModel.
    """
    lazy: bool = True

    def __lazy_post_init__(self, seed: int):
        self._seed = seed

    @property
    @check_initialized
    def seed(self) -> int:
        return self._seed

    def set_seed(self):
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)

    @jaxtyped(typechecker=typechecked)
    def dynamics_matrix(
        self,
        num_systems: int,
        dim: int,
    ) -> Float[Tensor, "{num_systems} {dim} {dim}"]:
        self.set_seed()
        return self._dynamics_matrix(num_systems, dim)

    @jaxtyped(typechecker=typechecked)
    def dynamics_bias(
        self,
        num_systems: int,
        dim: int,
    ) -> Float[Tensor, "{num_systems} {dim}"]:
        self.set_seed()
        return self._dynamics_bias(num_systems, dim)

    @abstractmethod
    @jaxtyped(typechecker=typechecked)
    def _dynamics_matrix(
        self,
        num_systems: int,
        dim: int,
    ) -> Float[Tensor, "{num_systems} {dim} {dim}"]:
        pass

    @abstractmethod
    @jaxtyped(typechecker=typechecked)
    def _dynamics_bias(
        self,
        num_systems: int,
        dim: int,
    ) -> Float[Tensor, "{num_systems} {dim}"]:
        pass


@jaxtyped(typechecker=typechecked)
@torch_dataclass
class NormalLDSParameters(LDSParameterInitializer):
    """
    Initializes Linear Dynamical System parameters from normal distribution.
    """
    dynamics_matrix_mean: float = config_field(default=0.0)
    dynamics_matrix_std: float = config_field(default=1.0)
    dynamics_bias_mean: float = config_field(default=0.0)
    dynamics_bias_std: float = config_field(default=1.0)

    @jaxtyped(typechecker=typechecked)
    def _dynamics_matrix(
        self,
        num_systems: int,
        dim: int,
    ) -> Float[Tensor, "{num_systems} {dim} {dim}"]:
        return torch.normal(mean=self.dynamics_matrix_mean,
                            std=self.dynamics_matrix_std,
                            size=(num_systems, dim, dim))

    @jaxtyped(typechecker=typechecked)
    def _dynamics_bias(
        self,
        num_systems: int,
        dim: int,
    ) -> Float[Tensor, "{num_systems} {dim}"]:
        return torch.normal(mean=self.dynamics_bias_mean,
                            std=self.dynamics_bias_std,
                            size=(num_systems, dim))


@jaxtyped(typechecker=typechecked)
@torch_dataclass
class IdentityLDSParameters(LDSParameterInitializer):
    """
    LinearDynamicsModel initialized with identity matrices.
    """

    @jaxtyped(typechecker=typechecked)
    def _dynamics_matrix(
        self,
        num_systems: int,
        dim: int,
    ) -> Float[Tensor, "{num_systems} {dim} {dim}"]:
        A = torch.stack([torch.eye(dim) for _ in range(num_systems)])
        return A

    @jaxtyped(typechecker=typechecked)
    def _dynamics_bias(
        self,
        num_systems: int,
        dim: int,
    ) -> Float[Tensor, "{num_systems} {dim}"]:
        b = torch.zeros((num_systems, dim))
        return b


@jaxtyped(typechecker=typechecked)
@torch_dataclass
class RotationLDSParameters(LDSParameterInitializer):
    """
    LinearDynamicsModel initialized with rotation matrices.
    """
    rotation_sampler: RotationSampler = config_field(
        default_factory=MinMaxRotationSampler)

    @jaxtyped(typechecker=typechecked)
    def _dynamics_matrix(
        self,
        num_systems: int,
        dim: int,
    ) -> Float[Tensor, "{num_systems} {dim} {dim}"]:
        A = torch.stack(
            [self.rotation_sampler.sample(dim) for _ in range(num_systems)])
        return A

    @jaxtyped(typechecker=typechecked)
    def _dynamics_bias(
        self,
        num_systems: int,
        dim: int,
    ) -> Float[Tensor, "{num_systems} {dim}"]:
        b = torch.zeros((num_systems, dim))
        return b


@jaxtyped(typechecker=typechecked)
@torch_dataclass
class InvertibleMatrixLDSParameters(LDSParameterInitializer):
    """
    LinearDynamicsModel initialized with random invertible matrices.
    """

    # technically any non-zero minimum would suffice,
    # but to avoid degenerate/numerically unstable matrices, pick slighly larger than 0
    min_determinant: float = config_field(default=0.2)
    # again to avoid degenerate/numerically unstable matrices pick reasonably small maximum
    max_determinant: float = config_field(default=10.0)

    # A @ A^-1 should be identity within some error margin
    identity_threshold: float = config_field(default=1e-3)

    max_attempts: int = config_field(default=10_000)

    def sample_invertible_matrix(self,
                                 dim: int) -> Float[Tensor, "{dim} {dim}"]:
        max_attempts = self.max_attempts  # Avoid infinite loops

        As = torch.randn((self.max_attempts, dim, dim))

        # Check if determinant is in the desired range
        det = torch.abs(torch.det(As))
        det_mask = (det > self.min_determinant) & (det < self.max_determinant)
        As_filtered = As[det_mask]

        if len(As_filtered) > 0:
            # Check if inverse is stable
            for A in As_filtered:
                A_inv = torch.inverse(A)
                # Check if A @ A^-1 is close to identity
                identity_error = torch.norm(torch.eye(dim) - A @ A_inv)
                if identity_error < self.identity_threshold:
                    return A

        raise ValueError(
            f"Failed to sample invertible matrix after {max_attempts} attempts")

    @jaxtyped(typechecker=typechecked)
    def _dynamics_matrix(
        self,
        num_systems: int,
        dim: int,
    ) -> Float[Tensor, "{num_systems} {dim} {dim}"]:
        A = torch.stack(
            [self.sample_invertible_matrix(dim) for _ in range(num_systems)])
        return A

    @jaxtyped(typechecker=typechecked)
    def _dynamics_bias(
        self,
        num_systems: int,
        dim: int,
    ) -> Float[Tensor, "{num_systems} {dim}"]:
        b = torch.zeros((num_systems, dim))
        return b


@jaxtyped(typechecker=typechecked)
@torch_dataclass
class OrthogonalMatrixLDSParameters(LDSParameterInitializer):
    """
    LinearDynamicsModel initialized with random orthogonal matrices.
    """

    def sample_orthogonal_matrix(self,
                                 dim: int) -> Float[Tensor, "{dim} {dim}"]:
        return torch.tensor(ortho_group.rvs(dim), dtype=torch.float32)

    @jaxtyped(typechecker=typechecked)
    def _dynamics_matrix(
        self,
        num_systems: int,
        dim: int,
    ) -> Float[Tensor, "{num_systems} {dim} {dim}"]:
        A = torch.stack(
            [self.sample_orthogonal_matrix(dim) for _ in range(num_systems)])
        return A

    @jaxtyped(typechecker=typechecked)
    def _dynamics_bias(
        self,
        num_systems: int,
        dim: int,
    ) -> Float[Tensor, "{num_systems} {dim}"]:
        b = torch.zeros((num_systems, dim))
        return b


@jaxtyped(typechecker=typechecked)
@torch_dataclass
class SpecialOrthogonalMatrixLDSParameters(LDSParameterInitializer):
    """
    LinearDynamicsModel initialized with random special orthogonal matrices.
    """

    def sample_orthogonal_matrix(self,
                                 dim: int) -> Float[Tensor, "{dim} {dim}"]:
        return torch.tensor(special_ortho_group.rvs(dim), dtype=torch.float32)

    @jaxtyped(typechecker=typechecked)
    def _dynamics_matrix(
        self,
        num_systems: int,
        dim: int,
    ) -> Float[Tensor, "{num_systems} {dim} {dim}"]:
        A = torch.stack(
            [self.sample_orthogonal_matrix(dim) for _ in range(num_systems)])
        return A

    @jaxtyped(typechecker=typechecked)
    def _dynamics_bias(
        self,
        num_systems: int,
        dim: int,
    ) -> Float[Tensor, "{num_systems} {dim}"]:
        b = torch.zeros((num_systems, dim))
        return b


@jaxtyped(typechecker=typechecked)
@torch_dataclass
class RotationGroupLDSParameters(LDSParameterInitializer):
    """
    LinearDynamicsModel initialized with group rotation matrices.
    Each group 
    """
    min_group_size: int = config_field(default=2)
    max_group_size: int = config_field(default=10)
    num_groups: int = config_field(default=10)

    @jaxtyped(typechecker=typechecked)
    def _dynamics_matrix(
        self,
        num_systems: int,
        dim: int,
    ) -> Float[Tensor, "{num_systems} {dim} {dim}"]:

        # sample rotation groups
        self.rotation_groups, _, self.group_sizes = sample_rotation_groups(
            dim=dim,
            num_groups=self.num_groups,
            min_group_size=self.min_group_size,
            max_group_size=self.max_group_size,
            total_group_elements=num_systems,
        )

        # rotation_groups is a list of lists of numpy arrays (rotation matrices)
        # we need to flatten this and build a corresponding index for each group
        rotation_matrices = []
        group_indices = []
        group_matrix_indices = []

        for i, group in enumerate(self.rotation_groups):
            for j, rotation_matrix in enumerate(group):
                rotation_matrices.append(rotation_matrix)
                group_indices.append(i)
                group_matrix_indices.append(j)

        self.group_indices = torch.tensor(group_indices, dtype=torch.long)
        self.group_matrix_indices = torch.tensor(group_matrix_indices,
                                                 dtype=torch.long)

        return torch.tensor(rotation_matrices, dtype=torch.float32)

    @jaxtyped(typechecker=typechecked)
    def _dynamics_bias(
        self,
        num_systems: int,
        dim: int,
    ) -> Float[Tensor, "{num_systems} {dim}"]:
        b = torch.zeros((num_systems, dim))
        return b


@jaxtyped(typechecker=typechecked)
@torch_dataclass
class GroupPermutationLDSParameters(LDSParameterInitializer):
    """
    After sampling a set of rotation groups, construct system matrices by sampling elements from these groups and combining them.
    1. Sample the groups
    2. Sample elements from these groups
    3. sample amount of repetitions of each group element
    4. construct the final system matrix by taking the product of all these elements G = \prod_i G_i^{n_i}
    """
    num_groups: int = config_field(default=10)
    min_group_size: int = config_field(default=2)
    max_group_size: int = config_field(default=10)
    num_groups_to_combine: int = config_field(default=2)

    @jaxtyped(typechecker=typechecked)
    def _dynamics_matrix(
        self,
        num_systems: int,
        dim: int,
    ) -> Float[Tensor, "{num_systems} {dim} {dim}"]:

        (
            rotation_matrices,
            self.group_sequences,
            self.group_sequences_repetitions,
            self.base_rotation_matrices,
            self.group_sizes,
        ) = sample_groups_with_permutations(
            dim=dim,
            num_groups=self.num_groups,
            min_group_size=self.min_group_size,
            max_group_size=self.max_group_size,
            num_groups_to_combine=self.num_groups_to_combine,
            num_matrices=num_systems,
        )

        # convert to tensor
        rotation_matrices = torch.stack(
            [torch.from_numpy(m) for m in rotation_matrices], dim=0)
        return rotation_matrices.to(torch.float32)

    @jaxtyped(typechecker=typechecked)
    def _dynamics_bias(
        self,
        num_systems: int,
        dim: int,
    ) -> Float[Tensor, "{num_systems} {dim}"]:
        b = torch.zeros((num_systems, dim))
        return b


@jaxtyped(typechecker=typechecked)
def freeze_model(model: BaseDynamicsModel) -> None:
    """
    Freezes the given PyTorch model by disabling gradient updates for its parameters
    and setting it to evaluation mode.
    """
    # Disable gradient computation for all model parameters
    for param in model.parameters():
        param.requires_grad = False

    # Set the model to evaluation mode (important if the model has BatchNorm or Dropout layers)
    model.eval()
