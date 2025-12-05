import random
from dataclasses import dataclass
from typing import List, Literal, TypeVar, Union, Optional, Tuple

import numpy as np
import torch
from jaxtyping import Float
from jaxtyping import jaxtyped
from jaxtyping import Integer
from torch import Tensor
from jaxtyping._typeguard import typechecked

from groupcl.models.dynamics.base import BaseDynamicsModel
from groupcl.models.dynamics.utils import IdentityLDSParameters
from groupcl.models.dynamics.utils import LDSParameterInitializer
from groupcl.models.utils import ModelStorageMixin
from config_dataclass import torch_dataclass, config_field

T = TypeVar("T")


@jaxtyped(typechecker=typechecked)
@torch_dataclass
class LinearDynamicsModel(ModelStorageMixin, BaseDynamicsModel):
    """
    Simple linear dynamics model that can handle both single and multiple linear dynamical systems.
    For multiple systems, parameters are stacked along first dimension.
    """

    dim: int = config_field(default=2)
    num_systems: int = config_field(default=1)
    use_bias: bool = config_field(default=False)
    noise_std: Union[
        float,
        Tuple[float],  # per system noise
        Tuple[Tuple[float]],
    ] = config_field(default=0.0)
    learnable_noise: bool = config_field(default=False)
    initializer: LDSParameterInitializer = config_field(
        default_factory=IdentityLDSParameters)

    def __lazy_post_init__(self):
        ModelStorageMixin.__lazy_post_init__(self)
        BaseDynamicsModel.__lazy_post_init__(self)
        self.init_parameters()

    def init_parameters(self):
        # Set random seed for reproducibility
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)
        self.initializer.lazy_init(seed=self.seed)

        self.init_dynamics_matrix()
        self.init_bias()
        self.init_log_noise()

    def init_dynamics_matrix(self):
        self.A = torch.nn.Parameter(
            self.initializer.dynamics_matrix(self.num_systems, self.dim))

    def init_bias(self):
        if self.use_bias:
            self.b = torch.nn.Parameter(
                self.initializer.dynamics_bias(self.num_systems, self.dim))
        else:
            self.register_buffer(
                'b',
                torch.zeros((self.num_systems, self.dim)),
            )

    def init_log_noise(self):
        attr_name = 'log_noise_std'
        value = self._init_log_noise()
        if self.learnable_noise:
            self.register_parameter(attr_name, torch.nn.Parameter(value))
        else:
            self.register_buffer(attr_name, value)

    @jaxtyped(typechecker=typechecked)
    def _init_log_noise(self) -> Float[Tensor, "{self.num_systems} {self.dim}"]:
        noise_std = torch.tensor(self.noise_std, dtype=torch.float32)
        if noise_std.ndim == 0:
            # Scalar noise means same noise for all systems
            noise_std = noise_std.unsqueeze(-1).repeat(self.num_systems)

        if noise_std.ndim == 1:
            # Per-system noise, means same noise for all dimensions
            noise_std = noise_std.unsqueeze(-1).repeat(1, self.dim)

        if noise_std.ndim == 2:
            # Per-system, per-dimension noise
            pass
        return torch.log(noise_std)

    @jaxtyped(typechecker=typechecked)
    def forward(
        self,
        x: Float[Tensor, "batch dim"],
        system_idx: Optional[Integer[Tensor, "batch"]] = None,
    ) -> Float[Tensor, "batch dim"]:

        if system_idx is None:
            assert self.num_systems == 1, "system_idx must be provided when num_systems > 1"
            system_idx = torch.zeros(x.shape[0], dtype=torch.long)

        A = self.A[system_idx]
        b = self.b[system_idx]

        x = (A @ x.unsqueeze(-1)).squeeze(-1) + b

        if self.noise_std > 0:
            noise_std = torch.exp(self.log_noise_std[system_idx])
            noise_dist = torch.distributions.normal.Normal(
                loc=torch.zeros_like(noise_std),
                scale=noise_std,
            )
            x += noise_dist.rsample()

        return x

    @jaxtyped(typechecker=typechecked)
    def to_gt_dynamics(self: T, set_noise_to_zero: bool = True) -> T:
        linear_dynamics = self.clone()
        if set_noise_to_zero:
            linear_dynamics.noise_std = 0.
        return linear_dynamics


@jaxtyped(typechecker=typechecked)
@torch_dataclass
class OrthogonalLinearDynamicsModel(LinearDynamicsModel):
    """
    A LinearDynamicsModel that is parametrized to only allow orthogonal dynamics matrices.
    """
    orthogonal_map: Literal[
        "matrix_exp",
        "cayley",
        "householder",
    ] = config_field(default="matrix_exp")

    use_trivialization: bool = config_field(default=True)

    def init_dynamics_matrix(self):
        super().init_dynamics_matrix()
        torch.nn.utils.parametrizations.orthogonal(
            self,
            name="A",
            orthogonal_map=self.orthogonal_map,
            use_trivialization=self.use_trivialization,
        )
