from dataclasses import dataclass
from typing import Tuple

import torch
from jaxtyping import Float
from jaxtyping import Integer
from jaxtyping import jaxtyped
from torch import Tensor
from jaxtyping._typeguard import typechecked

from groupcl.models.dynamics.base import BaseDynamicsModel
from config_dataclass import torch_dataclass, config_field


@torch_dataclass
class IdentitySLDSModel(BaseDynamicsModel):
    """
    A dynamics model that does nothing and works as a debug replacement for the GumbelSLDS model.
    """
    # for compatibility with GumbelSLDS
    num_systems: int = config_field(default=1)
    dim: int = config_field(default=1)

    @property
    def tau(self) -> float:
        return 1.0

    @tau.setter
    def tau(self, value: float):
        """Set temperature parameter for Gumbel-Softmax sampling."""
        pass

    @property
    def num_modes(self) -> int:
        """Alias for num_systems."""
        return self.num_systems

    @jaxtyped(typechecker=typechecked)
    def forward(
        self,
        x: Float[Tensor, "batch dim"],
        y: Float[Tensor, "batch dim"],
        x_prime: Float[Tensor, "batch dim"],
        y_prime: Float[Tensor, "batch dim"],
        num_samples: int = 1,
        **kwargs,
    ) -> Tuple[
            Float[Tensor, "batch num_samples dim"],
            Float[Tensor, "batch num_samples dim"],
            Float[Tensor, "batch num_samples {self.num_modes}"],
    ]:
        """Forward pass of the SLDS model.
        Uses Gumbel-Softmax sampling for the mode prediction.

        Args: Paired Group Action data
            x: Pair 1 original state of shape 
            y: Pair 2 original state of shape 
            x_prime: Pair 1 transformed state of shape 
            y_prime: Pair 2 transformed state of shape 
            num_samples: Number of gumbel samples to draw 

        Returns:
            Tuple containing:
            - x_prime_pred: Predicted transformed state of pair 1 of shape
            - y_prime_pred: Predicted transformed state of pair 2 of shape
            - mode_samples: Gumbel samples of shape
        """
        fake_x_prime_pred = x.unsqueeze(1).repeat(1, num_samples, 1)
        fake_y_prime_pred = y.unsqueeze(1).repeat(1, num_samples, 1)
        fake_mode_samples = torch.ones(x.shape[0],
                                       device=x.device) / self.num_systems
        fake_mode_samples = fake_mode_samples.unsqueeze(-1).repeat(
            1, self.num_systems).unsqueeze(-1).repeat(1, 1, self.num_systems)

        return (fake_x_prime_pred, fake_y_prime_pred, fake_mode_samples)
