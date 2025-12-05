from dataclasses import dataclass
from typing import Tuple

import torch
from jaxtyping import Float
from jaxtyping import Integer
from jaxtyping import jaxtyped
from torch import Tensor
from jaxtyping._typeguard import typechecked

from groupcl.models.dynamics.base import BaseDynamicsModel
from groupcl.models.dynamics.linear_dynamics import LinearDynamicsModel
from groupcl.models.dynamics.switching_dynamics import MSESwitchingModel
from config_dataclass import torch_dataclass, config_field


@torch_dataclass
class GumbelSLDS(BaseDynamicsModel):
    """Switching Linear Dynamical System that uses MSE to determine switching state."""

    linear_dynamics: LinearDynamicsModel = config_field(
        default_factory=LinearDynamicsModel)
    switching_model: MSESwitchingModel = config_field(
        default_factory=MSESwitchingModel)

    @property
    def tau(self) -> float:
        """Temperature parameter for Gumbel-Softmax sampling."""
        return self.switching_model.tau

    @tau.setter
    def tau(self, value: float):
        """Set temperature parameter for Gumbel-Softmax sampling."""
        self.switching_model.tau = value

    @property
    def num_systems(self) -> int:
        """Number of linear dynamical systems."""
        return self.linear_dynamics.num_systems

    @property
    def num_modes(self) -> int:
        """Alias for num_systems."""
        return self.num_systems

    @property
    def dim(self) -> int:
        """Dimension of the state space."""
        return self.linear_dynamics.dim

    # alias
    @property
    def dynamics_dim(self) -> int:
        """Dimension of the dynamics space."""
        return self.dim

    @jaxtyped(typechecker=typechecked)
    def forward(
        self,
        x: Float[Tensor, "batch dim"],
        x_prime: Float[Tensor, "batch dim"],
        **kwargs,
    ) -> Tuple[Float[Tensor, "batch dim"], Float[Tensor, "batch dim dim"]]:

        # if model's dim is smaller than data dim, we need to fit only on the first model.dim dimensions and predict with the identity on the rest

        if self.dim < x.shape[-1]:
            x_first = x[..., :self.dim]
            x_prime_first = x_prime[..., :self.dim]
            # fit on first
            x_prime_pred, Q = self._forward(x_first, x_prime_first, **kwargs)
            # append last dims of target to target_pred
            x_prime_pred = torch.cat([x_prime_pred, x[..., self.dim:]], dim=-1)
            # append block identity to Q
            latent_dim = x.shape[-1]
            content_dim = latent_dim - self.dim
            num_systems = Q.shape[0]
            I = torch.eye(content_dim, device=Q.device)
            Q_full = torch.zeros(num_systems,
                                 latent_dim,
                                 latent_dim,
                                 device=Q.device)
            Q_full[:, :self.dim, :self.dim] = Q
            Q_full[:, self.dim:, self.dim:] = I
            Q = Q_full
            return x_prime_pred, Q

        else:
            return self._forward(x, x_prime, **kwargs)

    @jaxtyped(typechecker=typechecked)
    def _forward(
        self,
        x: Float[Tensor, "batch dim"],
        x_prime: Float[Tensor, "batch dim"],
        num_samples=1,
        **kwargs,
    ) -> Tuple[Float[Tensor, "batch dim"], Float[Tensor, "batch dim dim"]]:
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
        # Predict next state for all possible modes
        # Expand x to (batch, num_modes, dim)
        x_prime_pred_all_modes = self.forward_all_modes(x)

        # Sample modes using Gumbel-Softmax
        # mode_samples shape (batch, num_samples, num_modes)
        mode_samples = self.switching_model(
            x=x_prime,
            x_pred_all_modes=x_prime_pred_all_modes,
            num_samples=num_samples,
        )

        # Expand predictions to include samples dimension
        # x_pred shape (batch, num_samples, num_modes, dim)
        x_prime_pred = x_prime_pred_all_modes.unsqueeze(1).expand(
            -1, num_samples, -1, -1)
        # when in training mode, mode_samples are probabilities and this is a weighted average
        # when in evaluation mode, mode_samples are one-hot encoded and this is a selection (think argmax)
        # x_next_pred shape (batch, num_samples, dim)
        x_prime_pred = (x_prime_pred * mode_samples.unsqueeze(-1)).sum(dim=-2)
        return x_prime_pred.squeeze(1), self.linear_dynamics.A.detach()[
            mode_samples.argmax(dim=-1).squeeze(1)]

    @jaxtyped(typechecker=typechecked)
    def forward_all_modes(
        self,
        x: Float[Tensor, "batch dim"],
    ) -> Float[Tensor, "batch {self.num_modes} dim"]:
        batch_size, dim = x.shape
        x = x.clone()
        # Expand x to predict with all possible modes
        x = x.unsqueeze(1).expand(-1, self.num_systems, -1)
        # create a tensor of all modes
        system_idx = torch.arange(self.num_systems, device=x.device)
        # expand to match batch
        system_idx = system_idx.expand(batch_size, self.num_systems)

        # and now we need to flatten the batch and systems dimension
        x = x.flatten(start_dim=0, end_dim=1)
        system_idx = system_idx.flatten(start_dim=0, end_dim=1)

        # Get predictions for all modes
        x = self.linear_dynamics(
            x=x,
            system_idx=system_idx,
        )

        # unflatten the result and reshape to (batch, num_modes, dim)
        return x.reshape(batch_size, self.num_systems, dim)
