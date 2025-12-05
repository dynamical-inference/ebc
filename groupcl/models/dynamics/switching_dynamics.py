from abc import ABC
from abc import abstractmethod
from dataclasses import dataclass
from typing import Type

import numpy as np
import torch
from jaxtyping import Float
from jaxtyping import jaxtyped
from torch import Tensor
from torch.nn import functional as F
from jaxtyping._typeguard import typechecked

from groupcl.models.dynamics.base import BaseDynamicsModel
from groupcl.models.dynamics.mse_logits import ScaledInverseMSE
from groupcl.models.dynamics.mse_logits import MSEToLogitFunction
from groupcl.models.utils import ModelStorageMixin

from config_dataclass import torch_dataclass, config_field, state_field


@jaxtyped(typechecker=typechecked)
@torch_dataclass
class SwitchingModel(ModelStorageMixin, BaseDynamicsModel, ABC):
    """Models the switching behavior of a switching linear dynamical system.
    This model predicts the mode (the lds system to be used for the next timestep) of the SLDS for the given input."""

    def __new__(cls, *args, **k):
        inst = super().__new__(cls)
        torch.nn.Module.__init__(inst)
        return inst

    def __lazy_post_init__(self):
        ModelStorageMixin.__lazy_post_init__(self)
        BaseDynamicsModel.__lazy_post_init__(self)
        self.init_parameters()

    def init_parameters(self):
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        self._init_parameters()

    @abstractmethod
    def _init_parameters(self):
        pass


@jaxtyped(typechecker=typechecked)
@torch_dataclass
class GumbelSwitchingModel(SwitchingModel, ABC):
    """
    These types of switching models don't predict the mode directly,
    but rather model the distribution of the modes and sample via gumbel_softmax in the forward pass.
    """

    tau: float = state_field(default=1.0,)

    hard_sampling: bool = config_field(default=False, skip_default=True)

    def forward(
        self,
        *,
        num_samples: int,
        **kwargs,
    ) -> Float[Tensor, "batch {num_samples} num_modes"]:
        """
        Returns the logits used for sampling via gumbel_softmax as well as the sampled modes.
        """

        if self.training:
            tau = self.tau
            hard_sampling = self.hard_sampling
        else:
            # during evaluation we use a lower temperature
            tau = 1e-16
            hard_sampling = True

        logits = self.logits(**kwargs)
        # expand logits to sample num_samples times
        logits = logits.unsqueeze(-2).expand(-1, num_samples, -1)

        # use gumple softmax for training
        gumbel_sampels = F.gumbel_softmax(
            logits,
            tau=tau,
            hard=hard_sampling,
        )

        return gumbel_sampels

    @abstractmethod
    def logits(
        self,
        x: Float[Tensor, "batch dim"],
        x_pred_all_modes: Float[Tensor, "batch num_modes dim"],
        **kwargs,
    ) -> Float[Tensor, "batch num_modes"]:
        """
        Returns the logits used for sampling via gumbel_softmax.
        """


@jaxtyped(typechecker=typechecked)
def probabilities_to_logits(probabilities, epsilon=1e-16):
    # Adjust probabilities to avoid log(0)
    adjusted_probabilities = torch.clamp(probabilities, epsilon, 1)
    logits = torch.log(adjusted_probabilities / torch.clamp(
        (1 - adjusted_probabilities), epsilon, float('inf')))
    return logits


@jaxtyped(typechecker=typechecked)
@torch_dataclass
class PredictionErrorSwitchingModel(GumbelSwitchingModel, ABC):
    """
    Switching model that predicts the mode based on the prediction error.
    """

    def _init_parameters(self):
        super()._init_parameters()
        # this is actually a parameter free model
        # however because of the inheritance from torch.nn.Module we need a parameter
        self.dummy_parameter = torch.nn.Parameter(torch.zeros(1))


@jaxtyped(typechecker=typechecked)
@torch_dataclass
class MSESwitchingModel(PredictionErrorSwitchingModel):
    """
    Switching model that predicts the mode based on the prediction error.
    Uses the MSE between the ground truth and the prediction to determine the logits.
    """

    mse_to_logits: MSEToLogitFunction = config_field(
        default_factory=ScaledInverseMSE)

    @jaxtyped(typechecker=typechecked)
    def logits(
        self,
        x: Float[Tensor, "batch dim"],
        x_pred_all_modes: Float[Tensor, "batch num_modes dim"],
        **kwargs,
    ) -> Float[Tensor, "batch num_modes"]:
        # Compute MSE between predictions and targets
        mse = torch.mean((x.unsqueeze(1) - x_pred_all_modes)**2, dim=-1)

        return self.mse_to_logits(mse)
