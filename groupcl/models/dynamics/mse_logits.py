from abc import ABC
from abc import abstractmethod

import torch

from config_dataclass import Configurable, torch_dataclass, config_field


@torch_dataclass
class MSEToLogitFunction(Configurable, ABC):
    """Base class for MSE to logit conversion functions."""

    @abstractmethod
    def __call__(self, mse: torch.Tensor) -> torch.Tensor:
        """Convert MSE to logits."""


@torch_dataclass
class NegativeMSE(MSEToLogitFunction):
    """Converts MSE to logits by taking the negative."""

    def __call__(self, mse: torch.Tensor) -> torch.Tensor:
        return -mse


@torch_dataclass
class ScaledNegativeMSE(MSEToLogitFunction):
    """Converts MSE to logits by taking the negative with scaling."""
    scale: float = config_field(default=1.0)

    def __call__(self, mse: torch.Tensor) -> torch.Tensor:
        return -mse * self.scale


@torch_dataclass
class InverseMSE(MSEToLogitFunction):
    """Converts MSE to logits by taking the inverse."""

    eps: float = config_field(default=1e-16)

    def __call__(self, mse: torch.Tensor) -> torch.Tensor:
        return 1 / (mse + self.eps)


@torch_dataclass
class ScaledInverseMSE(MSEToLogitFunction):
    """Converts MSE to logits by taking the inverse with scaling."""

    eps: float = config_field(default=1e-16)
    scale: float = config_field(default=1.0)

    def __call__(self, mse: torch.Tensor) -> torch.Tensor:
        return self.scale / (mse + (self.eps / self.scale))


@torch_dataclass
class SoftLogMSE(MSEToLogitFunction):
    """Converts MSE to logits using a soft log transformation."""
    eps: float = config_field(default=1e-8)

    def __call__(self, mse: torch.Tensor) -> torch.Tensor:
        return -torch.log(mse + self.eps)


@torch_dataclass
class TemperatureScaledMSE(MSEToLogitFunction):
    """Converts MSE to logits using temperature scaling."""
    tau: float = config_field(default=1.0)

    def __call__(self, mse: torch.Tensor) -> torch.Tensor:
        return -mse / self.tau


@torch_dataclass
class ExponentialMSE(MSEToLogitFunction):
    """Converts MSE to logits using an exponential transformation."""
    alpha: float = config_field(default=1.0)

    def __call__(self, mse: torch.Tensor) -> torch.Tensor:
        return torch.exp(-self.alpha * mse)


@torch_dataclass
class IdentityMSE(MSEToLogitFunction):
    """Returns MSE unchanged."""

    def __call__(self, mse: torch.Tensor) -> torch.Tensor:
        return mse
