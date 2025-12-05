import torch
from typing import TypeVar, Union
from abc import ABC
from dataclasses import dataclass
from dataclasses import field


def expand_dim(tensor: torch.Tensor, dim: int, size: int) -> torch.Tensor:
    """
    Expand a tensor along a specific dimension.
    """
    shape = list(tensor.shape)  # Get current shape
    shape[dim] = size  # Modify only the specified dimension
    return tensor.expand(*shape)


T = TypeVar("T")


@dataclass(kw_only=True)
class HasDevice():
    """Mixin class for objects that have a device."""

    _device: Union[torch.device, str] = field(
        default_factory=lambda: torch.device('cuda')
        if torch.cuda.is_available() else torch.device('cpu'),
        repr=False,
    )

    def __post_init__(self):
        if isinstance(self._device, str):
            self._device = torch.device(self._device)

    @property
    def device(self) -> torch.device:
        return self._device

    def to(self: T, device: torch.device) -> T:
        """Move the data loader to the specified device."""
        self._device = device
        return self
