import importlib
import pkgutil
from abc import ABC
from abc import abstractmethod
from dataclasses import dataclass
from dataclasses import field
from typing import Dict, Iterator, Optional, TypeVar

import torch
from jaxtyping import jaxtyped
from jaxtyping._typeguard import typechecked

from groupcl.datasets.base import BaseDataset
from groupcl.utils.torch import HasDevice
from config_dataclass import Configurable, config_dataclass, config_field, check_initialized

T = TypeVar("T")


@config_dataclass
class BaseDataLoader(HasDevice, Configurable, ABC):
    """Base class for iterative data loaders that can be used for training.

    This class uses lazy initialization, meaning the dataset is not loaded immediately
    upon instantiation. Instead, you must call lazy_init() with the dataset after creating
    the loader object:

    Example:
        >>> loader = BaseDataLoader(batch_size=32)  # Create loader without dataset
        >>> loader.lazy_init(dataset)  # Initialize with dataset when ready

    Args:
        batch_size: Size of each batch (default: 32)
        seed: Random seed for shuffling (default: 42)
        lazy: Whether to use lazy initialization (default: True)

    Yields:
        Dict[str, Tensor]: Batches of the specified size from the dataset, with exact
            contents depending on the dataset and loader implementation.

    Note:
        The loader will automatically move batches to the specified device (default: 'cpu').
        Use the .to() method to change devices.
    """

    # By default dataloaders need lazy initialization
    lazy: bool = True

    batch_size: int = config_field(default=32)
    seed: int = config_field(default=42)

    _dataset: Optional[BaseDataset] = field(default=None,
                                            init=False,
                                            repr=False)

    @classmethod
    def config_prefix(cls) -> str:
        """Prefix for the configuration file."""
        return "loader"

    @jaxtyped(typechecker=typechecked)
    def __lazy_post_init__(self, dataset: BaseDataset):
        """Initialize the data loader with a dataset."""
        self._dataset = dataset
        torch.manual_seed(self.seed)
        self.reset()
        self.to(self.device)
        return self

    def validate_config(self):
        """Validate the configuration of the data loader."""
        if self.batch_size is not None and self.batch_size <= 0:
            raise ValueError(
                f"Batch size has to be None, or a non-negative value. Got {self.batch_size}."
            )
        super().validate_config()

    @property
    @check_initialized
    def dataset(self) -> BaseDataset:
        """Get the dataset of the data loader."""
        return self._dataset

    @property
    def device(self) -> torch.device:
        """Get the device of the data loader."""
        return self._device

    def to(self: T, device: torch.device) -> T:
        """Move the data loader to the specified device."""
        self.dataset.to(device)
        return super().to(device)

    @abstractmethod
    def reset(self):
        raise NotImplementedError(
            "reset method must be implemented by subclass.")

    @abstractmethod
    def __len__(self) -> int:
        raise NotImplementedError(
            "__len__ method must be implemented by subclass.")

    @abstractmethod
    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        raise NotImplementedError(
            "__iter__ method must be implemented by subclass.")
