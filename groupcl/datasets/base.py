from abc import ABC
from abc import abstractmethod
from dataclasses import dataclass
from typing import Dict, List, TypeVar, Union, Any

import torch
from jaxtyping import Integer
from jaxtyping import jaxtyped
from jaxtyping import Shaped
from torch import Tensor
from jaxtyping._typeguard import typechecked

from config_dataclass import Configurable, config_dataclass, check_initialized
from groupcl.utils.torch import HasDevice

T = TypeVar("T")


@jaxtyped(typechecker=typechecked)
@config_dataclass
class BaseDataset(Configurable, HasDevice, ABC):
    """Abstract base class for dataset"""

    @property
    @check_initialized
    def index(self) -> Integer[Tensor, " num_samples"]:
        """Get the index of the dataset."""
        return torch.arange(len(self))

    @abstractmethod
    def __len__(self) -> int:
        """Return number of samples in dataset."""
        raise NotImplementedError("Data length not implemented")

    # no typechecking here, jaxtyping just for docs
    def __getitem__(
        self, idx: Union[int, List[int], Integer[Tensor, " batch_shape"]]
    ) -> Dict[str, Union[
            Shaped[Tensor, " batch_shape ..."],
            Any,
    ]]:
        """Returns a dictionary with all the available data for the given index."""
        if isinstance(idx, int):
            return self.__getitems__(torch.tensor([idx]))
        else:
            return self.__getitems__(idx)

    # no typechecking here, jaxtyping just for docs
    @abstractmethod
    def __getitems__(
        self, indices: Union[List[int], Integer[Tensor, " batch_shape"]]
    ) -> Dict[str, Union[
            Shaped[Tensor, " batch_shape ..."],
            Any,
    ]]:
        """Get a list of samples from dataset."""
        raise NotImplementedError("Data item access not implemented")

    @abstractmethod
    def split(self: T, indices: Integer[Tensor, " batch_shape"]) -> T:
        """Split the dataset into a new dataset with the given indices."""
        raise NotImplementedError("Split not implemented")
