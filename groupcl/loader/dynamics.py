from dataclasses import dataclass
from dataclasses import field
from typing import Dict, Iterator, Tuple, TypeVar

import torch
from jaxtyping import Integer
from jaxtyping import jaxtyped
from torch import Tensor
from jaxtyping._typeguard import typechecked

from groupcl.loader.base import BaseDataLoader

from groupcl.datasets.paired_actions import PairedActionsDataset
from groupcl.loader.base import BaseDataLoader

from groupcl.utils.datatypes import GroundTruthData
from groupcl.utils.datatypes import PairedGroupData
from config_dataclass import config_dataclass, config_field, check_initialized

T = TypeVar("T")


@jaxtyped(typechecker=typechecked)
@config_dataclass
class GroupDataLoader(BaseDataLoader):
    """Data loader that yields batches of sequences from a dataset.
    Used for training models that take a sequence of datapoints as input (i.e. dynamics models).
    """

    shuffle: bool = config_field(default=True)
    _indices: Tensor = field(default_factory=lambda: torch.tensor([]),)

    @jaxtyped(typechecker=typechecked)
    def __lazy_post_init__(
        self,
        dataset: PairedActionsDataset,
    ):
        super().__lazy_post_init__(dataset)

    @property
    @check_initialized
    def indices(self) -> Tensor:
        return self._indices

    @property
    @jaxtyped(typechecker=typechecked)
    def dataset(self) -> PairedActionsDataset:
        return super().dataset

    @check_initialized
    def __len__(self) -> int:
        """Return number of iterations per epoch."""
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size

    @check_initialized
    def __iter__(self) -> Iterator[
            PairedGroupData,
    ]:
        """Return iterator over batches."""
        self.reset()

        for batch_id in range(len(self)):
            yield self.sample_batch(batch_id).to(self.device)

    @check_initialized
    def reset(self):
        """Reset the data loader by reshuffling the indices."""
        indices = self.dataset.index.to(self.device)
        if self.shuffle:
            permutation_index = torch.randperm(len(indices))
            self._indices = indices[permutation_index]
        else:
            self._indices = indices

    @jaxtyped(typechecker=typechecked)
    def sample_batch(self, batch_id: int) -> PairedGroupData:
        batch_indices = self.indices[batch_id * self.batch_size:(batch_id + 1) *
                                     self.batch_size]
        group_data = self.dataset.get_observed_data(batch_indices)

        return group_data

    @property
    def validation_data(self) -> PairedGroupData:
        """Get a batch of data from the dataset for validation."""

        return self.dataset.get_observed_data(self.dataset.index).to(
            self.device)

    @property
    def ground_truth_data(self) -> GroundTruthData:
        """Get the ground truth data for the dataset."""
        return self.dataset.ground_truth_data.to(self.device)

    def to(self: T, device: torch.device) -> T:
        """Move the data loader to the specified device."""
        self._indices = self._indices.to(device)
        return super().to(device)
