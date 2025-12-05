from abc import ABC
from abc import abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, TypeVar, Tuple, Union

import torch
from jaxtyping import Float
from jaxtyping import Integer
from jaxtyping import jaxtyped
from jaxtyping import Shaped
from torch import Tensor
from jaxtyping._typeguard import typechecked

from groupcl.datasets.base import BaseDataset
from config_dataclass import Configurable, config_dataclass, config_field, check_initialized
from groupcl.utils import datatypes as dt

T = TypeVar("T")


@config_dataclass
class SinglePairedActionsDataset(BaseDataset, ABC):
    """
    Dataset of labeled actions. 
    We assume there's some unknown transformations applied to different datapoints. 
    We don't know the transformations themselves, but we know which points are transformed in the same way.
    Which means each sample of this dataset contains two datapoints: 
    x and x_prime, where x_prime is a transformed version of x: 
    x_prime = G_i(x)
    We also know i, which is the index of the transformation G_i. But G_i is unknown.
    """

    @property
    @check_initialized
    def auxilary_variables(self) -> dt.AuxilaryVariables:
        """Get the auxilary variables of the dataset."""
        return dt.AuxilaryVariables()

    @property
    def ground_truth_data(self) -> dt.GroundTruthData:
        """Get the ground truth data for evaluation purposes."""
        return dt.GroundTruthData(
            observed=self.get_observed_data(self.index),
            index=self.index,
            auxilary=self.auxilary_variables,
        )

    @abstractmethod
    def get_observed_data(
            self,
            indices: Integer[Tensor,
                             " *batch_shape"]) -> dt.SinglePairedGroupData:
        """Get a batch of data from the dataset."""
        raise NotImplementedError("Data batch access not implemented")


@jaxtyped(typechecker=typechecked)
@config_dataclass
class TensorSinglePairedActionsDataset(SinglePairedActionsDataset):
    """In Memory SinglePaired Actions Dataset"""

    x: Float[Tensor, "num_samples feature_dim"]
    x_prime: Float[Tensor, "num_samples feature_dim"]
    actions_idx: Integer[Tensor, "num_samples"]

    @property
    def observed_dim(self) -> int:
        return self.x.shape[-1]

    @jaxtyped(typechecker=typechecked)
    def get_observed_data(
            self,
            indices: Integer[Tensor,
                             " *batch_shape"]) -> dt.SinglePairedGroupData:
        """Get a batch of data from the dataset."""
        return dt.SinglePairedGroupData(
            x=self.x[indices],
            x_prime=self.x_prime[indices],
            indices=indices,
            actions_idx=self.actions_idx[indices],
        )

    def __getitems__(
        self, indices: Union[List[int], Integer[Tensor, " batch_shape"]]
    ) -> Dict[str, Shaped[Tensor, " batch_shape ..."]]:
        return dict(
            x=self.x[indices],
            x_prime=self.x_prime[indices],
            actions_idx=self.actions_idx[indices],
        )

    @jaxtyped(typechecker=typechecked)
    def split(self: T, indices: Integer[Tensor, " *batch_shape"]) -> T:
        """Split the dataset into a new dataset with the given indices."""

        data_dict = self[indices]
        return self.__class__(**data_dict)

    def __len__(self) -> int:
        return self.x.shape[0]

    def to(self: T, device: torch.device) -> T:
        self.x = self.x.to(device)
        self.x_prime = self.x_prime.to(device)
        self.actions_idx = self.actions_idx.to(device)
        return super().to(device)


@jaxtyped(typechecker=typechecked)
@config_dataclass
class TensorSinglePairedActionsDatasetWithLatents(
        TensorSinglePairedActionsDataset):
    """TensorSinglePairedActionsDataset with ground truth data"""

    latents_x: Float[Tensor, "num_samples latent_dim"]
    latents_x_prime: Float[Tensor, "num_samples latent_dim"]

    @property
    def latent_dim(self) -> int:
        return self.latents_x.shape[-1]

    def get_latent_data(
            self,
            indices: Integer[Tensor,
                             " *batch_shape"]) -> dt.SinglePairedGroupData:
        """Get the latent data of the dataset."""
        return dt.SinglePairedGroupData(
            x=self.latents_x[indices],
            x_prime=self.latents_x_prime[indices],
            indices=indices,
            actions_idx=self.actions_idx[indices],
        )

    @property
    @check_initialized
    def ground_truth_data(self) -> dt.GroundTruthData:
        gt_batch = super().ground_truth_data
        gt_batch.latents = self.get_latent_data(gt_batch.index)
        return dt.GroundTruthData.from_batch(gt_batch)

    def __getitems__(
        self, indices: Union[List[int], Integer[Tensor, " batch_shape"]]
    ) -> Dict[str, Shaped[Tensor, " batch_shape ..."]]:
        batch_dict = super().__getitems__(indices)
        batch_dict.update(
            dict(
                latents_x=self.latents_x[indices],
                latents_x_prime=self.latents_x_prime[indices],
            ))
        return batch_dict

    def to(self: T, device: torch.device) -> T:
        self.latents_x = self.latents_x.to(device)
        self.latents_x_prime = self.latents_x_prime.to(device)
        return super().to(device)


@jaxtyped(typechecker=typechecked)
@config_dataclass
class TensorSinglePairedActionsDatasetFromFile(TensorSinglePairedActionsDataset
                                              ):
    """TensorSinglePairedActionsDataset from a file"""

    data_path: str = config_field()
    x_key: str = config_field(default="x")
    x_prime_key: str = config_field(default="x_prime")
    actions_idx_key: str = config_field(default="actions_idx")

    # NOTE: make these optional, so the init doesn't require them
    x: Optional[Float[Tensor, "num_samples feature_dim"]] = None
    x_prime: Optional[Float[Tensor, "num_samples feature_dim"]] = None
    actions_idx: Optional[Integer[Tensor, "num_samples"]] = None

    def __post_init__(self):
        self.data_dict = torch.load(self.data_path,
                                    map_location=torch.device('cpu'))
        self.x = self.data_dict[self.x_key]
        self.x_prime = self.data_dict[self.x_prime_key]
        self.actions_idx = self.data_dict[self.actions_idx_key]
        super().__post_init__()


@jaxtyped(typechecker=typechecked)
@config_dataclass
class TensorSinglePairedActionsDatasetWithLatentsFromFile(
        TensorSinglePairedActionsDatasetWithLatents):
    """TensorSinglePairedActionsDatasetWithLatents from a file"""

    data_path: str = config_field()
    x_key: str = config_field(default="x")
    x_prime_key: str = config_field(default="x_prime")
    actions_idx_key: str = config_field(default="actions_idx")
    latents_x_key: str = config_field(default="latents_x")
    latents_x_prime_key: str = config_field(default="latents_x_prime")

    # NOTE: make these optional, so the init doesn't require them
    x: Optional[Float[Tensor, "num_samples feature_dim"]] = None
    x_prime: Optional[Float[Tensor, "num_samples feature_dim"]] = None
    actions_idx: Optional[Integer[Tensor, "num_samples"]] = None
    latents_x: Optional[Float[Tensor, "num_samples latent_dim"]] = None
    latents_x_prime: Optional[Float[Tensor, "num_samples latent_dim"]] = None

    def __post_init__(self):
        self.data_dict = torch.load(self.data_path,
                                    map_location=torch.device('cpu'))
        self.x = self.data_dict[self.x_key]
        self.x_prime = self.data_dict[self.x_prime_key]
        self.actions_idx = self.data_dict[self.actions_idx_key]
        self.latents_x = self.data_dict[self.latents_x_key]
        self.latents_x_prime = self.data_dict[self.latents_x_prime_key]
        super().__post_init__()
