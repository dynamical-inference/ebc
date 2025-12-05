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
from groupcl.utils.datatypes import AuxilaryVariables
from groupcl.utils.datatypes import GroundTruthData
from groupcl.utils.datatypes import PairedGroupData

T = TypeVar("T")


@config_dataclass
class PairedActionsDataset(BaseDataset, ABC):
    """
    Dataset of paired actions. 
    Paired actions means we assume there exist some unknown transformation that is applied to two datapoints. 
    Which means each sample of this dataset contains four datapoints: 
    (x, y, x', y') where (x, y) are the original datapoints and (x', y') are the transformed datapoints x' = G(x) and y' = G(y).
    So the two pairs (x, x') and (y, y') are related by the same transformation G.
    """

    @property
    @check_initialized
    def auxilary_variables(self) -> AuxilaryVariables:
        """Get the auxilary variables of the dataset."""
        return AuxilaryVariables()

    @property
    def ground_truth_data(self) -> GroundTruthData:
        """Get the ground truth data for evaluation purposes."""
        return GroundTruthData(
            observed=self.get_observed_data(self.index),
            index=self.index,
            auxilary=self.auxilary_variables,
        )

    @abstractmethod
    def get_observed_data(
            self, indices: Integer[Tensor, " *batch_shape"]) -> PairedGroupData:
        """Get a batch of data from the dataset."""
        raise NotImplementedError("Data batch access not implemented")


@jaxtyped(typechecker=typechecked)
@config_dataclass
class TensorPairedActionsDataset(PairedActionsDataset):
    """In Memory Paired Actions Dataset"""

    x: Float[Tensor, "num_samples feature_dim"]
    y: Float[Tensor, "num_samples feature_dim"]
    x_prime: Float[Tensor, "num_samples feature_dim"]
    y_prime: Float[Tensor, "num_samples feature_dim"]

    @property
    def observed_dim(self) -> int:
        return self.x.shape[-1]

    @jaxtyped(typechecker=typechecked)
    def get_observed_data(
            self, indices: Integer[Tensor, " *batch_shape"]) -> PairedGroupData:
        """Get a batch of data from the dataset."""
        return PairedGroupData(
            x=self.x[indices],
            y=self.y[indices],
            x_prime=self.x_prime[indices],
            y_prime=self.y_prime[indices],
            indices=indices,
        )

    def __getitems__(
        self, indices: Union[List[int], Integer[Tensor, " batch_shape"]]
    ) -> Dict[str, Shaped[Tensor, " batch_shape ..."]]:
        return dict(
            x=self.x[indices],
            y=self.y[indices],
            x_prime=self.x_prime[indices],
            y_prime=self.y_prime[indices],
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
        self.y = self.y.to(device)
        self.x_prime = self.x_prime.to(device)
        self.y_prime = self.y_prime.to(device)
        return super().to(device)


@jaxtyped(typechecker=typechecked)
@config_dataclass
class TensorPairedActionsDatasetWithLatents(TensorPairedActionsDataset):
    """TensorPairedActionsDataset with ground truth data"""

    latents_x: Float[Tensor, "num_samples latent_dim"]
    latents_y: Float[Tensor, "num_samples latent_dim"]
    latents_x_prime: Float[Tensor, "num_samples latent_dim"]
    latents_y_prime: Float[Tensor, "num_samples latent_dim"]

    @property
    def latent_dim(self) -> int:
        return self.latents_x.shape[-1]

    def get_latent_data(
            self, indices: Integer[Tensor, " *batch_shape"]) -> PairedGroupData:
        """Get the latent data of the dataset."""
        return PairedGroupData(
            x=self.latents_x[indices],
            y=self.latents_y[indices],
            x_prime=self.latents_x_prime[indices],
            y_prime=self.latents_y_prime[indices],
            indices=indices,
        )

    @property
    @check_initialized
    def ground_truth_data(self) -> GroundTruthData:
        gt_batch = super().ground_truth_data
        gt_batch.latents = self.get_latent_data(gt_batch.index)
        return GroundTruthData.from_batch(gt_batch)

    def __getitems__(
        self, indices: Union[List[int], Integer[Tensor, " batch_shape"]]
    ) -> Dict[str, Shaped[Tensor, " batch_shape ..."]]:
        batch_dict = super().__getitems__(indices)
        batch_dict.update(
            dict(
                latents_x=self.latents_x[indices],
                latents_y=self.latents_y[indices],
                latents_x_prime=self.latents_x_prime[indices],
                latents_y_prime=self.latents_y_prime[indices],
            ))
        return batch_dict

    def to(self: T, device: torch.device) -> T:
        self.latents_x = self.latents_x.to(device)
        self.latents_y = self.latents_y.to(device)
        self.latents_x_prime = self.latents_x_prime.to(device)
        self.latents_y_prime = self.latents_y_prime.to(device)
        return super().to(device)


@jaxtyped(typechecker=typechecked)
@config_dataclass
class TensorPairedActionsDatasetFromFile(TensorPairedActionsDataset):
    """TensorPairedActionsDataset from a file"""

    data_path: str = config_field()
    x_key: str = config_field(default="x")
    y_key: str = config_field(default="y")
    x_prime_key: str = config_field(default="x_prime")
    y_prime_key: str = config_field(default="y_prime")

    # NOTE: make these optional, so the init doesn't require them
    x: Optional[Float[Tensor, "num_samples feature_dim"]] = None
    y: Optional[Float[Tensor, "num_samples feature_dim"]] = None
    x_prime: Optional[Float[Tensor, "num_samples feature_dim"]] = None
    y_prime: Optional[Float[Tensor, "num_samples feature_dim"]] = None

    def __post_init__(self):
        self.data_dict = torch.load(self.data_path,
                                    map_location=torch.device('cpu'))
        self.x = self.data_dict[self.x_key]
        self.y = self.data_dict[self.y_key]
        self.x_prime = self.data_dict[self.x_prime_key]
        self.y_prime = self.data_dict[self.y_prime_key]
        super().__post_init__()

    @jaxtyped(typechecker=typechecked)
    def split(self: T, indices: Integer[Tensor, " *batch_shape"]) -> T:
        data_dict = self[indices]
        return TensorPairedActionsDataset(**data_dict)


@jaxtyped(typechecker=typechecked)
@config_dataclass
class TensorPairedActionsDatasetWithLatentsFromFile(
        TensorPairedActionsDatasetWithLatents):
    """TensorPairedActionsDatasetWithLatents from a file"""

    data_path: str = config_field()
    x_key: str = config_field(default="x")
    y_key: str = config_field(default="y")
    x_prime_key: str = config_field(default="x_prime")
    y_prime_key: str = config_field(default="y_prime")
    latents_x_key: str = config_field(default="latents_x")
    latents_y_key: str = config_field(default="latents_y")
    latents_x_prime_key: str = config_field(default="latents_x_prime")
    latents_y_prime_key: str = config_field(default="latents_y_prime")

    # NOTE: make these optional, so the init doesn't require them
    x: Optional[Float[Tensor, "num_samples feature_dim"]] = None
    y: Optional[Float[Tensor, "num_samples feature_dim"]] = None
    x_prime: Optional[Float[Tensor, "num_samples feature_dim"]] = None
    y_prime: Optional[Float[Tensor, "num_samples feature_dim"]] = None
    latents_x: Optional[Float[Tensor, "num_samples latent_dim"]] = None
    latents_y: Optional[Float[Tensor, "num_samples latent_dim"]] = None
    latents_x_prime: Optional[Float[Tensor, "num_samples latent_dim"]] = None
    latents_y_prime: Optional[Float[Tensor, "num_samples latent_dim"]] = None

    def __post_init__(self):
        self.data_dict = torch.load(self.data_path,
                                    map_location=torch.device('cpu'))
        self.x = self.data_dict[self.x_key]
        self.y = self.data_dict[self.y_key]
        self.x_prime = self.data_dict[self.x_prime_key]
        self.y_prime = self.data_dict[self.y_prime_key]
        self.latents_x = self.data_dict[self.latents_x_key]
        self.latents_y = self.data_dict[self.latents_y_key]
        self.latents_x_prime = self.data_dict[self.latents_x_prime_key]
        self.latents_y_prime = self.data_dict[self.latents_y_prime_key]
        super().__post_init__()

    @jaxtyped(typechecker=typechecked)
    def split(self: T, indices: Integer[Tensor, " *batch_shape"]) -> T:
        data_dict = self[indices]
        return TensorPairedActionsDatasetWithLatentsFromFile(**data_dict)
