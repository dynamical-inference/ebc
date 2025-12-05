from abc import ABC
from abc import abstractmethod
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path
from typing import Dict, List, Optional, TypeVar, Union, Tuple

import numpy as np
import torch
from jaxtyping import Float
from jaxtyping import Integer
from jaxtyping import jaxtyped
from jaxtyping import Shaped
from torch import Tensor
from tqdm import tqdm
from jaxtyping._typeguard import typechecked
from groupcl.datasets.base_synthetic import BaseSyntheticDataset, BaseSyntheticPairedDataset
from groupcl.datasets.paired_actions_single import SinglePairedActionsDataset
from groupcl.datasets.infinite_dsprites import ConfigurableInfiniteDSprites
from groupcl.models.dynamics.linear_dynamics import LinearDynamicsModel
from groupcl.models.mixing import IdentityMixingModel
from groupcl.models.mixing import MixingModel
from groupcl.utils import datatypes as dt
from groupcl.utils.torch import HasDevice
from groupcl.datasets import matrix_transforms as mt

from config_dataclass import Configurable, config_dataclass, config_field, check_initialized

T = TypeVar("T")


@jaxtyped(typechecker=typechecked)
@config_dataclass
class StartingPointsSampler(Configurable, ABC):

    @jaxtyped(typechecker=typechecked)
    @abstractmethod
    def sample(
        self,
        num_samples: int,
        dim: int,
    ) -> Float[Tensor, "{num_samples} {dim}"]:
        pass

    @jaxtyped(typechecker=typechecked)
    def sample_dual(
        self,
        num_samples: int,
        dim: int,
    ) -> Tuple[
            Float[Tensor, "{num_samples} {dim}"],
            Float[Tensor, "{num_samples} {dim}"],
    ]:
        return (
            self.sample(num_samples, dim),
            self.sample(num_samples, dim),
        )


@jaxtyped(typechecker=typechecked)
@config_dataclass
class HypersphereSampler(StartingPointsSampler):

    @jaxtyped(typechecker=typechecked)
    def sample(
        self,
        num_samples: int,
        dim: int,
    ) -> Float[Tensor, "{num_samples} {dim}"]:
        # generate random points from a standard normal distribution and then project them onto the unit sphere
        latent_1 = torch.randn(num_samples, dim)
        # Project onto unit hypersphere by normalizing
        latent_1 = latent_1 / torch.norm(latent_1, dim=1, keepdim=True)

        return latent_1


@jaxtyped(typechecker=typechecked)
@config_dataclass
class BoxSampler(StartingPointsSampler):

    box_size: float = config_field(default=1.0,
                                   help="Size of the box to sample points from")

    @jaxtyped(typechecker=typechecked)
    def sample(
        self,
        num_samples: int,
        dim: int,
    ) -> Float[Tensor, "{num_samples} {dim}"]:
        # generate random points from a uniform distribution and then scale them to the box size
        points_1 = torch.rand(num_samples, dim) * self.box_size
        return points_1


@jaxtyped(typechecker=typechecked)
@config_dataclass
class NxNMatrixSampler(StartingPointsSampler):

    value_min: float = config_field(default=0.0)
    value_max: float = config_field(default=1.0)
    value_size: int = config_field(default=2)

    @property
    @jaxtyped(typechecker=typechecked)
    def value_range(self) -> Float[np.ndarray, "num_values"]:
        return np.linspace(self.value_min, self.value_max, self.value_size)

    @jaxtyped(typechecker=typechecked)
    def sample(
        self,
        num_samples: int,
        dim: int,
    ) -> Float[Tensor, "{num_samples} {dim}"]:
        # use np random choice
        return torch.tensor(
            np.random.choice(
                self.value_range,
                size=(num_samples, dim),
                replace=True,
            ),
            dtype=torch.float32,
        )


@jaxtyped(typechecker=typechecked)
@config_dataclass
class SyntheticPairedRotationsDataset(BaseSyntheticPairedDataset):
    """
    Dataset of paired points under some transformations with mixing.
    
    For each group action, we:
    1. Sample pairs of latent points (latent_1, latent_2)
    2. Apply some transformation to get (latent_1_prime, latent_2_prime)
    3. Apply mixing to get observed variables (observed_1, observed_2, observed_1_prime, observed_2_prime)
    """

    starting_points_sampler: StartingPointsSampler = config_field(
        default_factory=HypersphereSampler)
    mixing_model: MixingModel = config_field(
        default_factory=IdentityMixingModel)
    dynamics_model: LinearDynamicsModel = config_field(
        default_factory=LinearDynamicsModel)

    num_samples: int = config_field(default=1000)

    @property
    def latent_dim(self) -> int:
        return self.mixing_model.input_dim

    @property
    def observed_dim(self) -> int:
        return self.mixing_model.output_dim

    @property
    def num_group_elements(self) -> int:
        return self.dynamics_model.num_systems

    @property
    def samples_per_group_element(self) -> Dict[int, int]:
        """
        Returns a dictionary mapping each group element to the number of samples that should be generated for that group element.
        The samples are distributed as evenly as possible across the group elements.
        """
        naive_samples_per_group_element = self.num_samples // self.num_group_elements
        remaining_samples = self.num_samples - naive_samples_per_group_element * self.num_group_elements

        samples_per_group_element = {
            i: naive_samples_per_group_element
            for i in range(self.num_group_elements)
        }
        for i in range(remaining_samples):
            samples_per_group_element[i % self.num_group_elements] += 1
        return samples_per_group_element

    @jaxtyped(typechecker=typechecked)
    def _generate_starting_points(
        self
    ) -> Tuple[
            Float[Tensor, "num_samples latent_dim"],
            Float[Tensor, "num_samples latent_dim"],
    ]:
        return self.starting_points_sampler.sample_dual(
            num_samples=self.num_samples,
            dim=self.latent_dim,
        )

    @torch.no_grad()
    def _generate_data(
            self
    ) -> Dict[str, Union[Float[Tensor, "..."], Integer[Tensor, "..."]]]:
        """Generate data."""
        latent_x, latent_y = self._generate_starting_points()
        latent_x = latent_x.to(self.device)
        latent_y = latent_y.to(self.device)
        # next we need to determine the group element to apply to each pair of latent points
        # we use the samples_per_group_element dictionary to determine this
        actions_idx = []
        for system_id, num_samples in self.samples_per_group_element.items():
            actions_idx.append(torch.full((num_samples,), system_id))
        actions_idx = torch.cat(actions_idx, dim=0).to(self.device)

        # now we apply the dynamics model to each pair of latent points
        self.dynamics_model.to(self.device)
        latent_x_prime = self.dynamics_model(x=latent_x, system_idx=actions_idx)
        latent_y_prime = self.dynamics_model(x=latent_y, system_idx=actions_idx)

        # now we apply the mixing model to each pair of latent points
        self.mixing_model.to(self.device)
        x = self.mixing_model(latent_x)
        y = self.mixing_model(latent_y)
        x_prime = self.mixing_model(latent_x_prime)
        y_prime = self.mixing_model(latent_y_prime)

        return dict(
            x=x,
            y=y,
            x_prime=x_prime,
            y_prime=y_prime,
            latents_x=latent_x,
            latents_y=latent_y,
            latents_x_prime=latent_x_prime,
            latents_y_prime=latent_y_prime,
            actions_idx=actions_idx,
        )

    @property
    @check_initialized
    def ground_truth_data(self) -> dt.GroundTruthData:
        gt_batch = super().ground_truth_data
        gt_batch.dynamics_model = self.dynamics_model
        return gt_batch

    def to(self: T, device: torch.device) -> T:
        self.dynamics_model.to(device)
        self.mixing_model.to(device)
        return super().to(device)


@jaxtyped(typechecker=typechecked)
@config_dataclass
class BaseSyntheticSinglePairedDataset(
        BaseSyntheticDataset,
        SinglePairedActionsDataset,
        ABC,
):
    """
    Synthetic dataset of paired points under some transformations.
    
    For each group action, we:
    1. Sample a latent point (latent_1) and a transformation index (actions_idx)
    2. Apply the transformation to get (latent_1_prime)
    3. Apply some mixing to get (observed_1)
    """

    @property
    def _data_keys(self) -> List[str]:
        """
        Returns a list of keys of the self._data dictionary that can be indexed (and are required to be present).
        The first key is used to determine the length of the dataset.
        All keys are passed to _validate_types for type checking.
        """
        return [
            "x",
            "x_prime",
            "latents_x",
            "latents_x_prime",
            "actions_idx",
        ]

    @jaxtyped(typechecker=typechecked)
    def _validate_data_types(
        self,
        x: Float[Tensor, "num_samples feature_dim"],
        x_prime: Float[Tensor, "num_samples feature_dim"],
        latents_x: Float[Tensor, "num_samples latent_dim"],
        latents_x_prime: Float[Tensor, "num_samples latent_dim"],
        actions_idx: Integer[Tensor, "num_samples"],
        **kwargs,
    ):
        """Call this function to run jaxtyping typechecks on the data."""
        pass

    @jaxtyped(typechecker=typechecked)
    def get_observed_data(
            self,
            indices: Integer[Tensor,
                             " *batch_shape"]) -> dt.SinglePairedGroupData:
        """Get a batch of data from the dataset."""
        return dt.SinglePairedGroupData(
            x=self._data["x"][indices],
            x_prime=self._data["x_prime"][indices],
            indices=indices,
            actions_idx=self._data["actions_idx"][indices],
        )

    @jaxtyped(typechecker=typechecked)
    def get_latent_data(
            self,
            indices: Integer[Tensor,
                             " *batch_shape"]) -> dt.SinglePairedGroupData:
        """Get the latent data of the dataset."""
        return dt.SinglePairedGroupData(
            x=self._data["latents_x"][indices],
            x_prime=self._data["latents_x_prime"][indices],
            indices=indices,
            actions_idx=self._data["actions_idx"][indices],
        )

    @jaxtyped(typechecker=typechecked)
    def get_action_idx(
        self, indices: Integer[Tensor, " *batch_shape"]
    ) -> Integer[Tensor, " *batch_shape"]:

        return self._data["actions_idx"][indices]

    def __len__(self) -> int:
        return self._data["x"].shape[0]

    @property
    @check_initialized
    def ground_truth_data(self) -> dt.GroundTruthData:
        gt_batch = super().ground_truth_data
        gt_batch.latents = self.get_latent_data(gt_batch.index)
        gt_batch.actions_idx = self._data["actions_idx"][gt_batch.index]
        return dt.GroundTruthData.from_batch(gt_batch)


@jaxtyped(typechecker=typechecked)
@config_dataclass
class SyntheticSinglePairedRotationsDataset(BaseSyntheticSinglePairedDataset):
    """
    Dataset of paired points under some transformations with mixing.
    
    For each group action, we:
    1. Sample a latent point (latent_1) and a transformation index (actions_idx)
    2. Apply the transformation to get (latent_1_prime)
    3. Apply some mixing to get (observed_1)
    """

    starting_points_sampler: StartingPointsSampler = config_field(
        default_factory=HypersphereSampler)
    mixing_model: MixingModel = config_field(
        default_factory=IdentityMixingModel)
    dynamics_model: LinearDynamicsModel = config_field(
        default_factory=LinearDynamicsModel)

    num_samples: int = config_field(default=1000)

    @property
    def latent_dim(self) -> int:
        return self.mixing_model.input_dim

    @property
    def observed_dim(self) -> int:
        return self.mixing_model.output_dim

    @property
    def num_group_elements(self) -> int:
        return self.dynamics_model.num_systems

    @property
    def samples_per_group_element(self) -> Dict[int, int]:
        """
        Returns a dictionary mapping each group element to the number of samples that should be generated for that group element.
        The samples are distributed as evenly as possible across the group elements.
        """
        naive_samples_per_group_element = self.num_samples // self.num_group_elements
        remaining_samples = self.num_samples - naive_samples_per_group_element * self.num_group_elements

        samples_per_group_element = {
            i: naive_samples_per_group_element
            for i in range(self.num_group_elements)
        }
        for i in range(remaining_samples):
            samples_per_group_element[i % self.num_group_elements] += 1
        return samples_per_group_element

    @jaxtyped(typechecker=typechecked)
    def _generate_starting_points(
            self) -> Float[Tensor, "num_samples dynamics_dim"]:
        return self.starting_points_sampler.sample(
            num_samples=self.num_samples,
            dim=self.dynamics_model.dim,
        )

    @torch.no_grad()
    def _generate_data(
            self
    ) -> Dict[str, Union[Float[Tensor, "..."], Integer[Tensor, "..."]]]:
        """Generate data."""
        latent_x = self._generate_starting_points()
        latent_x = latent_x.to(self.device)
        # next we need to determine the group element to apply to each pair of latent points
        # we use the samples_per_group_element dictionary to determine this
        actions_idx = []
        for system_id, num_samples in self.samples_per_group_element.items():
            actions_idx.append(torch.full((num_samples,), system_id))
        actions_idx = torch.cat(actions_idx, dim=0).to(self.device)

        # now we apply the dynamics model to each pair of latent points
        self.dynamics_model.to(self.device)
        latent_x_prime = self.dynamics_model(x=latent_x, system_idx=actions_idx)

        # now we apply the mixing model to each pair of latent points
        self.mixing_model.to(self.device)
        x = self.mixing_model(latent_x)
        x_prime = self.mixing_model(latent_x_prime)

        return dict(
            x=x,
            x_prime=x_prime,
            latents_x=latent_x,
            latents_x_prime=latent_x_prime,
            actions_idx=actions_idx,
        )

    @property
    @check_initialized
    def ground_truth_data(self) -> dt.GroundTruthData:
        gt_batch = super().ground_truth_data
        gt_batch.dynamics_model = self.dynamics_model
        return gt_batch

    def to(self: T, device: torch.device) -> T:
        self.dynamics_model.to(device)
        self.mixing_model.to(device)
        return super().to(device)


@jaxtyped(typechecker=typechecked)
@config_dataclass
class ContentEmbedding(Configurable, HasDevice, ABC):
    """
    A class for embedding content classes into a vector space.
    """
    num_classes: int = config_field(default=10)
    dim: int = config_field(default=2)
    seed: int = config_field(default=42)
    embeddings: Tensor = field(default_factory=lambda: torch.tensor([]),
                               init=False)
    noise: float = config_field(default=0.0, skip_default=True)

    def __lazy_post_init__(self, *args, **kwargs):
        super().__lazy_post_init__(*args, **kwargs)
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        self.embeddings = self.generate_content_embeddings()
        # unset the seed
        torch.random.seed()
        np.random.seed()

    @abstractmethod
    def generate_content_embeddings(
            self) -> Float[Tensor, "{self.num_classes} {self.dim}"]:
        pass

    @jaxtyped(typechecker=typechecked)
    def embed(
        self, content_classes: Integer[Tensor, "num_samples"]
    ) -> Float[Tensor, "num_samples content_dim"]:
        emb = self.embeddings[content_classes]
        if self.noise > 0:
            emb = emb + torch.randn_like(emb) * self.noise
        return emb

    @jaxtyped(typechecker=typechecked)
    def sample_class_idx(self,
                         num_samples: int) -> Integer[Tensor, "{num_samples}"]:
        return torch.randint(
            0,
            self.num_classes,
            (num_samples,),
            device=self.device,
        )

    def _save_additional(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        emb_path = path / "embeddings.pt"
        torch.save(self.embeddings.cpu(), emb_path)

    def _load_additional(self, path: Path) -> None:
        emb_path = path / "embeddings.pt"
        self.embeddings = torch.load(emb_path,
                                     weights_only=False,
                                     map_location=torch.device('cpu'))

    def to(self: T, device: torch.device) -> T:
        self.embeddings = self.embeddings.to(device)
        return super().to(device)


class HypersphereContentEmbedding(ContentEmbedding):

    @jaxtyped(typechecker=typechecked)
    def generate_content_embeddings(
            self) -> Float[Tensor, "{self.num_classes} {self.dim}"]:
        x = torch.randn(self.num_classes, self.dim)
        x = x / torch.norm(x, dim=1, keepdim=True)
        return x


@jaxtyped(typechecker=typechecked)
@config_dataclass
class SyntheticContentDataset(SyntheticSinglePairedRotationsDataset):
    """
    Adds a content vector to the latent space, which is not transformed by the group action.
    """

    content_embedding: ContentEmbedding = config_field(
        default_factory=HypersphereContentEmbedding)

    def validate_config(self):
        # check that content_dim and dynamics_model.dim sum up to latent_dim
        sum_dim = self.content_embedding.dim + self.dynamics_model.dim
        if sum_dim != self.latent_dim:
            raise ValueError(
                f"latent_dim ({self.latent_dim}) must be equal to content_dim ({self.content_embedding.dim}) + dynamics_model.dim ({self.dynamics_model.dim})"
            )

    @property
    def _data_keys(self) -> List[str]:
        """
        Returns a list of keys of the self._data dictionary that can be indexed (and are required to be present).
        The first key is used to determine the length of the dataset.
        All keys are passed to _validate_types for type checking.
        """
        return super()._data_keys + ["class_idx"]

    @torch.no_grad()
    def _generate_data(
            self
    ) -> Dict[str, Union[Float[Tensor, "..."], Integer[Tensor, "..."]]]:
        """Generate data."""
        self.content_embedding.to(self.device)
        latent_x = self._generate_starting_points()
        latent_x = latent_x.to(self.device)
        # next we need to determine the group element to apply to each pair of latent points
        # we use the samples_per_group_element dictionary to determine this
        actions_idx = []
        for system_id, num_samples in self.samples_per_group_element.items():
            actions_idx.append(torch.full((num_samples,), system_id))
        actions_idx = torch.cat(actions_idx, dim=0).to(self.device)

        # now we apply the dynamics model to each pair of latent points
        self.dynamics_model.to(self.device)
        latent_x_prime = self.dynamics_model(x=latent_x, system_idx=actions_idx)

        # sample and add content embeddings
        class_idx = self.content_embedding.sample_class_idx(
            num_samples=latent_x.shape[0])
        latent_content = self.content_embedding.embed(class_idx)
        latent_x = torch.cat([latent_x, latent_content], dim=-1)
        latent_x_prime = torch.cat([latent_x_prime, latent_content], dim=-1)

        # now we apply the mixing model to each pair of latent points
        self.mixing_model.to(self.device)
        x = self.mixing_model(latent_x)
        x_prime = self.mixing_model(latent_x_prime)

        return dict(
            x=x,
            x_prime=x_prime,
            latents_x=latent_x,
            latents_x_prime=latent_x_prime,
            actions_idx=actions_idx,
            class_idx=class_idx,
        )

    @jaxtyped(typechecker=typechecked)
    def get_observed_data(
            self,
            indices: Integer[Tensor,
                             " *batch_shape"]) -> dt.SinglePairedGroupData:
        """Get a batch of data from the dataset."""
        obs_data = super().get_observed_data(indices=indices)
        obs_data.class_idx = self._data["class_idx"][indices]
        return obs_data

    @jaxtyped(typechecker=typechecked)
    def get_latent_data(
            self,
            indices: Integer[Tensor,
                             " *batch_shape"]) -> dt.SinglePairedGroupData:
        """Get the latent data of the dataset."""
        latent_data = super().get_latent_data(indices=indices)
        latent_data.class_idx = self._data["class_idx"][indices]
        return latent_data

    @property
    @check_initialized
    def ground_truth_data(self) -> dt.GroundTruthData:
        gt_batch = super().ground_truth_data
        gt_batch.class_idx = self._data["class_idx"]
        return gt_batch

    def to(self: T, device: torch.device) -> T:
        self.content_embedding.to(device)
        return super().to(device)


@jaxtyped(typechecker=typechecked)
@config_dataclass
class NxNSyntheticDataset(BaseSyntheticSinglePairedDataset):
    """
    In this dataset, we don't have latents, but instead directly sample observed data. 
    Our observed data are NxN matrices, which we apply some group action to in the observation space. 
    
    """

    starting_points_sampler: StartingPointsSampler = config_field(
        default_factory=NxNMatrixSampler)
    num_samples: int = config_field(default=1000)
    matrix_dim: int = config_field(default=7)
    augmentations: mt.MatrixTransformCollection = config_field(
        default_factory=mt.MatrixTransformCollection)

    @property
    def observed_dim(self) -> int:
        return self.matrix_dim * self.matrix_dim

    @jaxtyped(typechecker=typechecked)
    def _generate_starting_points(
        self,
        num_samples: int,
    ) -> Float[Tensor, "{num_samples} {self.observed_dim}"]:
        return self.starting_points_sampler.sample(
            num_samples=num_samples,
            dim=self.observed_dim,
        )

    @property
    def num_group_elements(self) -> int:
        return len(self.augmentations)

    @property
    def samples_per_group_element(self) -> Dict[int, int]:
        """
        Returns a dictionary mapping each group element to the number of samples that should be generated for that group element.
        The samples are distributed as evenly as possible across the group elements.
        """
        naive_samples_per_group_element = self.num_samples // self.num_group_elements
        remaining_samples = self.num_samples - naive_samples_per_group_element * self.num_group_elements

        samples_per_group_element = {
            i: naive_samples_per_group_element
            for i in range(self.num_group_elements)
        }
        for i in range(remaining_samples):
            samples_per_group_element[i % self.num_group_elements] += 1
        return samples_per_group_element

    @torch.no_grad()
    def _generate_data(self) -> Dict[str, Tensor]:
        return self._sample_data(num_samples=self.num_samples)

    def _sample_data(self, num_samples: int) -> Dict[str, Tensor]:
        """Generate data."""
        x = self._generate_starting_points(num_samples=num_samples)
        x = x.to(self.device)

        actions_idx = []
        for system_id, num_samples in self.samples_per_group_element.items():
            actions_idx.append(torch.full((num_samples,), system_id))
        actions_idx = torch.cat(actions_idx, dim=0).to(self.device)

        x_prime = self.augmentations(x, transform_idx=actions_idx)

        return dict(
            x=x,
            x_prime=x_prime,
            actions_idx=actions_idx,
        )

    @property
    def _data_keys(self) -> List[str]:
        """
        Returns a list of keys of the self._data dictionary that can be indexed (and are required to be present).
        The first key is used to determine the length of the dataset.
        All keys are passed to _validate_types for type checking.
        """
        return [
            "x",
            "x_prime",
            "actions_idx",
        ]

    @jaxtyped(typechecker=typechecked)
    def _validate_data_types(
        self,
        x: Float[Tensor, "num_samples feature_dim"],
        x_prime: Float[Tensor, "num_samples feature_dim"],
        actions_idx: Integer[Tensor, "num_samples"],
        **kwargs,
    ):
        """Call this function to run jaxtyping typechecks on the data."""
        pass

    def get_latent_data(
        self, indices: Integer[Tensor, " *batch_shape"]
    ) -> Optional[dt.SinglePairedGroupData]:
        """Get the latent data of the dataset."""
        return None

    def __len__(self) -> int:
        return len(self._data["x"])


@jaxtyped(typechecker=typechecked)
def basis_transform(x: Tensor, mixed_basis: Tensor) -> Tensor:
    return torch.matmul(x.float(), mixed_basis.float()).long()


@jaxtyped(typechecker=typechecked)
def basis_transform_inverse(inv_x: Tensor, mixed_basis: Tensor) -> Tensor:

    inv_x_idx_arr = inv_x  # Ensure input is a torch tensor
    original_shape = inv_x_idx_arr.shape
    inv_x_flat = inv_x_idx_arr.flatten()  # Work with a flat array

    num_actions = len(mixed_basis)
    batch_size = inv_x_flat.shape[0]
    x_out = torch.zeros((batch_size, num_actions), dtype=torch.long)

    # Make a copy to avoid modifying the input tensor
    current_indices = inv_x_flat.clone()

    # Iterate through actions from most significant to least
    for i in range(num_actions):
        # The base corresponding to the current action dimension
        current_base = mixed_basis[i]

        # The value of the current action is how many times the base fits
        x_out[:, i] = current_indices // current_base

        # The remainder carries the information for the subsequent actions
        current_indices %= current_base

    # If the original input was a scalar, return a 1D array
    if not original_shape:  # original input was scalar
        return x_out.squeeze(dim=0)
    # Reshape the output to match the batch dimensions of the input indices
    elif len(original_shape) > 1:
        return x_out.reshape(*original_shape, num_actions)
    # Otherwise return the (batch_size, num_actions) array
    else:
        return x_out


@jaxtyped(typechecker=typechecked)
@config_dataclass
class DSpritesActionSampler(Configurable, HasDevice):
    """
    Implements a sampling procedture that returns random actions, where actions are collection of transformations (scale, orientation, position). 
    This class assume the dataset is shaped as [color, shape, scale, orientation, posX, posY].
    """
    lazy: bool = field(default=True)
    group_combinations: List[List[str]] = config_field(default_factory=lambda: [
        # ["scale"],
        # ["orientation"],
        # ["posX"],
        # ["posY"],
        ["scale", "orientation", "posX", "posY"],
    ])

    @jaxtyped(typechecker=typechecked)
    def __lazy_post_init__(
        self,
        latent_names: List[str],
        latent_sizes: Integer[Tensor, "latent_dim"],
    ):

        self.latent_names = latent_names
        self.latent_sizes = latent_sizes
        # mask is of shape (len(action_space), 6) and indicates which dimensions may change at once
        self.action_dim_mask = torch.stack(
            [(torch.tensor([latent_names.index(a_)
                            for a_ in a]).unsqueeze(1) == torch.arange(
                                len(latent_names)).unsqueeze(0)).any(0)
             for a in self.group_combinations],
            dim=0,
        )

        self.valid_action_names = [
            "scale",
            "orientation",
            "posX",
            "posY",
        ]
        self.action_names = [
            n for n in self.valid_action_names if n in self.latent_names
        ]
        self.action_mask = torch.tensor(
            [n in self.action_names for n in self.latent_names],
            dtype=torch.bool,
        )
        self.action_sizes = torch.tensor([
            self.latent_sizes[self.latent_names.index(a)]
            for a in self.action_names
        ],
                                         device=self.device)

        # Calculate actions_base using torch operations
        action_sizes_flipped = self.action_sizes.flip(0)
        cumprod = torch.cumprod(action_sizes_flipped, dim=0).flip(0)
        self.actions_base = torch.cat(
            [cumprod[1:], torch.tensor([1], device=self.device)])

    def action_to_action_idx(
        self, action: Integer[Shaped,
                              "batch action_dim"]) -> Integer[Shaped, "batch"]:
        return basis_transform(action, self.actions_base)

    def action_idx_to_action(
        self, action_idx: Integer[Shaped, "batch"]
    ) -> Integer[Shaped, "batch action_dim"]:
        return basis_transform_inverse(action_idx, self.actions_base)

    @check_initialized
    def sample_action(self, num_samples: int):
        index_change = torch.zeros(
            (num_samples, len(self.latent_names)),
            dtype=torch.long,
            device=self.device,
        )

        # create random index:
        rand_index = torch.stack([
            torch.randint(
                low=0,
                high=int(size.item()),
                size=(num_samples,),
                device=self.device,
            ) for size in self.latent_sizes
        ],
                                 dim=-1)

        rand_action_dims = torch.randint(
            low=0,
            high=len(self.action_dim_mask),
            size=(num_samples,),
            device=self.device,
        )
        rand_mask = self.action_dim_mask[rand_action_dims]

        index_change[rand_mask] = rand_index[rand_mask]

        return index_change, self.action_to_action_idx(
            index_change[..., self.action_mask]).to(self.device)

    def to(self: T, device: torch.device) -> T:
        if self.initialized:
            self.action_dim_mask = self.action_dim_mask.to(device)
            self.action_mask = self.action_mask.to(device)
            self.actions_base = self.actions_base.to(device)
        return super().to(device)


@jaxtyped(typechecker=typechecked)
def min_max_scale(
        tensor: Shaped[Tensor, "*shape"],
        dim: int,
        target_range: List[float] = [0.0, 1.0]) -> Shaped[Tensor, "*shape"]:
    """
    Scales values in a tensor along a specific dimension to a target range.
    
    Args:
        tensor: Input tensor to be scaled
        dim: The dimension along which to scale
        target_range: The target range [min, max] for scaling
        
    Returns:
        Scaled tensor with values in the target range along the specified dimension
    """
    # Get min and max values along the specified dimension, keeping dimensions
    min_vals, _ = torch.min(tensor, dim=dim, keepdim=True)
    max_vals, _ = torch.max(tensor, dim=dim, keepdim=True)

    # Handle the case where min and max are the same (avoid division by zero)
    diff = max_vals - min_vals
    diff = torch.where(diff == 0, torch.ones_like(diff), diff)

    # Scale to [0, 1]
    normalized = (tensor - min_vals) / diff

    # Scale to target range
    target_min, target_max = target_range
    scaled = normalized * (target_max - target_min) + target_min

    return scaled


@jaxtyped(typechecker=typechecked)
@config_dataclass
class DSpritesDataset(BaseSyntheticSinglePairedDataset):

    dsprite_file_path: Union[str, Path] = field(default_factory=lambda: Path(
        "dsprites/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz"),
                                                init=False)

    action_sampler: DSpritesActionSampler = config_field(
        default_factory=DSpritesActionSampler)
    num_samples: int = config_field(default=1000)
    num_samples_per_action: int = config_field(default=1)

    restrict_latents: Optional[Dict[str, int]] = config_field(default=None,
                                                              skip_default=True)
    image_scale_range: Optional[List[float]] = config_field(default=None,
                                                            skip_default=True)

    def __lazy_post_init__(self):
        self.load_dsprites()
        self.action_sampler.to(self.device)
        self.action_sampler.lazy_init(
            latent_names=self.latent_names,
            latent_sizes=self.latent_sizes,
        )
        super().__lazy_post_init__()

    def load_dsprites(self):
        dsprite_file_path = self.root / Path(self.dsprite_file_path)
        assert dsprite_file_path.exists(
        ), f"File {dsprite_file_path} does not exist"
        dataset_zip = np.load(str(dsprite_file_path),
                              allow_pickle=True,
                              encoding='latin1')

        self.metadata = dataset_zip['metadata'][()]
        self._latents_values = torch.tensor(dataset_zip['latents_values'])
        self._latents = torch.tensor(dataset_zip['latents_classes'])
        self._latent_names = list(self.metadata['latents_names'])
        self._latent_sizes = torch.tensor(self.metadata['latents_sizes'])

        imgs: np.ndarray = dataset_zip['imgs']
        self._shaped_imgs = torch.tensor(imgs).reshape(
            *self.latent_sizes.tolist(),
            *imgs.shape[1:],
        )
        # dropping the color channel which is the first dimension
        self._shaped_imgs = self._shaped_imgs.squeeze(0)
        self._latents_values = self._latents_values[..., 1:]
        self._latents = self._latents[..., 1:]
        self._latent_names = self._latent_names[1:]
        self._latent_sizes = self._latent_sizes[1:]

        self._shaped_imgs = self._shaped_imgs.to(self.device)
        self._latents = self._latents.to(self.device)
        self._latent_sizes = self._latent_sizes.to(self.device)

    @property
    def images_tensor(
            self
    ) -> Float[Tensor, "shape scale orientation x y img_dim img_dim"]:
        return self._shaped_imgs

    @property
    def latent_dim(self) -> int:
        return len(self.latent_sizes)

    @property
    def image_dim(self) -> int:
        return 64

    @property
    def observed_dim(self) -> int:
        return self.image_dim * self.image_dim

    @property
    def latent_sizes(self) -> Integer[Tensor, "latent_dim"]:
        return self._latent_sizes

    @property
    def latent_names(self) -> List[str]:
        return self._latent_names

    @jaxtyped(typechecker=typechecked)
    def _generate_starting_points(
        self,
        num_samples: int,
    ) -> Integer[Tensor, "{num_samples} latent_dim"]:
        # sample random from self._latents
        rand_idx = torch.randint(0, self._latents.shape[0], (num_samples,))
        latents = self._latents[rand_idx]
        latents = self.apply_restrictions(latents)
        return latents

    def apply_restrictions(self, latents: Integer[Tensor,
                                                  "num_samples latent_dim"]):
        if self.restrict_latents is not None:
            for latent_name, restrict_value in self.restrict_latents.items():
                latents[...,
                        self.latent_names.index(latent_name)] = restrict_value
        return latents

    @jaxtyped(typechecker=typechecked)
    def sample_transformed_latent(
        self, x: Integer[Tensor, "num_samples latent_dim"]
    ) -> Tuple[Integer[Tensor, "num_samples latent_dim"], Integer[
            Tensor, "num_samples"]]:
        """
        computes x_prime based on x
        """
        # we need each action_id to occure at least num_samples_per_action times.
        # therefore we'll sample less than x.shape[0] actions and repeat them instead.
        num_unique_actions = x.shape[0] // self.num_samples_per_action
        latent_update, action_idx = self.action_sampler.sample_action(
            num_samples=num_unique_actions)
        # now we repeat the actions num_samples_per_action times
        action_idx = action_idx.repeat_interleave(self.num_samples_per_action,
                                                  dim=0)
        latent_update = latent_update.repeat_interleave(
            self.num_samples_per_action, dim=0)
        # it might be that num_unique_actions * num_samples_per_action is not exactly equal to x.shape[0]
        # so we'll need to fill up to batch_size by repeating the last action
        if action_idx.shape[0] < x.shape[0]:
            missing_action_idx = action_idx[-1].repeat(x.shape[0] -
                                                       action_idx.shape[0])
            missing_latent_update = latent_update[-1].repeat(
                x.shape[0] - latent_update.shape[0], 1)
            action_idx = torch.cat([action_idx, missing_action_idx], dim=0)
            latent_update = torch.cat([latent_update, missing_latent_update],
                                      dim=0)

        x_prime = x + latent_update
        x_prime = torch.remainder(x_prime, self.latent_sizes)

        return x_prime, action_idx

    @torch.no_grad()
    def _generate_data(
            self
    ) -> Dict[str, Union[Float[Tensor, "..."], Integer[Tensor, "..."]]]:
        """Generate data."""
        self.action_sampler.to(self.device)
        latent_x = self._generate_starting_points(num_samples=self.num_samples)
        latent_x = latent_x.to(self.device)
        # next we need to determine the group element to apply to each pair of latent points
        # we use the samples_per_group_element dictionary to determine this
        latent_x_prime, actions_idx = self.sample_transformed_latent(latent_x,)

        # sample and add content embeddings
        x_class_idx = self.class_idx_from_latent(latent_x)
        x_prime_class_idx = self.class_idx_from_latent(latent_x_prime)
        assert torch.all(x_class_idx == x_prime_class_idx)

        return dict(
            latents_x=latent_x.long(),
            latents_x_prime=latent_x_prime.long(),
            actions_idx=actions_idx.long(),
        )

    @jaxtyped(typechecker=typechecked)
    def class_idx_from_latent(
        self, latents: Shaped[Tensor, " *batch latent_dim"]
    ) -> Integer[Tensor, " *batch"]:
        content_class_names = ["shape", "scale"]
        # we consider shape and scale as content
        content_latents = torch.stack(
            [
                latents[..., self.latent_names.index(name)]
                for name in content_class_names
            ],
            dim=-1,
        )
        class_sizes = torch.tensor(
            [
                self.latent_sizes[self.latent_names.index(name)]
                for name in content_class_names
            ],
            device=self.device,
        )
        class_sizes_flipped = class_sizes.flip(0)
        cumprod = torch.cumprod(class_sizes_flipped, dim=0).flip(0)
        class_sizes_base = torch.cat(
            [cumprod[1:], torch.tensor([1], device=self.device)])

        # convert to class idx
        class_idx = basis_transform(
            content_latents,
            mixed_basis=class_sizes_base,
        )
        return class_idx.long()

    @jaxtyped(typechecker=typechecked)
    def latent_to_img(
        self, latents: Shaped[Tensor, " *batch latent_dim"]
    ) -> Float[Tensor, " *batch observed_dim"]:
        latents = latents.long()
        latent_shape = latents.shape[:-1]
        flatter_latents = latents.reshape(-1, self.latent_dim)
        flatter_latents = flatter_latents.long()
        tupled_shaped_index = tuple(flatter_latents.T)
        flattened_img = self.images_tensor[tupled_shaped_index].reshape(
            *latent_shape, -1).float()
        if self.image_scale_range is not None:
            flattened_img = min_max_scale(flattened_img,
                                          dim=-1,
                                          target_range=self.image_scale_range)
        return flattened_img

    @jaxtyped(typechecker=typechecked)
    def get_latent_data(
            self,
            indices: Integer[Tensor,
                             " *batch_shape"]) -> dt.SinglePairedGroupData:
        latents_x = self._data["latents_x"][indices]
        latents_x_prime = self._data["latents_x_prime"][indices]
        actions_idx = self._data["actions_idx"][indices]
        class_idx = self.class_idx_from_latent(latents_x)
        return dt.SinglePairedGroupData(
            x=latents_x.float(),
            x_prime=latents_x_prime.float(),
            indices=indices,
            actions_idx=actions_idx,
            class_idx=class_idx,
        )

    @jaxtyped(typechecker=typechecked)
    def get_observed_data(
            self,
            indices: Integer[Tensor,
                             " *batch_shape"]) -> dt.SinglePairedGroupData:

        latent_batch = self.get_latent_data(indices)
        latent_batch.x = self.latent_to_img(latent_batch.x)
        latent_batch.x_prime = self.latent_to_img(latent_batch.x_prime)
        return dt.SinglePairedGroupData.from_batch(latent_batch)

    def to(self: T, device: torch.device) -> T:
        self._shaped_imgs = self._shaped_imgs.to(device)
        self._latents = self._latents.to(device)
        self._latent_sizes = self._latent_sizes.to(device)
        self.action_sampler.to(device)
        return super().to(device)

    @property
    def _data_keys(self) -> List[str]:
        """
        Returns a list of keys of the self._data dictionary that can be indexed (and are required to be present).
        The first key is used to determine the length of the dataset.
        All keys are passed to _validate_types for type checking.
        """
        return [
            "latents_x",
            "latents_x_prime",
            "actions_idx",
            # "class_idx",
        ]

    def __len__(self) -> int:
        return len(self._data["latents_x"])

    @jaxtyped(typechecker=typechecked)
    def _validate_data_types(
        self,
        latents_x: Integer[Tensor, "num_samples latent_dim"],
        latents_x_prime: Integer[Tensor, "num_samples latent_dim"],
        actions_idx: Integer[Tensor, "num_samples"],
        **kwargs,
    ):
        """Call this function to run jaxtyping typechecks on the data."""
        pass


@jaxtyped(typechecker=typechecked)
@config_dataclass
class InfiniteDSpritesDataset(DSpritesDataset):

    action_sampler: DSpritesActionSampler = config_field(
        default_factory=DSpritesActionSampler)
    num_samples: int = config_field(default=1000)
    num_samples_per_action: int = config_field(default=1)
    dsprites: ConfigurableInfiniteDSprites = config_field(
        default_factory=ConfigurableInfiniteDSprites)

    def load_dsprites(self):
        self._latents = self.dsprites._data["latents"]
        self._latent_names = self.dsprites.latent_names
        self._latent_sizes = self.dsprites.latent_sizes
        # the infinite dsprites sometimes uses different latent names, to match the original dsprite dataset, we need to rename them
        name_map = {
            "position_x": "posX",
            "position_y": "posY",
        }
        for old_name, new_name in name_map.items():
            self._latent_names[self._latent_names.index(old_name)] = new_name

        imgs = self.dsprites._data["imgs"]
        self._shaped_imgs = imgs.view(
            *self.latent_sizes.tolist(),
            *imgs.shape[1:],
        )
        # dropping the color channel which is the first dimension
        self._shaped_imgs = self._shaped_imgs.squeeze(0)
        self._latents = self._latents[..., 1:]
        self._latent_names = self._latent_names[1:]
        self._latent_sizes = self._latent_sizes[1:]

        self._shaped_imgs = self._shaped_imgs.to(self.device)
        self._latents = self._latents.to(self.device)
        self._latent_sizes = self._latent_sizes.to(self.device)

    @property
    def image_dim(self) -> int:
        return self.dsprites.img_size
