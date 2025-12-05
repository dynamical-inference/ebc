import warnings
from abc import ABC, abstractmethod
from dataclasses import field
from typing import Iterator, List, Literal, Optional, Tuple, TypeVar

import torch
import torch.nn.utils.rnn as rnn_utils
from config_dataclass import check_initialized, config_dataclass, config_field
from jaxtyping import Integer, jaxtyped
from jaxtyping._typeguard import typechecked
from torch import Tensor

from groupcl.datasets.cebra import HippcampusRatDataset
from groupcl.datasets.paired_actions import PairedActionsDataset
from groupcl.datasets.paired_actions_single import SinglePairedActionsDataset
from groupcl.datasets.synthetic import (
    DSpritesDataset,
    SyntheticSinglePairedRotationsDataset,
)
from groupcl.loader.base import BaseDataLoader
from groupcl.utils import datatypes as dt
from groupcl.utils.behavior import (
    bin_differences,
    discretize_variable,
    filter_binned_differences,
    generate_group_factors,
    generate_group_factors_v2,
    paired_differences,
)
from groupcl.utils.datatypes import (
    ContrastiveGroupBatch,
    GroundTruthData,
    PairedData,
    PairedGroupData,
)


def internal_tensor_field(**kwargs):
    return field(default_factory=lambda: Tensor([]), init=False, repr=False, **kwargs)


T = TypeVar("T")


@jaxtyped(typechecker=typechecked)
@config_dataclass
class ContrastiveDataLoader(BaseDataLoader, ABC):
    """Data loader for contrastive learning.

    This class samples reference, positive, and negative samples from the dataset for contrastive learning.
    """

    batch_size_neg: Optional[int] = config_field(default=None)
    drop_last: bool = config_field(default=False, skip_default=True)

    @property
    def num_negatives(self) -> int:
        """Number of negative samples to sample."""
        if self.batch_size_neg is None:
            return self.batch_size
        else:
            return self.batch_size_neg

    @check_initialized
    def __len__(self) -> int:
        """Return number of iterations per epoch, i.e. number of batches."""
        if self.drop_last:
            return len(self.dataset) // self.batch_size
        else:
            return (len(self.dataset) + self.batch_size - 1) // self.batch_size

    @check_initialized
    def __iter__(self) -> Iterator[ContrastiveGroupBatch,]:
        """Return iterator over batches."""
        self.reset()

        for batch_id in range(len(self)):
            yield self.sample_batch(batch_id)

    @abstractmethod
    def sample_batch(self, batch_id: int) -> ContrastiveGroupBatch:
        """Sample a batch of contrastive samples."""


@jaxtyped(typechecker=typechecked)
@config_dataclass
class GroupContrastiveDataLoader(ContrastiveDataLoader):
    """Data loader for time contrastive learning with discrete (and equidistant) time steps."""

    shuffle: bool = config_field(default=True)
    _indices: Tensor = internal_tensor_field()

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
    def reset(self):
        """Reset the data loader by reshuffling the indices."""
        indices = self.dataset.index.to(self.device)
        if self.shuffle:
            permutation_index = torch.randperm(len(indices))
            self._indices = indices[permutation_index]
        else:
            self._indices = indices

        # sampling negatives at run time via torch.randint slows down the dataloader by factor 2-3
        # to circumvent this,
        # we sample a) _neg_inidices via shuffle like _inidices
        # and b) _neg_start_index via torch.randint to get random starting points
        # then we can use neg_slice = slice(_neg_start_index, _neg_start_index + self.num_negatives)
        # to get the negative indices, which is a slice of _neg_indices

        self._neg_indices = indices[torch.randperm(len(indices))]
        self._neg_start_index = torch.randint(
            low=0,
            high=len(self._neg_indices) - self.num_negatives,
            size=(len(self),),
        )

    @jaxtyped(typechecker=typechecked)
    def sample_batch_indices(
        self,
        batch_id: int,
    ) -> Tuple[
        Integer[Tensor, " batch_size"],
        Integer[Tensor, " neg_batch_size"],
    ]:
        batch_slice = slice(batch_id * self.batch_size, (batch_id + 1) * self.batch_size)

        positive_indices = self.indices[batch_slice]

        neg_batch_slice = slice(
            self._neg_start_index[batch_id],
            self._neg_start_index[batch_id] + self.num_negatives,
        )
        negative_indices = self._neg_indices[neg_batch_slice]
        return positive_indices, negative_indices

    @jaxtyped(typechecker=typechecked)
    def sample_batch(self, batch_id: int) -> ContrastiveGroupBatch:
        positive_indices, negative_indices = self.sample_batch_indices(batch_id)
        batch_data = self.batch_from_indices(
            negative_indices=negative_indices,
            positive_indices=positive_indices,
        )

        return batch_data

    @jaxtyped(typechecker=typechecked)
    def batch_from_indices(
        self,
        positive_indices: Integer[Tensor, " batch_size"],
        negative_indices: Integer[Tensor, " neg_batch_size"],
    ) -> ContrastiveGroupBatch:
        """Get a batch of data from the dataset for given indices."""

        group_data = self.dataset.get_observed_data(positive_indices)
        negative_data = self.dataset.get_observed_data(negative_indices)
        contrastive_data = ContrastiveGroupBatch(
            positives=group_data,
            negatives=PairedData.from_batch(negative_data),
        )

        return contrastive_data

    @property
    def validation_data(self) -> PairedGroupData:
        """Get a batch of data from the dataset for validation."""

        return self.dataset.get_observed_data(self.dataset.index).to(self.device)

    @property
    def ground_truth_data(self) -> GroundTruthData:
        """Get the ground truth data for the dataset."""
        return self.dataset.ground_truth_data.to(self.device)

    def to(self: T, device: torch.device) -> T:
        """Move the data loader to the specified device."""
        self._indices = self._indices.to(device)
        self._neg_indices = self._neg_indices.to(device)
        return super().to(device)


@jaxtyped(typechecker=typechecked)
@config_dataclass
class ProcrustesGroupContrastiveDataLoader(GroupContrastiveDataLoader):
    """
    Data loader for group cl for procrustes dynamics model
    that require a minimum number of samples per system in each batch.
    """

    samples_per_system: int = config_field(default=100)

    def validate_config(self):
        if self.drop_last:
            # warning
            warnings.warn("drop_last will be ignored for ProcrustesGroupContrastiveDataLoader sampling")
        if not self.shuffle:
            # warning
            warnings.warn("shuffle=False will be ignored for ProcrustesGroupContrastiveDataLoader sampling")
        # for simpler implementation, batch_size has to be divisible by samples_per_system
        assert self.batch_size % self.samples_per_system == 0, (
            f"Batch size {self.batch_size} has to be divisible by samples_per_system {self.samples_per_system}"
        )

    @check_initialized
    def reset(self):
        """Reset the data loader by reshuffling the indices."""
        # For the stratified sampling, we build a full index over all batches
        # i.e. via our index, we directly retrieve the sample indx via a single batch_id
        # the index therefore has to be (num_batches, batch_size)
        self.dataset.to(self.device)
        sytem_idx = self.dataset.get_observed_data(self.dataset.index).actions_idx.to(self.device)
        assert sytem_idx is not None, "Dataset must return system indices as part of the observed data"

        # get count of each system
        unique_actions_idx, counts = torch.unique(sytem_idx, return_counts=True)
        # assert torch.all(counts >= self.samples_per_system), \
        #     f"Some systems have less than {self.samples_per_system} samples {dict(zip(unique_actions_idx.cpu().numpy(), counts.cpu().numpy()))}"
        unique_actions_idx = unique_actions_idx.to(self.device)
        # To build an index, we simply iterate over all batches
        # for each batch we sample random systems
        # and for each system we sample samples_per_system samples that belong to that system

        # for efficiency, we precompute the filtered indices for each system
        system_samples = []
        for system_id in unique_actions_idx:
            system_samples.append(self.dataset.index.to(self.device)[sytem_idx == system_id])

        index = []
        for _ in range(len(self)):
            batch_samples = []
            systems_per_batch = self.batch_size // self.samples_per_system
            rand_systems = unique_actions_idx[torch.randperm(len(unique_actions_idx), device=self.device)][:systems_per_batch]

            # if we have less rand saystems than systems_per_batch, we need to increase the samples_per_system
            samples_per_system = self.samples_per_system
            if len(rand_systems) < systems_per_batch:
                samples_per_system = self.batch_size // len(rand_systems)
            for system_id in rand_systems:
                system_samples_for_id = system_samples[system_id.item()]
                batch_system_samples = system_samples_for_id[torch.randperm(len(system_samples_for_id))[:samples_per_system]]
                batch_samples.append(batch_system_samples)
            index.append(
                torch.cat(
                    batch_samples,
                )
            )

        self._indices = torch.stack(index, dim=0).to(self.device)

        # CHECK index for correct stratified sampling
        # check each batch has every system at least samples_per_system times
        for batch_id in range(len(self)):
            batch_index = self.indices[batch_id]
            batch_actions_idx = sytem_idx[batch_index]
            for system_id in torch.unique(batch_actions_idx):
                assert (batch_actions_idx == system_id).sum() >= self.samples_per_system, (
                    f"Batch {batch_id} has only {batch_actions_idx.count(system_id)} samples for system {system_id}"
                )

        # sampling negatives at run time via torch.randint slows down the dataloader by factor 2-3
        # to circumvent this,
        # we sample a) _neg_inidices via shuffle like _inidices
        # and b) _neg_start_index via torch.randint to get random starting points
        # then we can use neg_slice = slice(_neg_start_index, _neg_start_index + self.num_negatives)
        # to get the negative indices, which is a slice of _neg_indices

        indices = self.dataset.index.to(self.device)
        self._neg_indices = indices[torch.randperm(len(indices))]
        self._neg_start_index = torch.randint(
            low=0,
            high=len(self._neg_indices) - self.num_negatives,
            size=(len(self),),
        ).to(self.device)

    @jaxtyped(typechecker=typechecked)
    def sample_batch_indices(
        self,
        batch_id: int,
    ) -> Tuple[
        Integer[Tensor, " batch_size"],
        Integer[Tensor, " neg_batch_size"],
    ]:
        positive_indices = self.indices[batch_id]

        neg_batch_slice = slice(
            self._neg_start_index[batch_id],
            self._neg_start_index[batch_id] + self.num_negatives,
        )
        negative_indices = self._neg_indices[neg_batch_slice]
        return positive_indices, negative_indices


def vectorized_randint_shaped(max_values, sample_shape):
    """
    Generates random integer tensors with a specified shape for each batch element,
    where the upper bound of the uniform distribution varies across the batch.

    Args:
        max_values: A 1-D integer tensor of shape (batch,) containing the exclusive
                    upper bounds for each batch element's random samples.
        sample_shape: A tuple or list defining the desired shape of the random
                        samples for each batch element (e.g., (3, 4) or (5,)).

    Returns:
        A tensor of shape (batch, *sample_shape) containing the randomly
        sampled integers.
    """
    batch_size = max_values.size(0)
    target_shape = (batch_size,) + tuple(sample_shape)
    num_elements = torch.prod(torch.tensor(sample_shape))

    # Expand max_values to match the total number of random samples needed
    expanded_max_values = max_values.unsqueeze(-1).expand(-1, num_elements).flatten()

    # Generate uniform random numbers
    uniform_samples = torch.rand_like(expanded_max_values, dtype=torch.float)

    # Scale and convert to integers
    random_integers_flat = (uniform_samples * expanded_max_values).floor().long()

    # Reshape to the desired output shape
    return random_integers_flat.reshape(target_shape)


@jaxtyped(typechecker=typechecked)
@config_dataclass
class GCLDataLoader(ContrastiveDataLoader):
    """Data loader for single pair group cl with procrustes model"""

    shuffle: bool = config_field(default=True)
    samples_per_action: int = config_field(default=8)
    negative_distribution: Literal[
        "x",
        "x_prime",
        "both",
    ] = config_field(default="x", skip_default=True)

    _indices: Tensor = internal_tensor_field()
    _neg_indices: Tensor = internal_tensor_field()
    _neg_start_index: Tensor = internal_tensor_field()
    _indices_per_action: Tensor = internal_tensor_field()
    actions_idx: Tensor = internal_tensor_field()
    unique_actions_idx: Tensor = internal_tensor_field()
    actions_idx_unique_keys: Tensor = internal_tensor_field()

    @jaxtyped(typechecker=typechecked)
    def __lazy_post_init__(
        self,
        dataset: SinglePairedActionsDataset,
    ):
        self._dataset = dataset
        self.prep_actions_idx()
        torch.manual_seed(self.seed)
        self.reset()
        self.to(self.device)

    def prep_actions_idx(self):
        # self.actions_idx = self.dataset.get_observed_data(
        #     self.dataset.index).actions_idx.to(self.device)
        self.actions_idx = self.dataset.get_action_idx(self.dataset.index).to(self.device)
        assert self.actions_idx is not None, "Dataset must return system indices as part of the observed data"

        # get count of each system
        self.unique_actions_idx, counts = torch.unique(self.actions_idx, return_counts=True)
        assert torch.all(counts >= self.samples_per_action), (
            f"Some systems have less than {self.samples_per_action} samples {dict(zip(self.unique_actions_idx.cpu().numpy(), counts.cpu().numpy()))}"
        )
        self.unique_actions_idx = self.unique_actions_idx.to(self.device)
        # for efficiency, we precompute the filtered indices for each system
        system_samples = []
        self.actions_idx_unique_keys = self.actions_idx.clone()
        for unique_idx, system_id in enumerate(self.unique_actions_idx):
            mask = self.actions_idx == system_id
            system_samples.append(self.dataset.index.to(self.device)[mask])
            self.actions_idx_unique_keys[mask] = unique_idx
        self.actions_idx_unique_keys = self.actions_idx_unique_keys.to(self.device)

        # we may need to do some padding, so we can stack the tensors
        max_len = max([len(sample) for sample in system_samples])
        for i in range(len(system_samples)):
            if len(system_samples[i]) < max_len:
                system_samples[i] = torch.cat(
                    [
                        system_samples[i],
                        torch.ones(
                            max_len - len(system_samples[i]),
                            device=self.device,
                            dtype=torch.long,
                        )
                        * -1,
                    ]
                )

        self._indices_per_action = torch.stack(system_samples, dim=0)

    @property
    @check_initialized
    def indices(self) -> Tensor:
        return self._indices

    @property
    @jaxtyped(typechecker=typechecked)
    def dataset(self) -> SinglePairedActionsDataset:
        return super().dataset

    def validate_config_post_init(self):
        # dataset size must be larger than num_negatives
        assert len(self.dataset.index) > self.num_negatives, f"Dataset size {len(self.dataset.index)} must be larger than num_negatives {self.num_negatives}"
        return super().validate_config_post_init()

    @check_initialized
    def reset(self):
        """Reset the data loader by reshuffling the indices."""
        indices = self.dataset.index.to(self.device)
        if self.shuffle:
            permutation_index = torch.randperm(len(indices))
            self._indices = indices[permutation_index]
        else:
            self._indices = indices

        # sampling negatives at run time via torch.randint slows down the dataloader by factor 2-3
        # to circumvent this,
        # we sample a) _neg_inidices via shuffle like _inidices
        # and b) _neg_start_index via torch.randint to get random starting points
        # then we can use neg_slice = slice(_neg_start_index, _neg_start_index + self.num_negatives)
        # to get the negative indices, which is a slice of _neg_indices

        self._neg_indices = indices[torch.randperm(len(indices))]

        self._neg_start_index = torch.randint(
            low=0,
            high=len(self._neg_indices) - self.num_negatives,
            size=(len(self),),
        )

    def support_indicies_from_positive(self, positive_indices=None, actions_idx=None):
        # for each random positive sample,
        # get sample per system more samples for the same system
        if actions_idx is None:
            actions_idx = self.actions_idx_unique_keys[positive_indices]
        support_idx = self._indices_per_action[actions_idx]

        rand_support_idx = vectorized_randint_shaped(support_idx.max(-1).indices, (self.samples_per_action,))
        support_indices = torch.gather(support_idx, dim=1, index=rand_support_idx)

        return support_indices

    @jaxtyped(typechecker=typechecked)
    def sample_batch_indices(
        self,
        batch_id: int,
    ) -> Tuple[
        Integer[Tensor, " batch_size {self.samples_per_action}+1"],
        Integer[Tensor, " neg_batch_size"],
    ]:
        batch_slice = slice(batch_id * self.batch_size, (batch_id + 1) * self.batch_size)

        positive_indices = self.indices[batch_slice]
        support_indices = self.support_indicies_from_positive(positive_indices=positive_indices)
        positive_indices = torch.cat([positive_indices.unsqueeze(-1), support_indices], dim=-1)

        neg_batch_slice = slice(
            self._neg_start_index[batch_id],
            self._neg_start_index[batch_id] + self.num_negatives,
        )
        negative_indices = self._neg_indices[neg_batch_slice]
        return positive_indices, negative_indices

    @jaxtyped(typechecker=typechecked)
    def sample_batch(self, batch_id: int) -> ContrastiveGroupBatch:
        batch_data = self.batch_from_indices(
            *self.sample_batch_indices(batch_id),
        )

        return batch_data

    @jaxtyped(typechecker=typechecked)
    def batch_from_indices(
        self,
        positive_indices: Integer[Tensor, " batch_size {self.samples_per_action}+1"],
        negative_indices: Integer[Tensor, " neg_batch_size"],
    ) -> ContrastiveGroupBatch:
        """Get a batch of data from the dataset for given indices."""

        positive_data = self.dataset.get_observed_data(positive_indices)

        negative_data = self.dataset.get_observed_data(negative_indices)

        if self.negative_distribution == "x":
            negatives = dt.SingleData(
                x=negative_data.x,
                indices=negative_data.indices,
                class_idx=negative_data.class_idx,
            )
        elif self.negative_distribution == "x_prime":
            negatives = dt.SingleData(
                x=negative_data.x_prime,
                indices=negative_data.indices,
                class_idx=negative_data.class_idx,
            )
        elif self.negative_distribution == "both":
            # we don't want to double the number of samples as to simplify the analsis
            # and comparibility with other values for negative distribution
            # therefore we take half from x and half from x_prime

            half_batch_size = negative_data.x.shape[0] // 2
            combined_data = torch.cat(
                [
                    negative_data.x[:half_batch_size],
                    negative_data.x_prime[:half_batch_size],
                ],
                dim=0,
            )
            combined_indices = torch.cat(
                [
                    negative_data.indices[:half_batch_size],
                    negative_data.indices[:half_batch_size],
                ],
                dim=0,
            )
            combined_class_idx = torch.cat(
                [
                    negative_data.class_idx[:half_batch_size],
                    negative_data.class_idx[:half_batch_size],
                ],
                dim=0,
            )
            negatives = dt.SingleData(
                x=combined_data,
                indices=combined_indices,
                class_idx=combined_class_idx,
            )
        else:
            raise ValueError(f"Invalid negative distribution: {self.negative_distribution}")

        contrastive_data = ContrastiveGroupBatch(
            positives=positive_data,
            negatives=negatives,
        )

        return contrastive_data

    @check_initialized
    def iter_latents(self) -> Iterator[ContrastiveGroupBatch,]:
        """Return iterator over batches."""
        self.reset()

        for batch_id in range(len(self)):
            yield self.latent_batch_from_indices(
                *self.sample_batch_indices(batch_id),
            )

    def latent_batch_from_indices(
        self,
        positive_indices: Integer[Tensor, " batch_size {self.samples_per_action}+1"],
        negative_indices: Integer[Tensor, " neg_batch_size"],
    ) -> ContrastiveGroupBatch:
        """Get a batch of data from the dataset for given indices."""

        positive_data = self.dataset.get_latent_data(positive_indices)

        negative_data = self.dataset.get_latent_data(negative_indices)
        contrastive_data = ContrastiveGroupBatch(
            positives=positive_data,
            negatives=dt.SingleData.from_batch(negative_data),
        )

        return contrastive_data

    @property
    def validation_data(self) -> dt.SinglePairedGroupData:
        """Get a batch of data from the dataset for validation."""

        return self.dataset.get_observed_data(self.dataset.index).to(self.device)

    @property
    def ground_truth_data(self) -> GroundTruthData:
        """Get the ground truth data for the dataset."""
        return self.dataset.ground_truth_data.to(self.device)

    def to(self: "GCLDataLoader", device: torch.device) -> "GCLDataLoader":
        """Move the data loader to the specified device."""
        if self.initialized:
            self._indices = self._indices.to(device)
            self._neg_indices = self._neg_indices.to(device)
            self._neg_start_index = self._neg_start_index.to(device)
            self._indices_per_action = self._indices_per_action.to(device)
            self.actions_idx = self.actions_idx.to(device)
            self.actions_idx_unique_keys = self.actions_idx_unique_keys.to(device)
        return super().to(device)


@jaxtyped(typechecker=typechecked)
@config_dataclass
class InfiniteGCLDataLoader(GCLDataLoader):
    """Data loader for single pair group cl with procrustes model"""

    num_iterations: int = config_field(default=1000)

    def __len__(self) -> int:
        """Return number of iterations per epoch, i.e. number of batches."""
        return self.num_iterations

    def __post_init__(self):
        self.shuffle = False
        super().__post_init__()

    @jaxtyped(typechecker=typechecked)
    def __lazy_post_init__(
        self,
        dataset: SyntheticSinglePairedRotationsDataset,
    ):
        super().__lazy_post_init__(dataset)

    @property
    @jaxtyped(typechecker=typechecked)
    def dataset(self) -> SyntheticSinglePairedRotationsDataset:
        return super().dataset

    def prep_actions_idx(self):
        self.actions_idx = self.dataset.get_action_idx(self.dataset.index).to(self.device)
        assert self.actions_idx is not None, "Dataset must return system indices as part of the observed data"

    @check_initialized
    def reset(self):
        """Reset the data loader by reshuffling the indices."""
        pass

    def sample_batch_indices(self, **kwargs):
        raise NotImplementedError("InfiniteGCLDataLoader doesn't implement sampling indices")

    def batch_from_indices(self, **kwargs):
        raise NotImplementedError("InfiniteGCLDataLoader doesn't implement batch_from_indices")

    @jaxtyped(typechecker=typechecked)
    def sample_batch(self, batch_id: int) -> ContrastiveGroupBatch:
        self.dataset.to(self.device)
        # use the .dataset to generate the data on the fly
        positives_x = self.dataset.starting_points_sampler.sample(
            num_samples=self.batch_size * self.samples_per_action,
            dim=self.dataset.dynamics_model.dim,
        ).to(self.device)
        negatives_x = self.dataset.starting_points_sampler.sample(
            num_samples=self.num_negatives,
            dim=self.dataset.dynamics_model.dim,
        ).to(self.device)

        action_idx = self.actions_idx[
            torch.randint(
                0,
                len(self.actions_idx),
                (self.batch_size,),
            )
        ]
        action_idx = action_idx.unsqueeze(-1).repeat(1, self.samples_per_action)

        positives_x_prime = self.dataset.dynamics_model(
            x=positives_x,
            system_idx=action_idx.view(-1),
        )

        negatives_x_prime = None
        if self.negative_distribution == "x_prime" or self.negative_distribution == "both":
            action_idx_neg = torch.randint(
                0,
                self.dataset.num_group_elements,
                (self.num_negatives,),
            )
            action_idx_neg = action_idx_neg.unsqueeze(-1)
            negatives_x_prime = self.dataset.dynamics_model(
                x=negatives_x,
                system_idx=action_idx_neg.view(-1),
            )

        class_idx = None
        neg_class_idx = None
        if hasattr(self.dataset, "content_embedding"):
            # sample and add content embeddings
            class_idx = self.dataset.content_embedding.sample_class_idx(num_samples=positives_x.shape[0])
            latent_content = self.dataset.content_embedding.embed(class_idx)
            positives_x = torch.cat([positives_x, latent_content], dim=-1)
            positives_x_prime = torch.cat([positives_x_prime, latent_content], dim=-1)
            neg_class_idx = self.dataset.content_embedding.sample_class_idx(num_samples=negatives_x.shape[0])
            neg_latent_content = self.dataset.content_embedding.embed(neg_class_idx)

            negatives_x = torch.cat([negatives_x, neg_latent_content], dim=-1)
            if negatives_x_prime is not None:
                negatives_x_prime = torch.cat([negatives_x_prime, neg_latent_content], dim=-1)

            class_idx = class_idx.reshape(
                self.batch_size,
                self.samples_per_action,
            )

        # reshape x and x_prime to (batch_size, samples_per_action, latent_dim)
        positives_x = positives_x.reshape(self.batch_size, self.samples_per_action, self.dataset.latent_dim)
        positives_x_prime = positives_x_prime.reshape(self.batch_size, self.samples_per_action, self.dataset.latent_dim)

        if self.negative_distribution == "x":
            negatives = negatives_x
        elif self.negative_distribution == "x_prime":
            negatives = negatives_x_prime
        elif self.negative_distribution == "both":
            half_batch_size = negatives_x.shape[0] // 2
            negatives = torch.cat(
                [negatives_x[:half_batch_size], negatives_x_prime[:half_batch_size]],
                dim=0,
            )

        # mixing
        self.dataset.mixing_model.to(self.device)
        positives_x = self.dataset.mixing_model(positives_x)
        positives_x_prime = self.dataset.mixing_model(positives_x_prime)
        negatives = self.dataset.mixing_model(negatives)

        return dt.ContrastiveGroupBatch(
            positives=dt.SinglePairedGroupData(
                x=positives_x,
                x_prime=positives_x_prime,
                actions_idx=action_idx,
                class_idx=class_idx,
            ),
            negatives=dt.SingleData(
                x=negatives,
                class_idx=neg_class_idx,
            ),
        )

    @property
    def validation_data(self) -> dt.SinglePairedGroupData:
        """Get a batch of data from the dataset for validation."""
        raise NotImplementedError("InfiniteGCLDataLoader doesn't implement validation_data, only use for training")

    @property
    def ground_truth_data(self) -> GroundTruthData:
        raise NotImplementedError("InfiniteGCLDataLoader doesn't implement ground_truth_data, only use for training")


@jaxtyped(typechecker=typechecked)
@config_dataclass
class DspritesInfiniteGCLDataLoader(InfiniteGCLDataLoader):
    """Data loader for single pair group cl with procrustes model"""

    num_iterations: int = config_field(default=1000)

    @jaxtyped(typechecker=typechecked)
    def __lazy_post_init__(
        self,
        dataset: DSpritesDataset,
    ):
        GCLDataLoader.__lazy_post_init__(self, dataset=dataset)

    @property
    @jaxtyped(typechecker=typechecked)
    def dataset(self) -> DSpritesDataset:
        return self._dataset

    @jaxtyped(typechecker=typechecked)
    def sample_batch(self, batch_id: int) -> ContrastiveGroupBatch:
        self.dataset.to(self.device)
        # use the .dataset to generate the data on the fly
        positives_x = self.dataset._generate_starting_points(
            num_samples=self.batch_size * self.samples_per_action,
        ).to(self.device)
        negatives_x = self.dataset._generate_starting_points(
            num_samples=self.num_negatives,
        ).to(self.device)

        latent_update, action_idx = self.dataset.action_sampler.sample_action(num_samples=self.batch_size)

        action_idx = action_idx.unsqueeze(-1).repeat(1, self.samples_per_action)
        latent_update = latent_update.unsqueeze(1).repeat(1, self.samples_per_action, 1)

        # reshape x and x_prime to (batch_size, samples_per_action, latent_dim)
        positives_x = positives_x.reshape(self.batch_size, self.samples_per_action, self.dataset.latent_dim)
        positives_x_prime = positives_x + latent_update
        positives_x_prime = torch.remainder(positives_x_prime, self.dataset.latent_sizes)

        negatives_x_prime = None
        if self.negative_distribution == "x_prime" or self.negative_distribution == "both":
            latent_update_neg, action_idx_neg = self.dataset.action_sampler.sample_action(
                num_samples=self.num_negatives,
            )
            negatives_x_prime = negatives_x + latent_update_neg
            negatives_x_prime = torch.remainder(negatives_x_prime, self.dataset.latent_sizes)

        class_idx = self.dataset.class_idx_from_latent(positives_x)
        x_prime_class_idx = self.dataset.class_idx_from_latent(positives_x_prime)
        assert torch.all(class_idx == x_prime_class_idx)
        neg_class_idx = self.dataset.class_idx_from_latent(negatives_x)

        if self.negative_distribution == "x":
            negatives = negatives_x
        elif self.negative_distribution == "x_prime":
            negatives = negatives_x_prime
        elif self.negative_distribution == "both":
            half_batch_size = negatives_x.shape[0] // 2
            negatives = torch.cat(
                [negatives_x[:half_batch_size], negatives_x_prime[:half_batch_size]],
                dim=0,
            )

        # mixing
        positives_x = self.dataset.latent_to_img(positives_x)
        positives_x_prime = self.dataset.latent_to_img(positives_x_prime)
        negatives = self.dataset.latent_to_img(negatives)

        return dt.ContrastiveGroupBatch(
            positives=dt.SinglePairedGroupData(
                x=positives_x,
                x_prime=positives_x_prime,
                actions_idx=action_idx,
                class_idx=class_idx,
            ),
            negatives=dt.SingleData(
                x=negatives,
                class_idx=neg_class_idx,
            ),
        )


@jaxtyped(typechecker=typechecked)
@config_dataclass
class BehaviorEquivariantDataLoader(GCLDataLoader):
    """
    Data loader for datasets with scalar continuous behavior variable.

    Uses behavioral variable to generate positive and negative pairs,
    by first creating pairs of data by time difference and then grouping pairs via difference in behavior variable.

    """

    time_offsets: List[int] = config_field(default_factory=lambda: [1])
    num_bins: int = config_field(default=10)
    symlog_scale_alpha: Optional[float] = config_field(default=None, skip_default=True)
    min_samples_per_action: Optional[int] = config_field(default=None, skip_default=True)
    min_abs_difference: Optional[float] = config_field(default=None, skip_default=True)
    max_abs_difference: Optional[float] = config_field(default=None, skip_default=True)

    # Internal state (not part of metadata/config)
    _behavior_diff: Tensor = internal_tensor_field()
    _x_index: Tensor = internal_tensor_field()
    _x_prime_index: Tensor = internal_tensor_field()
    _actions_counts: Tensor = internal_tensor_field()
    _actions_bin_edges: Tensor = internal_tensor_field()

    def __len__(self) -> int:
        """Return number of iterations per epoch, i.e. number of batches."""
        if self.drop_last:
            return len(self.clean_indices) // self.batch_size
        else:
            return (len(self.clean_indices) + self.batch_size - 1) // self.batch_size

    @jaxtyped(typechecker=typechecked)
    def __lazy_post_init__(
        self,
        dataset: HippcampusRatDataset,
    ):
        self._dataset = dataset
        self.prep_actions_idx()
        torch.manual_seed(self.seed)
        self.reset()
        self.to(self.device)

    @property
    @jaxtyped(typechecker=typechecked)
    def dataset(self) -> HippcampusRatDataset:
        return self._dataset

    def compute_action_pairs(self):
        (
            self._behavior_diff,
            self._x_index,
            self._x_prime_index,
            self._time_offsets,
        ) = paired_differences(
            auxiliary_data=self.dataset.behavior_variable,
            offsets=self.time_offsets,
        )
        self.actions_idx, self.actions_counts, self.actions_bin_edges = bin_differences(
            n_bins=self.num_bins,
            aux_data_diff=self._behavior_diff,
            symlog_scale_alpha=self.symlog_scale_alpha,
        )

        # apply filters to the paired difference data and actions_idx
        (
            self._behavior_diff,
            self._x_index,
            self._x_prime_index,
            self._time_offsets,
            self.actions_idx,
            self.actions_counts,
            self.actions_bin_edges,
        ) = filter_binned_differences(
            auxiliary_diff=self._behavior_diff,
            x_index=self._x_index,
            x_prime_index=self._x_prime_index,
            time_offsets=self._time_offsets,
            actions_idx=self.actions_idx,
            actions_counts=self.actions_counts,
            actions_bin_edges=self.actions_bin_edges,
            min_samples_per_action=self.min_samples_per_action,
            min_abs_difference=self.min_abs_difference,
            max_abs_difference=self.max_abs_difference,
        )

        self.unique_actions_idx = torch.unique(self.actions_idx)

        print(
            f"actions_counts for time_offset {self.time_offsets} and num_bins {self.num_bins}:\n {self.actions_counts} \n min: {self.actions_counts.min()} \n max: {self.actions_counts.max()}",
            flush=True,
        )

    def validate_config_post_init(self):
        # dataset size must be larger than num_negatives
        assert len(self.clean_indices) > self.num_negatives, f"Dataset size {len(self.clean_indices)} must be larger than num_negatives {self.num_negatives}"
        return ContrastiveDataLoader.validate_config_post_init(self)

    @property
    def num_effective_bins(self) -> int:
        return len(self.unique_actions_idx)

    @property
    def clean_indices(self) -> Integer[Tensor, " num_samples"]:
        return torch.arange(len(self._x_index)).to(self.device)

    def reset(self):
        """Reset the data loader by reshuffling the indices."""
        indices = self.clean_indices
        if self.shuffle:
            permutation_index = torch.randperm(len(indices))
            self._indices = indices[permutation_index]
        else:
            self._indices = indices

        # sampling negatives at run time via torch.randint slows down the dataloader by factor 2-3
        # to circumvent this,
        # we sample a) _neg_inidices via shuffle like _inidices
        # and b) _neg_start_index via torch.randint to get random starting points
        # then we can use neg_slice = slice(_neg_start_index, _neg_start_index + self.num_negatives)
        # to get the negative indices, which is a slice of _neg_indices

        self._neg_indices = indices[torch.randperm(len(indices))]

        self._neg_start_index = torch.randint(
            low=0,
            high=len(self._neg_indices) - self.num_negatives,
            size=(len(self),),
        )

    def prep_actions_idx(self):
        self.compute_action_pairs()
        self.actions_idx = self.actions_idx.to(self.device)
        # get count of each system
        counts = self.actions_counts
        assert torch.all(counts >= self.samples_per_action), (
            f"Some systems have less than {self.samples_per_action} samples {dict(zip(self.unique_actions_idx.cpu().numpy(), counts.cpu().numpy()))}"
        )
        self.unique_actions_idx = self.unique_actions_idx.to(self.device)
        # for efficiency, we precompute the filtered indices for each system
        system_samples = []
        self.actions_idx_unique_keys = self.actions_idx.clone()
        for unique_idx, system_id in enumerate(self.unique_actions_idx):
            mask = self.actions_idx == system_id
            system_samples.append(self.clean_indices[mask])
            self.actions_idx_unique_keys[mask] = unique_idx
        self.actions_idx_unique_keys = self.actions_idx_unique_keys.to(self.device)

        # we may need to do some padding, so we can stack the tensors
        max_len = max([len(sample) for sample in system_samples])
        for i in range(len(system_samples)):
            if len(system_samples[i]) < max_len:
                system_samples[i] = torch.cat(
                    [
                        system_samples[i],
                        torch.ones(
                            max_len - len(system_samples[i]),
                            device=self.device,
                            dtype=torch.long,
                        )
                        * -1,
                    ]
                )

        self._indices_per_action = torch.stack(system_samples, dim=0)

    @jaxtyped(typechecker=typechecked)
    def sample_batch_indices(
        self,
        batch_id: int,
    ) -> Tuple[
        Integer[Tensor, " batch_size {self.samples_per_action}+1"],
        Integer[Tensor, " neg_batch_size"],
    ]:
        batch_slice = slice(batch_id * self.batch_size, (batch_id + 1) * self.batch_size)

        positive_indices = self.indices[batch_slice]
        support_indices = self.support_indicies_from_positive(positive_indices=positive_indices)
        positive_indices = torch.cat([positive_indices.unsqueeze(-1), support_indices], dim=-1)

        neg_batch_slice = slice(
            self._neg_start_index[batch_id],
            self._neg_start_index[batch_id] + self.num_negatives,
        )
        negative_indices = self._neg_indices[neg_batch_slice]
        return positive_indices, negative_indices

    @jaxtyped(typechecker=typechecked)
    def sample_batch(self, batch_id: int) -> ContrastiveGroupBatch:
        batch_data = self.batch_from_indices(
            *self.sample_batch_indices(batch_id),
        )

        return batch_data

    @jaxtyped(typechecker=typechecked)
    def get_observed_data(self, indices: Integer[Tensor, " *batch_shape"]) -> dt.SinglePairedGroupData:
        """Get a batch of data from the dataset."""

        # use the self._x_index and self._x_prime_index to get the correct data from the dataset
        x = self.dataset.get_observed_variable(self._x_index[indices])
        x_prime = self.dataset.get_observed_variable(self._x_prime_index[indices])

        return dt.SinglePairedGroupData(
            x=x,
            x_prime=x_prime,
            indices=indices,
            actions_idx=self.actions_idx[indices],
        )

    @jaxtyped(typechecker=typechecked)
    def batch_from_indices(
        self,
        positive_indices: Integer[Tensor, " batch_size {self.samples_per_action}+1"],
        negative_indices: Integer[Tensor, " neg_batch_size"],
    ) -> ContrastiveGroupBatch:
        """Get a batch of data from the dataset for given indices."""

        positive_data = self.get_observed_data(positive_indices)

        negative_data = self.get_observed_data(negative_indices)

        if self.negative_distribution == "x":
            negatives = dt.SingleData(
                x=negative_data.x,
                indices=negative_data.indices,
                class_idx=negative_data.class_idx,
            )
        elif self.negative_distribution == "x_prime":
            negatives = dt.SingleData(
                x=negative_data.x_prime,
                indices=negative_data.indices,
                class_idx=negative_data.class_idx,
            )
        elif self.negative_distribution == "both":
            # we don't want to double the number of samples as to simplify the analsis
            # and comparibility with other values for negative distribution
            # therefore we take half from x and half from x_prime

            half_batch_size = negative_data.x.shape[0] // 2
            combined_data = torch.cat(
                [
                    negative_data.x[:half_batch_size],
                    negative_data.x_prime[:half_batch_size],
                ],
                dim=0,
            )
            combined_indices = torch.cat(
                [
                    negative_data.indices[:half_batch_size],
                    negative_data.indices[:half_batch_size],
                ],
                dim=0,
            )

            if hasattr(negative_data, "class_idx") and negative_data.class_idx is not None:
                combined_class_idx = torch.cat(
                    [
                        negative_data.class_idx[:half_batch_size],
                        negative_data.class_idx[:half_batch_size],
                    ],
                    dim=0,
                )
            else:
                combined_class_idx = None

            negatives = dt.SingleData(
                x=combined_data,
                indices=combined_indices,
                class_idx=combined_class_idx,
            )
        else:
            raise ValueError(f"Invalid negative distribution: {self.negative_distribution}")

        contrastive_data = ContrastiveGroupBatch(
            positives=positive_data,
            negatives=negatives,
        )

        return contrastive_data

    @property
    def validation_data(self) -> dt.SinglePairedGroupData:
        """Get a batch of data from the dataset for validation."""
        return self.get_observed_data(self.clean_indices)

    @property
    def ground_truth_data(self) -> GroundTruthData:
        return GroundTruthData(
            index=self.clean_indices,
            observed=self.validation_data,
            actions_idx=self.actions_idx,
            class_idx=None,
        )

    def to(self: "BehaviorEquivariantDataLoader", device: torch.device) -> "BehaviorEquivariantDataLoader":
        if self.initialized:
            self._x_index = self._x_index.to(device)
            self._x_prime_index = self._x_prime_index.to(device)
            self._actions_counts = self._actions_counts.to(device)
            self._actions_bin_edges = self._actions_bin_edges.to(device)
        return super().to(device)


@jaxtyped(typechecker=typechecked)
@config_dataclass
class BehaviorEquivariantDataLoaderV2(GCLDataLoader):
    """
    Data loader for datasets with scalar continuous behavior variable.

    Uses behavioral variable to generate positive and negative pairs,
    by first creating pairs of data by time difference and then grouping pairs via difference in behavior variable.

    """

    max_num_pairs: Optional[int] = config_field(default=None)
    bin_sizes: List[Optional[float]] = config_field(default_factory=lambda: None)
    equivariant_dims: Optional[List[int]] = config_field(default_factory=lambda: None)
    # Internal state (not part of metadata/config)
    _x_index: Tensor = internal_tensor_field()
    _x_prime_index: Tensor = internal_tensor_field()
    _actions_counts: Tensor = internal_tensor_field()
    _actions_bin_edges: Tensor = internal_tensor_field()

    def __len__(self) -> int:
        """Return number of iterations per epoch, i.e. number of batches."""
        if self.drop_last:
            return len(self.clean_indices) // self.batch_size
        else:
            return (len(self.clean_indices) + self.batch_size - 1) // self.batch_size

    @jaxtyped(typechecker=typechecked)
    def __lazy_post_init__(
        self,
        dataset: HippcampusRatDataset,
    ):
        self._dataset = dataset.to(self.device)
        self.prep_actions_idx()
        torch.manual_seed(self.seed)
        self.reset()
        self.to(self.device)

    @property
    @jaxtyped(typechecker=typechecked)
    def dataset(self) -> HippcampusRatDataset:
        return self._dataset

    def compute_action_pairs(self):
        behavior_variables = self.dataset.behavior_variable.to(self.device)
        index = self.dataset.index.to(self.device)

        auxilary_variable, boundaries = discretize_variable(
            behavior_variables,
            bin_size=self.bin_sizes,
            return_boundaries=True,
        )
        full_product = torch.cartesian_prod(index, index)
        # randomly downsample to max_num_pairs
        if self.max_num_pairs is not None and full_product.shape[0] > self.max_num_pairs:
            print(
                f"Downsampling from {full_product.shape[0]:,} to {self.max_num_pairs:,}",
                flush=True,
            )
            generator = torch.Generator()
            generator.manual_seed(self.seed)
            index_pairs = full_product[torch.randperm(full_product.shape[0], generator=generator)[: self.max_num_pairs]]
        else:
            print(
                f"No downsampling. Using all {full_product.shape[0]:,} pairs",
                flush=True,
            )
            index_pairs = full_product

        group_factors, index_pairs = generate_group_factors(
            auxilary_variable,
            index_pairs,
            equivariant_dims=self.equivariant_dims,
        )
        (
            unique_group_factors,
            group_factor_ids,
            group_factor_counts,
        ) = torch.unique(group_factors, return_counts=True, return_inverse=True, dim=0)
        print("Before filtering:", flush=True)
        print(f"\tNumber of unique group factors: {len(unique_group_factors)}", flush=True)
        print(f"\tNumber of pairs: {len(group_factor_ids)}", flush=True)

        min_action_count = self.samples_per_action

        min_filter = group_factor_counts >= min_action_count
        filtered_unique_group_factors = unique_group_factors[min_filter]
        filtered_group_factor_counts = group_factor_counts[min_filter]

        valid_group_ids = torch.where(min_filter)[0]
        # Create mask for samples that belong to valid actions
        sample_mask = torch.isin(group_factor_ids, valid_group_ids)
        filtered_group_factor_ids = group_factor_ids[sample_mask]
        filtered_index_pairs = index_pairs[sample_mask]

        print("After filtering:", flush=True)
        print(
            f"\tNumber of unique group factors: {len(filtered_unique_group_factors)}",
            flush=True,
        )
        print(f"\tNumber of pairs: {len(filtered_index_pairs)}", flush=True)

        self.actions_idx = filtered_group_factor_ids
        self._unique_group_factors = filtered_unique_group_factors
        self._actions_counts = filtered_group_factor_counts
        self._x_index = filtered_index_pairs[:, 0]
        self._x_prime_index = filtered_index_pairs[:, 1]
        self._actions_bin_edges = boundaries

    def validate_config_post_init(self):
        # dataset size must be larger than num_negatives
        assert len(self.clean_indices) > self.num_negatives, f"Dataset size {len(self.clean_indices)} must be larger than num_negatives {self.num_negatives}"
        return ContrastiveDataLoader.validate_config_post_init(self)

    @property
    def num_effective_bins(self) -> int:
        return len(self.unique_actions_idx)

    @property
    def clean_indices(self) -> Integer[Tensor, " num_samples"]:
        return torch.arange(len(self._x_index)).to(self.device)

    def reset(self):
        """Reset the data loader by reshuffling the indices."""
        indices = self.clean_indices
        if self.shuffle:
            permutation_index = torch.randperm(len(indices))
            self._indices = indices[permutation_index]
        else:
            self._indices = indices

        # sampling negatives at run time via torch.randint slows down the dataloader by factor 2-3
        # to circumvent this,
        # we sample a) _neg_inidices via shuffle like _inidices
        # and b) _neg_start_index via torch.randint to get random starting points
        # then we can use neg_slice = slice(_neg_start_index, _neg_start_index + self.num_negatives)
        # to get the negative indices, which is a slice of _neg_indices

        self._neg_indices = indices[torch.randperm(len(indices))]

        self._neg_start_index = torch.randint(
            low=0,
            high=len(self._neg_indices) - self.num_negatives,
            size=(len(self),),
        )

    def prep_actions_idx(self):
        self.compute_action_pairs()
        self.actions_idx = self.actions_idx.to(self.device)
        self.actions_idx = self.actions_idx.to(self.device)
        counts = self._actions_counts
        assert torch.all(counts >= self.samples_per_action), f"Some systems have less than {self.samples_per_action} samples"

        # Step 1 & 2: Remap indices and get group counts in one atomic operation.
        unique_actions, self.actions_idx_unique_keys, counts = torch.unique(self.actions_idx, return_inverse=True, return_counts=True)
        self.unique_actions_idx = unique_actions.to(self.device)
        # Step 3: Group all clean_indices by their new unique key.
        sorter = torch.argsort(self.actions_idx_unique_keys)
        sorted_indices = self.clean_indices.to(self.device)[sorter]
        grouped_indices_list = torch.split(sorted_indices, counts.tolist())

        # Step 3: Pad all groups to the same length and stack them.
        # `pad_sequence` is a utility designed for exactly this task.

        self._indices_per_action = rnn_utils.pad_sequence(grouped_indices_list, batch_first=True, padding_value=-1)

    @jaxtyped(typechecker=typechecked)
    def sample_batch_indices(
        self,
        batch_id: int,
    ) -> Tuple[
        Integer[Tensor, " batch_size {self.samples_per_action}+1"],
        Integer[Tensor, " neg_batch_size"],
    ]:
        batch_slice = slice(batch_id * self.batch_size, (batch_id + 1) * self.batch_size)

        positive_indices = self.indices[batch_slice]
        support_indices = self.support_indicies_from_positive(positive_indices=positive_indices)
        positive_indices = torch.cat([positive_indices.unsqueeze(-1), support_indices], dim=-1)

        neg_batch_slice = slice(
            self._neg_start_index[batch_id],
            self._neg_start_index[batch_id] + self.num_negatives,
        )
        negative_indices = self._neg_indices[neg_batch_slice]
        return positive_indices, negative_indices

    @jaxtyped(typechecker=typechecked)
    def sample_batch(self, batch_id: int) -> ContrastiveGroupBatch:
        batch_data = self.batch_from_indices(
            *self.sample_batch_indices(batch_id),
        )

        return batch_data

    @jaxtyped(typechecker=typechecked)
    def get_observed_data(self, indices: Integer[Tensor, " *batch_shape"]) -> dt.SinglePairedGroupData:
        """Get a batch of data from the dataset."""

        # use the self._x_index and self._x_prime_index to get the correct data from the dataset
        x = self.dataset.get_observed_variable(self._x_index[indices])
        x_prime = self.dataset.get_observed_variable(self._x_prime_index[indices])

        return dt.SinglePairedGroupData(
            x=x,
            x_prime=x_prime,
            indices=indices,
            actions_idx=self.actions_idx[indices],
        )

    @jaxtyped(typechecker=typechecked)
    def batch_from_indices(
        self,
        positive_indices: Integer[Tensor, " batch_size {self.samples_per_action}+1"],
        negative_indices: Integer[Tensor, " neg_batch_size"],
    ) -> ContrastiveGroupBatch:
        """Get a batch of data from the dataset for given indices."""

        positive_data = self.get_observed_data(positive_indices)

        negative_data = self.get_observed_data(negative_indices)

        if self.negative_distribution == "x":
            negatives = dt.SingleData(
                x=negative_data.x,
                indices=negative_data.indices,
                class_idx=negative_data.class_idx,
            )
        elif self.negative_distribution == "x_prime":
            negatives = dt.SingleData(
                x=negative_data.x_prime,
                indices=negative_data.indices,
                class_idx=negative_data.class_idx,
            )
        elif self.negative_distribution == "both":
            # we don't want to double the number of samples as to simplify the analsis
            # and comparibility with other values for negative distribution
            # therefore we take half from x and half from x_prime

            half_batch_size = negative_data.x.shape[0] // 2
            combined_data = torch.cat(
                [
                    negative_data.x[:half_batch_size],
                    negative_data.x_prime[:half_batch_size],
                ],
                dim=0,
            )
            combined_indices = torch.cat(
                [
                    negative_data.indices[:half_batch_size],
                    negative_data.indices[:half_batch_size],
                ],
                dim=0,
            )

            if hasattr(negative_data, "class_idx") and negative_data.class_idx is not None:
                combined_class_idx = torch.cat(
                    [
                        negative_data.class_idx[:half_batch_size],
                        negative_data.class_idx[:half_batch_size],
                    ],
                    dim=0,
                )
            else:
                combined_class_idx = None

            negatives = dt.SingleData(
                x=combined_data,
                indices=combined_indices,
                class_idx=combined_class_idx,
            )
        else:
            raise ValueError(f"Invalid negative distribution: {self.negative_distribution}")

        contrastive_data = ContrastiveGroupBatch(
            positives=positive_data,
            negatives=negatives,
        )

        return contrastive_data

    @property
    def validation_data(self) -> dt.SinglePairedGroupData:
        """Get a batch of data from the dataset for validation."""
        return self.get_observed_data(self.clean_indices)

    @property
    def ground_truth_data(self) -> GroundTruthData:
        return GroundTruthData(
            index=self.clean_indices,
            observed=self.validation_data,
            actions_idx=self.actions_idx,
            class_idx=None,
        )

    def to(self: "BehaviorEquivariantDataLoader", device: torch.device) -> "BehaviorEquivariantDataLoader":
        if self.initialized:
            self._x_index = self._x_index.to(device)
            self._x_prime_index = self._x_prime_index.to(device)
            self._actions_counts = self._actions_counts.to(device)
            self._actions_bin_edges = [edges.to(device) for edges in self._actions_bin_edges]
        return super().to(device)


@jaxtyped(typechecker=typechecked)
@config_dataclass
class BehaviorEquivariantDataLoaderV3(BehaviorEquivariantDataLoaderV2):
    """
    Data loader for datasets with scalar continuous behavior variable.

    Uses behavioral variable to generate positive and negative pairs,
    by first creating pairs of data by time difference and then grouping pairs via difference in behavior variable.

    """

    max_num_pairs: Optional[int] = config_field(default=None)
    bin_sizes: List[Optional[float]] = config_field(default_factory=lambda: None)
    group_action_dims: Optional[List[int]] = config_field(default_factory=lambda: None)
    content_dims: Optional[List[int]] = config_field(default_factory=lambda: None)
    # Internal state (not part of metadata/config)
    _x_index: Tensor = internal_tensor_field()
    _x_prime_index: Tensor = internal_tensor_field()
    _actions_counts: Tensor = internal_tensor_field()
    _actions_bin_edges: Tensor = internal_tensor_field()

    def compute_action_pairs(self):
        behavior_variables = self.dataset.behavior_variable.to(self.device)
        index = self.dataset.index.to(self.device)

        auxilary_variable, boundaries = discretize_variable(
            behavior_variables,
            bin_size=self.bin_sizes,
            return_boundaries=True,
        )
        full_product = torch.cartesian_prod(index, index)
        # randomly downsample to max_num_pairs
        if self.max_num_pairs is not None and full_product.shape[0] > self.max_num_pairs:
            print(
                f"Downsampling from {full_product.shape[0]:,} to {self.max_num_pairs:,}",
                flush=True,
            )
            generator = torch.Generator()
            generator.manual_seed(self.seed)
            index_pairs = full_product[torch.randperm(full_product.shape[0], generator=generator)[: self.max_num_pairs]]
        else:
            print(
                f"No downsampling. Using all {full_product.shape[0]:,} pairs",
                flush=True,
            )
            index_pairs = full_product

        group_factors, index_pairs = generate_group_factors_v2(
            auxilary_variable,
            index_pairs,
            group_action_dims=self.group_action_dims,
            content_dims=self.content_dims,
        )
        (
            unique_group_factors,
            group_factor_ids,
            group_factor_counts,
        ) = torch.unique(group_factors, return_counts=True, return_inverse=True, dim=0)
        print("Before filtering:", flush=True)
        print(f"\tNumber of unique group factors: {len(unique_group_factors)}", flush=True)
        print(f"\tNumber of pairs: {len(group_factor_ids)}", flush=True)

        min_action_count = self.samples_per_action

        min_filter = group_factor_counts >= min_action_count
        filtered_unique_group_factors = unique_group_factors[min_filter]
        filtered_group_factor_counts = group_factor_counts[min_filter]

        valid_group_ids = torch.where(min_filter)[0]
        # Create mask for samples that belong to valid actions
        sample_mask = torch.isin(group_factor_ids, valid_group_ids)
        filtered_group_factor_ids = group_factor_ids[sample_mask]
        filtered_index_pairs = index_pairs[sample_mask]

        print("After filtering:", flush=True)
        print(
            f"\tNumber of unique group factors: {len(filtered_unique_group_factors)}",
            flush=True,
        )
        print(f"\tNumber of pairs: {len(filtered_index_pairs)}", flush=True)

        self.actions_idx = filtered_group_factor_ids
        self._unique_group_factors = filtered_unique_group_factors
        self._actions_counts = filtered_group_factor_counts
        self._x_index = filtered_index_pairs[:, 0]
        self._x_prime_index = filtered_index_pairs[:, 1]
        self._actions_bin_edges = boundaries
