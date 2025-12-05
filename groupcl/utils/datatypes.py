import warnings
from dataclasses import dataclass, field, fields
from typing import Any, Dict, List, Optional, Tuple, TypeVar, Union

import torch
from config_dataclass import config_dataclass
from jaxtyping import Float, Integer, jaxtyped
from jaxtyping._typeguard import typechecked
from torch import Tensor

B = TypeVar("B")


@jaxtyped(typechecker=typechecked)
@config_dataclass
class Batch:
    def clone(self: B) -> B:
        return self.__class__.from_batch(self, clone_tensors=True)

    def view(self: B) -> B:
        """
        Returns a new batch object where all tensors are a view of the original tensors.
        This may be used when we want to adjust the shapes of tensors in the batch object
        without chaning the original batch object but also without actually copying the underlying data.
        """
        kwargs = {}
        self = self.clone()
        for f in fields(self):
            field_value = getattr(self, f.name, None)
            if isinstance(field_value, Tensor):
                # no change in shape just yet
                kwargs[f.name] = field_value.view(field_value.shape)
            elif isinstance(field_value, Batch):
                kwargs[f.name] = field_value.view()
            else:
                kwargs[f.name] = field_value
        return self

    def unsqueeze(self: B, dim: int) -> B:
        for f in fields(self):
            data = getattr(self, f.name)
            if isinstance(data, Tensor) or isinstance(data, Batch):
                setattr(self, f.name, data.unsqueeze(dim))
        return self

    def repeat_interleave(self: B, dim: int, repeats: int) -> B:
        for f in fields(self):
            data = getattr(self, f.name)
            if isinstance(data, Tensor) or isinstance(data, Batch):
                setattr(self, f.name, data.repeat_interleave(repeats, dim=dim))
        return self

    def to_dict(self) -> Dict[str, Any]:
        return {f.name: getattr(self, f.name) for f in fields(self)}

    def to(self: B, device: torch.device) -> B:
        for f in fields(self):
            field_value = getattr(self, f.name, None)
            if isinstance(field_value, Tensor) or isinstance(field_value, Batch):
                setattr(self, f.name, field_value.to(device))
        return self

    def append(self, batch: "Batch", dim: int = 0, dim_named: Optional[Dict[str, int]] = None):
        """Append another batch to this batch.

        This method concatenates the tensors from another batch object to this batch object.
        For each field, it concatenates along the specified dimension.

        Args:
            batch: The batch to append to this batch
            dim: The default dimension to concatenate along (default: 0, typically batch dimension)
            dim_named: Optional dictionary mapping field names to dimensions for field-specific concatenation
                       This allows different fields to be concatenated along different dimensions

        Returns:
            None, modifies the batch in place

        Raises:
            AssertionError: If trying to append incompatible tensor types
            NotImplementedError: If trying to append fields that are not tensors or Batch objects
        """
        for f in fields(self):
            # concat along the batch dimension
            existing_data = getattr(self, f.name, None)
            if existing_data is None:
                continue
            additional_data = getattr(batch, f.name, None)

            # Determine which dimension to use for this field
            field_dim = dim
            if dim_named is not None and f.name in dim_named:
                field_dim = dim_named[f.name]

            if isinstance(existing_data, Tensor):
                assert isinstance(additional_data, Tensor), f"Cannot append {f.name} of type {type(existing_data)} with {type(additional_data)}"
                new_data = torch.cat(
                    [existing_data, additional_data],
                    dim=field_dim,
                )
                setattr(self, f.name, new_data)
            elif isinstance(existing_data, Batch):
                assert isinstance(additional_data, Batch), f"Cannot append {f.name} of type {type(existing_data)} with {type(additional_data)}"
                existing_data.append(additional_data, dim=field_dim, dim_named=dim_named)
            else:
                raise NotImplementedError(f"Appending {f.name} of type {type(existing_data)} is not implemented")

    def add_fields(self, batch: "Batch"):
        """
        Adds the fields from another batch to this batch.
        """
        for f in fields(batch):
            # make sure we don't overwrite existing fields
            # but we do allow overwriting fields that are None
            if getattr(self, f.name, None) is not None:
                # warning if we are overwriting a field
                warnings.warn(f"Tried overwriting field {f.name} of Batch type {type(self)} with value of Batch type {type(batch)}.")
                continue
            setattr(self, f.name, getattr(batch, f.name))

    def __getitem__(self: B, idx) -> B:
        self = self.clone()
        for f in fields(self):
            field_value = getattr(self, f.name, None)
            if field_value is not None and hasattr(field_value, "__getitem__"):
                setattr(self, f.name, field_value[idx])
        return self

    @property
    def shape(self) -> Dict[str, Tuple[int, ...]]:
        return {
            f.name: getattr(self, f.name).shape for f in fields(self) if isinstance(getattr(self, f.name), Tensor) or isinstance(getattr(self, f.name), Batch)
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Batch":
        return cls(**data)

    @classmethod
    def from_batch(cls, batch: "Batch", clone_tensors: bool = False, keep_all_tensors: bool = False):
        kwargs = {f.name: getattr(batch, f.name, None) for f in fields(cls)}

        if clone_tensors:
            for k, v in kwargs.items():
                if isinstance(v, Tensor) or isinstance(v, Batch):
                    kwargs[k] = v.clone()

        return cls(**kwargs)

    @classmethod
    def concat(cls, batches: List[B], dim: int = 0, dim_named: Optional[Dict[str, int]] = None) -> B:
        """Concatenate a list of batches along a specified dimension.

        This method concatenates a list of batch objects along the specified dimension.
        For each field, it concatenates along the specified dimension.

        Args:
            batches: List of batches to concatenate
            dim: The dimension to concatenate along (default: 0, typically batch dimension)
            dim_named: Optional dictionary mapping field names to dimensions for field-specific concatenation

        Returns:
            A new batch object with concatenated data
        """
        if not batches:
            raise ValueError("Cannot concatenate empty list of batches")

        # Get all field names from the first batch
        field_names = [f.name for f in fields(batches[0])]

        batch_class = type(batches[0])

        # Initialize dict to store concatenated tensors
        concat_data = {}

        # For each field, concatenate the tensors from all batches
        for field_name in field_names:
            field_values = [getattr(batch, field_name, None) for batch in batches]
            # check all field values have same type
            if not all(isinstance(value, type(field_values[0])) for value in field_values):
                raise ValueError(
                    f"All field values must have the same type of {type(field_values[0])}. Found types: {set(type(value) for value in field_values)}"
                )
            if field_values:
                # Get concatenation dimension for this field
                field_dim = dim_named.get(field_name, dim) if dim_named else dim

                if isinstance(field_values[0], Tensor):
                    # Concatenate tensors
                    concat_data[field_name] = torch.cat(field_values, dim=field_dim)
                elif isinstance(field_values[0], Batch):
                    # Recursively concatenate nested batch objects
                    concat_data[field_name] = type(field_values[0]).concat(field_values, dim=field_dim, dim_named=dim_named)
                elif field_values[0] is None:
                    concat_data[field_name] = None
                else:
                    # For non-tensor/batch fields, use the first value
                    concat_data[field_name] = field_values[0]
            else:
                concat_data[field_name] = None

        return batch_class(**concat_data)

    def __eq__(self, other: B) -> bool:
        if not isinstance(other, type(self)):
            return False
        for f in fields(self):
            if isinstance(getattr(self, f.name), Tensor):
                if not torch.all(getattr(self, f.name) == getattr(other, f.name)):
                    return False
            elif getattr(self, f.name) != getattr(other, f.name):
                return False
        return True


@jaxtyped(typechecker=typechecked)
@config_dataclass
class PairedData(Batch):
    x: Float[Tensor, "batch dim"]
    y: Float[Tensor, "batch dim"]
    indices: Optional[Integer[Tensor, "batch"]] = None

    @property
    def batch(self) -> int:
        return self.x.shape[0]

    @property
    def dim(self) -> int:
        return self.x.shape[-1]

    def __eq__(self, other) -> bool:
        return super().__eq__(other)


@jaxtyped(typechecker=typechecked)
@config_dataclass
class PairedGroupData(PairedData):
    """
    Batch of paired group data.
    (x, y, x', y') where (x, y) are the original datapoints and (x', y') are the transformed datapoints x' = G(x) and y' = G(y).
    So the two pairs (x, x') and (y, y') are related by the same transformation G.
    """

    x: Float[Tensor, "batch dim"]
    y: Float[Tensor, "batch dim"]
    x_prime: Float[Tensor, "batch dim"]
    y_prime: Float[Tensor, "batch dim"]
    indices: Optional[Integer[Tensor, " batch"]] = None
    actions_idx: Optional[Integer[Tensor, " batch"]] = None

    @property
    def batch(self) -> int:
        return self.x.shape[0]

    @property
    def dim(self) -> int:
        return self.x.shape[-1]

    def __eq__(self, other) -> bool:
        return super().__eq__(other)


@jaxtyped(typechecker=typechecked)
@config_dataclass
class SingleData(Batch):
    """
    Batch of labeled group data.
    (x, x') where x is the original datapoint and x' is the transformed datapoint x' = G_i(x).
    The transformation G_i is identified by actions_idx.
    """

    x: Float[Tensor, "batch dim"]
    indices: Optional[Integer[Tensor, " batch"]] = None
    class_idx: Optional[Integer[Tensor, " batch"]] = None

    @property
    def batch(self) -> int:
        return self.x.shape[0]

    @property
    def dim(self) -> int:
        return self.x.shape[-1]

    def __eq__(self, other) -> bool:
        return super().__eq__(other)


@jaxtyped(typechecker=typechecked)
@config_dataclass
class SinglePairedGroupData(Batch):
    """
    Batch of labeled group data.
    (x, x') where x is the original datapoint and x' is the transformed datapoint x' = G_i(x).
    The transformation G_i is identified by actions_idx.
    """

    x: Float[Tensor, " *batch dim"]
    x_prime: Float[Tensor, " *batch dim"]
    indices: Optional[Integer[Tensor, " *batch"]] = None
    actions_idx: Optional[Integer[Tensor, " *batch"]] = None
    class_idx: Optional[Integer[Tensor, " *batch"]] = None

    @property
    def batch(self) -> int:
        return self.x.shape[0]

    @property
    def dim(self) -> int:
        return self.x.shape[-1]

    def __eq__(self, other) -> bool:
        return super().__eq__(other)


@jaxtyped(typechecker=typechecked)
@config_dataclass
class AuxilaryVariables(Batch):
    """Collection of auxilary variables of a dataset"""

    trial_id: Optional[Integer[Tensor, " batch"]] = None
    trial_time: Optional[Union[Float[Tensor, " batch"], Integer[Tensor, " batch"]]] = None

    def __eq__(self, other: B) -> bool:
        return super().__eq__(other)


@jaxtyped(typechecker=typechecked)
@config_dataclass
class GroundTruthData(Batch):
    """Batch of ground truth data."""

    index: Integer[Tensor, " batch"]
    observed: Union[PairedGroupData, SinglePairedGroupData]
    latents: Optional[Union[PairedGroupData, SinglePairedGroupData]] = None
    # Should be BaseDynamicsModel but can't import it here because of circular import
    # also can't use TYPE_CHECKING because jaxtyping doesn't like forward references
    dynamics_model: Optional[torch.nn.Module] = None
    actions_idx: Optional[Integer[Tensor, " batch"]] = None
    class_idx: Optional[Integer[Tensor, " batch"]] = None
    auxilary: AuxilaryVariables = field(default_factory=AuxilaryVariables)

    def __eq__(self, other) -> bool:
        return super().__eq__(other)

    @property
    def num_samples(self) -> int:
        return self.index.shape[0]


@jaxtyped(typechecker=typechecked)
@config_dataclass
class ContrastiveGroupBatch(Batch):
    """
    Additional negative datapoints
    """

    positives: Union[PairedGroupData, SinglePairedGroupData]
    negatives: Union[PairedData, SingleData]

    def __eq__(self, other) -> bool:
        return super().__eq__(other)

    def slice_latent_dim(
        self,
        dim_slice: slice,
    ) -> "ContrastiveGroupBatch":
        """
        Create a new ContrastiveGroupBatch with embeddings sliced along the latent dimension.

        Args:
            embeddings: Original embeddings
            dim_slice: Slice to apply to the latent dimension

        Returns:
            Sliced embeddings
        """
        self = self.clone()
        self.positives.x = self.positives.x[..., dim_slice]
        self.positives.x_prime = self.positives.x_prime[..., dim_slice]
        self.negatives.x = self.negatives.x[..., dim_slice]
        return self

    def normalize(self: B) -> B:
        self = self.clone()
        # normalize x, x_prime, and negative x
        self.positives.x = torch.nn.functional.normalize(self.positives.x, dim=-1)
        self.positives.x_prime = torch.nn.functional.normalize(self.positives.x_prime, dim=-1)
        self.negatives.x = torch.nn.functional.normalize(self.negatives.x, dim=-1)
        return self


@jaxtyped(typechecker=typechecked)
@dataclass
class ContrastiveLossBatch(Batch):
    """Batch of latent variables with indices."""

    reference: Float[Tensor, "batch dim"]
    positive: Float[Tensor, "batch dim"]
    negative: Float[Tensor, "neg_batch dim"]
    reference_indices: Optional[Integer[Tensor, "batch"]] = None
    positive_indices: Optional[Integer[Tensor, "batch"]] = None
    negative_indices: Optional[Integer[Tensor, " neg_batch"]] = None

    @property
    def batch(self) -> int:
        return self.reference.shape[0]

    @property
    def dim(self) -> int:
        return self.reference.shape[-1]

    def __eq__(self, other) -> bool:
        return super().__eq__(other)


@jaxtyped(typechecker=typechecked)
@config_dataclass
class PairedContrastiveLossBatch(Batch):
    """Batch of contrastive loss for paired data."""

    x: ContrastiveLossBatch
    y: ContrastiveLossBatch

    def __eq__(self, other) -> bool:
        return super().__eq__(other)


@jaxtyped(typechecker=typechecked)
@config_dataclass
class DynamicsPredictions(Batch):
    """Batch of SLDS predictions."""

    x_prime: Float[Tensor, "batch  dim"]
    y_prime: Optional[Float[Tensor, "batch  dim"]] = None
    Qx: Optional[Float[Tensor, "batch dim dim"]] = None
    Qy: Optional[Float[Tensor, "batch dim dim"]] = None
    indices: Optional[Integer[Tensor, "batch"]] = None

    @property
    def batch(self) -> int:
        return self.x_prime.shape[0]

    @property
    def dim(self) -> int:
        return self.x_prime.shape[-1]

    def __eq__(self, other) -> bool:
        return super().__eq__(other)


@jaxtyped(typechecker=typechecked)
@config_dataclass
class GumbelPairedPredictions(Batch):
    """Batch of SLDS predictions."""

    x_prime: Float[Tensor, "batch gumbel_samples dim"]
    y_prime: Float[Tensor, "batch gumbel_samples dim"]
    mode_samples: Float[Tensor, "batch gumbel_samples num_modes"]

    @property
    def batch(self) -> int:
        return self.x_prime.shape[0]

    @property
    def gumbel_samples(self) -> int:
        return self.x_prime.shape[1]

    @property
    def dim(self) -> int:
        return self.x_prime.shape[-1]

    def __eq__(self, other) -> bool:
        return super().__eq__(other)


@jaxtyped(typechecker=typechecked)
@config_dataclass
class GroupSolverPrediction(Batch):
    """Batch of dynamics solver validation predictions."""

    embeddings: Union[PairedGroupData, SinglePairedGroupData]
    dynamics: Optional[Union[GumbelPairedPredictions, DynamicsPredictions]] = None

    def __eq__(self, other) -> bool:
        return super().__eq__(other)


@jaxtyped(typechecker=typechecked)
@config_dataclass
class EmbeddingPrediction(Batch):
    """Batch of embedding predictions."""

    x: Float[Tensor, "batch dim"]
    indices: Optional[Integer[Tensor, " batch"]] = None

    def __eq__(self, other) -> bool:
        return super().__eq__(other)


@jaxtyped(typechecker=typechecked)
@config_dataclass
class GroupSolverContrastivePrediction(Batch):
    """Batch of dynamics solver contrastive predictions."""

    embeddings: ContrastiveGroupBatch
    dynamics: Optional[Union[GumbelPairedPredictions, DynamicsPredictions]] = None

    def __eq__(self, other) -> bool:
        return super().__eq__(other)
