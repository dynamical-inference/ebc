from abc import ABC, abstractmethod
from jaxtyping import Float, Integer
import torch
from config_dataclass import Configurable, config_dataclass, config_field
from jaxtyping import jaxtyped
from jaxtyping._typeguard import typechecked
from dataclasses import dataclass
from typing import Union, List
import math
from itertools import product


@jaxtyped(typechecker=typechecked)
@config_dataclass
class MatrixTransform(Configurable, ABC):

    @jaxtyped(typechecker=typechecked)
    def __call__(
        self,
        matrix: Float[torch.Tensor, "num_samples feature_dim feature_dim"],
    ) -> Float[torch.Tensor, "num_samples feature_dim feature_dim"]:
        return self.transform(matrix)

    @abstractmethod
    def transform(
        self, matrix: Float[torch.Tensor, "num_samples feature_dim feature_dim"]
    ) -> Float[torch.Tensor, "num_samples feature_dim feature_dim"]:
        pass


@jaxtyped(typechecker=typechecked)
@config_dataclass
class MatrixTransformCombination(MatrixTransform, ABC):
    """
    Combines multiple matrix transforms into a single transform.
    """

    transforms: List[MatrixTransform] = config_field(default_factory=list)

    def transform(
        self, matrix: Float[torch.Tensor, "num_samples feature_dim feature_dim"]
    ) -> Float[torch.Tensor, "num_samples feature_dim feature_dim"]:
        for transform in self.transforms:
            matrix = transform(matrix)
        return matrix


@jaxtyped(typechecker=typechecked)
@config_dataclass
class Translate(MatrixTransform):
    """
    Translates the data along the specified axis.
    
    Args:
        axis: The axis to translate along (1 for y-axis, 2 for x-axis)
        shift: Number of positions to shift (default: 1)
    """

    axis: int = config_field(default=-1)
    shift: int = config_field(default=1)

    def validate_config(self):
        if self.axis not in [-1, -2]:
            raise ValueError("Axis must be either -1 (y-axis) or -2 (x-axis)")

    def transform(
        self, matrix: Float[torch.Tensor, "num_samples feature_dim feature_dim"]
    ) -> Float[torch.Tensor, "num_samples feature_dim feature_dim"]:
        return torch.roll(
            matrix,
            shifts=self.shift,
            dims=self.axis,
        )


@jaxtyped(typechecker=typechecked)
@config_dataclass
class Rotate90(MatrixTransform):
    """
    Rotates the data by k*90 degrees.
    
    Args:
        k: Number of 90-degree rotations (default: 1)
    """
    k: int = config_field(default=1)

    def _transform_90(
        self, matrix: Float[torch.Tensor, "num_samples feature_dim feature_dim"]
    ) -> Float[torch.Tensor, "num_samples feature_dim feature_dim"]:
        """Perform a single 90-degree rotation."""
        return matrix.transpose(1, 2).flip(1)

    def transform(
        self, matrix: Float[torch.Tensor, "num_samples feature_dim feature_dim"]
    ) -> Float[torch.Tensor, "num_samples feature_dim feature_dim"]:
        k_mod = self.k % 4
        if k_mod == 0:
            return matrix

        result = matrix
        for _ in range(k_mod):
            result = self._transform_90(result)

        return result


@jaxtyped(typechecker=typechecked)
@config_dataclass
class MirrorX(MatrixTransform):
    """
    Mirrors the data along the x-axis.
    """

    def transform(
        self, matrix: Float[torch.Tensor, "num_samples feature_dim feature_dim"]
    ) -> Float[torch.Tensor, "num_samples feature_dim feature_dim"]:
        return matrix.flip(2)


@jaxtyped(typechecker=typechecked)
@config_dataclass
class MirrorY(MatrixTransform):
    """
    Mirrors the data along the y-axis.
    """

    def transform(
        self, matrix: Float[torch.Tensor, "num_samples feature_dim feature_dim"]
    ) -> Float[torch.Tensor, "num_samples feature_dim feature_dim"]:
        return matrix.flip(1)


@jaxtyped(typechecker=typechecked)
@config_dataclass
class MatrixTransformCollection(Configurable, ABC):
    """
    Collection of matrix transforms.
    """

    transforms: List[MatrixTransform] = config_field(
        default_factory=lambda: [Translate(axis=-1, shift=1)])

    def __post_init__(self):
        super().__post_init__()
        # order transforms by sort_key
        self.transforms = sorted(self.transforms, key=lambda x: str(x))

    def __len__(self):
        return len(self.transforms)

    @jaxtyped(typechecker=typechecked)
    def __call__(
        self,
        x: Union[
            Float[torch.Tensor, "num_samples feature_dim feature_dim"],
            Float[torch.Tensor, "num_samples observed_dim"],
        ],
        transform_idx: Integer[torch.Tensor, "num_samples"],
    ) -> Union[
            Float[torch.Tensor, "num_samples feature_dim feature_dim"],
            Float[torch.Tensor, "num_samples observed_dim"],
    ]:
        input_shape = x.shape
        if len(input_shape) == 3:
            feature_dim = input_shape[-1]
        elif len(input_shape) == 2:
            observed_dim = input_shape[-1]
            feature_dim = int(math.sqrt(observed_dim))
        else:
            raise ValueError(f"Input shape {input_shape} is not supported")
        matrices = x.view(-1, feature_dim, feature_dim)

        unique_transforms = torch.unique(transform_idx)
        transformed_matrices = torch.zeros_like(matrices)
        for t_idx in unique_transforms:
            t_idx: int = t_idx.item()
            t_mask = transform_idx == t_idx
            transformed_matrices[t_mask] = self.transforms[t_idx](
                matrices[t_mask])

        # reshape the matrices back to the original shape
        transformed_matrices = transformed_matrices.view(*input_shape)

        return transformed_matrices

    @staticmethod
    def from_transform_types(
            *transform_types: List[MatrixTransform]
    ) -> "MatrixTransformCollection":
        """
        Performs a cross product of all transforms        
        """

        transform_combinations = product(*transform_types)
        transform_combinations = [
            MatrixTransformCombination(
                transforms=list(t)) if len(t) > 1 else t[0]
            for t in transform_combinations
        ]
        return MatrixTransformCollection(transforms=transform_combinations)
