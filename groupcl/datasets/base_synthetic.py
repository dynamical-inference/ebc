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
from jaxtyping._typeguard import typechecked
from groupcl.datasets.base import BaseDataset
from groupcl.datasets.paired_actions import PairedActionsDataset
from config_dataclass import config_dataclass, config_field, check_initialized
from groupcl.utils import datatypes as dt

T = TypeVar("T")


@jaxtyped(typechecker=typechecked)
@config_dataclass
class BaseSyntheticDataset(
        BaseDataset,
        ABC,
):
    """
    Synthetic dataset of paired points under some transformations.
    """

    root: Union[str, Path] = "./data"
    force_regenerate: bool = False

    # Base fields about the configuration of the dataset
    version: str = config_field(default="1.0")
    seed: int = config_field(default=42)
    # Internal state (not part of metadata/config)
    _data: Dict[str, Tensor] = field(default_factory=dict,
                                     init=False,
                                     repr=False)

    def __lazy_post_init__(self):
        self.root = Path(self.root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._initialize_dataset()
        self.to(self.device)

    @property
    def dataset_id(self) -> str:
        """Generate unique dataset ID from configuration fields."""
        return self.config_hash

    @property
    @check_initialized
    def dataset_path(self) -> Path:
        """Get base path for dataset storage."""
        return Path(self.root) / self.__class__.__name__ / self.dataset_id

    @property
    def data_path(self) -> Path:
        return self.dataset_path / "data.pt"

    def _load_data(self):
        """Load data  from disk"""
        data_path = self.data_path
        self._data = torch.load(data_path,
                                weights_only=False,
                                map_location=torch.device('cpu'))

    def _acquire_data(self) -> None:
        """Implement data acquisition"""
        # Set all random seeds
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)
        self._data = self._generate_data()
        self.save()

    def _initialize_dataset(self) -> None:
        """Initialize dataset ensuring proper seeding and storage."""
        if not self.force_regenerate:
            try:
                self.locked_recursive_load_additional(self.dataset_path,
                                                      timeout=60 * 5)
                self.validate_data()
                print(f"Loaded {self.__class__.__name__} from disk", flush=True)
                return
            except Exception as e:
                print(
                    f"Could not load{self.__class__.__name__} from {self.dataset_path}: {e}",
                    flush=True)

        print(
            f"Could not load {self.__class__.__name__} from disk, acquiring data...",
            flush=True)
        # Acquire and validate data
        try:
            self._acquire_data()
        except Exception as e:
            # Clean up any partially created files
            if self.dataset_path.exists():
                import shutil
                shutil.rmtree(self.dataset_path)
            raise e

        # Load from disk for consistency
        # longer time out, since dataset may be large
        self.locked_recursive_load_additional(self.dataset_path, timeout=60)
        self.validate_data()

    def save(self, path: Optional[Path] = None) -> Path:
        """We ignore the path argument and always use the self.dataset_path"""
        return super().save(self.dataset_path)

    def _save_additional(self, path: Path) -> None:
        data_path = self.data_path
        data_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self._data, data_path)

    def _load_additional(self, path: Path) -> None:
        data_path = self.data_path
        self._data = torch.load(data_path,
                                weights_only=False,
                                map_location=torch.device('cpu'))

    def validate_data(self):
        try:
            self._validate_data()
        except Exception as e:
            raise ValueError(f"Data validation failed: {e}") from e

    @property
    def _data_keys(self) -> List[str]:
        """
        Returns a list of keys of the self._data dictionary that can be indexed (and are required to be present).
        The first key is used to determine the length of the dataset.
        All keys are passed to _validate_types for type checking.
        """
        raise NotImplementedError("Data keys not implemented")

    def _validate_data(self):
        """Validate acquired data."""
        # check that all required keys are present
        missing_keys = [key for key in self._data_keys if key not in self._data]
        if missing_keys:
            raise ValueError(
                f"Missing required keys in self._data: {missing_keys}")

        self._validate_data_types(**self._data)
        return True

    @abstractmethod
    def _validate_data_types(
        self,
        **kwargs,
    ):
        """Call this function to run jaxtyping typechecks on the data."""
        pass

    def __getitems__(
        self, indices: Union[List[int], Integer[Tensor, " batch_shape"]]
    ) -> Dict[str, Shaped[Tensor, " batch_shape ..."]]:
        return {key: self._data[key][indices] for key in self._data_keys}

    def to(self: T, device: torch.device) -> T:
        for key in self._data_keys:
            self._data[key] = self._data[key].to(device)
        return super().to(device)

    @abstractmethod
    def _generate_data(
            self
    ) -> Dict[str, Union[Float[Tensor, "..."], Integer[Tensor, "..."]]]:
        """Generate data."""
        raise NotImplementedError("Data generation not implemented")

    @jaxtyped(typechecker=typechecked)
    def split(self: T, indices: Integer[Tensor, " *batch_shape"]) -> T:
        """Split the dataset into a new dataset with the given indices."""
        new_dataset = self.clone()
        new_dataset._data = self[indices]
        return new_dataset


@jaxtyped(typechecker=typechecked)
@config_dataclass
class BaseSyntheticPairedDataset(
        BaseSyntheticDataset,
        PairedActionsDataset,
        ABC,
):
    """
    Synthetic dataset of paired points under some transformations.
    
    For each group action, we:
    1. Sample pairs of latent points (latent_1, latent_2)
    2. Apply the same transformation to get (latent_1_prime, latent_2_prime)
    3. Apply mixing to get observed variables (observed_1, observed_2, observed_1_prime, observed_2_prime)
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
            "y",
            "x_prime",
            "y_prime",
            "latents_x",
            "latents_y",
            "latents_x_prime",
            "latents_y_prime",
            "actions_idx",
        ]

    @jaxtyped(typechecker=typechecked)
    def _validate_data_types(
        self,
        x: Float[Tensor, "num_samples feature_dim"],
        y: Float[Tensor, "num_samples feature_dim"],
        x_prime: Float[Tensor, "num_samples feature_dim"],
        y_prime: Float[Tensor, "num_samples feature_dim"],
        latents_x: Float[Tensor, "num_samples latent_dim"],
        latents_y: Float[Tensor, "num_samples latent_dim"],
        latents_x_prime: Float[Tensor, "num_samples latent_dim"],
        latents_y_prime: Float[Tensor, "num_samples latent_dim"],
        **kwargs,
    ):
        """Call this function to run jaxtyping typechecks on the data."""
        pass

    @jaxtyped(typechecker=typechecked)
    def get_observed_data(
            self, indices: Integer[Tensor,
                                   " *batch_shape"]) -> dt.PairedGroupData:
        """Get a batch of data from the dataset."""
        return dt.PairedGroupData(
            x=self._data["x"][indices],
            y=self._data["y"][indices],
            x_prime=self._data["x_prime"][indices],
            y_prime=self._data["y_prime"][indices],
            indices=indices,
            actions_idx=self._data["actions_idx"][indices],
        )

    @jaxtyped(typechecker=typechecked)
    def get_latent_data(
            self, indices: Integer[Tensor,
                                   " *batch_shape"]) -> dt.PairedGroupData:
        """Get the latent data of the dataset."""
        return dt.PairedGroupData(
            x=self._data["latents_x"][indices],
            y=self._data["latents_y"][indices],
            x_prime=self._data["latents_x_prime"][indices],
            y_prime=self._data["latents_y_prime"][indices],
            indices=indices,
            actions_idx=self._data["actions_idx"][indices],
        )

    def __len__(self) -> int:
        return self._data["x"].shape[0]

    @property
    @check_initialized
    def ground_truth_data(self) -> dt.GroundTruthData:
        gt_batch = super().ground_truth_data
        gt_batch.latents = self.get_latent_data(gt_batch.index)
        gt_batch.actions_idx = self._data["actions_idx"][gt_batch.index]
        return dt.GroundTruthData.from_batch(gt_batch)
