from dataclasses import field
from typing import Dict, List, Literal, Optional, TypeVar, Union

import cebra.datasets
import torch
from config_dataclass import config_dataclass, config_field
from jaxtyping import Float, Integer, Shaped, jaxtyped
from jaxtyping._typeguard import typechecked
from torch import Tensor

from groupcl.datasets.base import BaseDataset

T = TypeVar("T")


def concatenate_with_multiple_shifts(x: torch.Tensor, shifts: List[int]):
    """
    Concatenate a tensor with multiple shifts of itself.
    """
    results = []
    for i in range(len(x)):
        concatenated_list = [x[i, :]]  # Always include the current row
        for shift in shifts:
            if i - shift < 0:
                # add nans as padding
                nans = torch.full_like(x[i, :], float("nan"))
                concatenated_list.append(nans)
                # concatenated_list.append(x[i, :])
            else:
                concatenated_list.append(x[i - shift, :])  # Concatenate with shifted row

        concatenated = torch.cat(concatenated_list, dim=0)
        results.append(concatenated)

    return torch.stack(results)  # Convert list to tensor


@jaxtyped(typechecker=typechecked)
@config_dataclass
class HippcampusRatDataset(BaseDataset):
    """
    Cebra dataset.
    """

    name: Literal[
        "achilles",
        "buddy",
        "cicero",
        "gatsby",
    ] = config_field(default="achilles")

    # time delay preprocessing
    time_delayed_step: Optional[int] = config_field(default=None, skip_default=True)
    time_delayed_num_steps: Optional[int] = config_field(default=None, skip_default=True)

    behavior_variable_key: Union[
        str,
        List[
            Literal[
                "trial_id",
                "time",
                "position",
                "direction_left",
                "direction_right",
            ]
        ],
    ] = config_field(default_factory=lambda: ["position"], skip_default=True)

    shuffle_neural: bool = config_field(default=False, skip_default=True)
    shuffle_behavior: bool = config_field(default=False, skip_default=True)
    seed: int = config_field(default=42, skip_default=True)

    # Internal state (not part of metadata/config)
    _data: Dict[str, Tensor] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self):
        # for backwards compatibility, handle string behavior_variable_key
        if isinstance(self.behavior_variable_key, str):
            if self.behavior_variable_key == "position_direction":
                self.behavior_variable_key = ["position", "direction_left", "direction_right"]
            else:
                self.behavior_variable_key = [self.behavior_variable_key]
        super().__post_init__()

    def __lazy_post_init__(self):
        self._cebra_dataset = cebra.datasets.hippocampus.SingleRatDataset(
            name=self.name,
            root="data",
            download=True,
        ).to("cpu")

        generator = torch.Generator()
        generator.manual_seed(self.seed)
        neural = self._cebra_dataset.neural.clone()
        continuous_index = self._cebra_dataset.continuous_index.clone()
        if self.shuffle_neural:
            neural = neural[torch.randperm(len(neural), generator=generator)]

        if self.shuffle_behavior:
            continuous_index = continuous_index[torch.randperm(len(continuous_index), generator=generator)]

        direction_left = continuous_index[:, 2].clone()
        direction_right = continuous_index[:, 1].clone()

        direction_change_idx = torch.where(direction_left[1:] != direction_left[:-1])[0]
        trial_change_idx = torch.cat(
            [
                torch.cat([torch.tensor([0]), direction_change_idx[1::2]]),
                torch.tensor([len(direction_left)]),
            ]
        )

        trial_id = torch.ones(len(direction_left), dtype=torch.int64) * -1
        for i in range(len(trial_change_idx) - 1):
            trial_id[trial_change_idx[i] : trial_change_idx[i + 1]] = i
        assert torch.all(trial_id != -1)

        sample_rate = 40
        self._data = dict(
            neural=neural,
            time=torch.arange(len(neural)) / sample_rate,
            position=continuous_index[:, 0].clone(),
            direction_right=continuous_index[:, 1].clone(),
            direction_left=continuous_index[:, 2].clone(),
            trial_id=trial_id,
        )

        if self.time_delayed_step is not None and self.time_delayed_num_steps is not None:
            shifts = [self.time_delayed_step * i for i in range(1, self.time_delayed_num_steps + 1)]
            neural_data = concatenate_with_multiple_shifts(
                self._data["neural"],
                shifts=shifts,
            )
            # some samples may contain padded nans now,
            # those samples we remove across the full dataset
            mask = ~torch.any(torch.isnan(neural_data), dim=1)
            self._data = dict(
                neural=neural_data[mask],
                time=self._data["time"][mask],
                position=self._data["position"][mask],
                direction_right=self._data["direction_right"][mask],
                direction_left=self._data["direction_left"][mask],
                trial_id=self._data["trial_id"][mask],
            )

        super().__lazy_post_init__()

    def __getitems__(self, indices: Union[List[int], Integer[Tensor, " batch_shape"]]) -> Dict[str, Shaped[Tensor, " batch_shape ..."]]:
        return {key: self._data[key][indices] for key in self._data.keys()}

    def to(self: T, device: torch.device) -> T:
        for key in self._data.keys():
            self._data[key] = self._data[key].to(device)
        return super().to(device)

    @jaxtyped(typechecker=typechecked)
    def split(self: T, indices: Integer[Tensor, " *batch_shape"]) -> T:
        """Split the dataset into a new dataset with the given indices."""
        new_dataset = self.clone()
        new_dataset._data = self[indices]
        return new_dataset

    @jaxtyped(typechecker=typechecked)
    @property
    def observed_variable(self) -> Float[Tensor, " num_samples num_features"]:
        return self._data["neural"]

    def get_observed_variable(self, indices: Integer[Tensor, " *batch_shape"]) -> Float[Tensor, " *batch_shape num_features"]:
        return self._data["neural"][indices]

    @jaxtyped(typechecker=typechecked)
    @property
    def get_behavior_variable(self, indices: Integer[Tensor, " *batch_shape"]) -> Float[Tensor, " *batch_shape"]:
        return self.behavior_variable[indices]

    @jaxtyped(typechecker=typechecked)
    @property
    def behavior_variable(
        self,
    ) -> Union[
        Float[Tensor, " num_samples"],
        Float[Tensor, " num_samples {len(self.behavior_variable_key)}"],
    ]:
        return torch.stack(
            [self._data[key] for key in self.behavior_variable_key],
            dim=1,
        ).to(torch.float32)

    def __len__(self) -> int:
        return len(self._data["neural"])

    @property
    def observed_dim(self) -> int:
        return self._data["neural"].shape[1]
