from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Tuple, TypeVar, Union

import numpy as np
import torch
from config_dataclass import Configurable, config_dataclass, config_field
from jaxtyping import Integer, jaxtyped
from jaxtyping._typeguard import typechecked
from torch import Tensor

if TYPE_CHECKING:
    from groupcl.datasets.base import BaseDataset
    from groupcl.datasets.cebra import HippcampusRatDataset

T = TypeVar("T", bound="BaseDataset")


@jaxtyped(typechecker=typechecked)
@config_dataclass
class DatasetSplit(Configurable, ABC):
    """Handles dataset splitting into train, val and test sets with reproducible seeding."""

    seed: int = config_field(default=42)

    train_ratio: float = config_field(default=0.7)
    test_ratio: float = config_field(default=0.15)

    @property
    def val_ratio(self) -> float:
        _val_ratio = 1 - self.train_ratio - self.test_ratio
        if _val_ratio <= 0:
            raise ValueError(
                f"Validation ratio is less than 0. Got {_val_ratio} with train_ratio={self.train_ratio} and test_ratio={self.test_ratio}."
            )
        return _val_ratio

    def reseed(self):
        self.rng = np.random.RandomState(self.seed)

    def create_split(
        self,
        dataset: T,
    ) -> Tuple[T, T, T]:
        """Create reproducible train/val/test splits."""
        self.reseed()
        train_idx, val_idx, test_idx = self._create_split(dataset)
        return (
            dataset.split(train_idx),
            dataset.split(val_idx),
            dataset.split(test_idx),
        )

    @abstractmethod
    @jaxtyped(typechecker=typechecked)
    def _create_split(
        self,
        dataset: "BaseDataset",
    ) -> Tuple[
        Integer[Union[Tensor, np.ndarray], " train_size"],
        Integer[Union[Tensor, np.ndarray], " val_size"],
        Integer[Union[Tensor, np.ndarray], " test_size"],
    ]:
        pass

    def absolut_split_sizes(self, total_size: int) -> Tuple[int, int, int]:
        """Calculate the absolute sizes of the splits."""
        train_size = int(self.train_ratio * total_size)
        val_size = int(self.val_ratio * total_size)
        test_size = total_size - train_size - val_size
        assert train_size > 0, "Absolute train size is less than 0"
        assert val_size > 0, "Absolute validation size is less than 0"
        assert test_size > 0, "Absolute test size is less than 0"
        return train_size, val_size, test_size


@config_dataclass
class RandomSplit(DatasetSplit):
    """Handles dataset splitting with reproducible seeding."""

    @jaxtyped(typechecker=typechecked)
    def _create_split(
        self,
        dataset: "BaseDataset",
    ) -> Tuple[
        Integer[Union[Tensor, np.ndarray], " train_size"],
        Integer[Union[Tensor, np.ndarray], " val_size"],
        Integer[Union[Tensor, np.ndarray], " test_size"],
    ]:
        """Create reproducible train/val/test splits."""

        dataset_index = dataset.index

        rand_permutation = np.arange(len(dataset_index))
        self.rng.shuffle(rand_permutation)

        (
            train_size,
            val_size,
            test_size,
        ) = self.absolut_split_sizes(len(dataset_index))

        train_rand_permutation = rand_permutation[:train_size]
        val_rand_permutation = rand_permutation[train_size : train_size + val_size]
        test_rand_permutation = rand_permutation[train_size + val_size :]

        return (
            dataset_index[train_rand_permutation],
            dataset_index[val_rand_permutation],
            dataset_index[test_rand_permutation],
        )


@config_dataclass
class SequentialSplit(DatasetSplit):
    """For splitting data into sequential train/val/test splits."""

    @jaxtyped(typechecker=typechecked)
    def _create_split(
        self,
        dataset: "BaseDataset",
    ) -> Tuple[
        Integer[Union[Tensor, np.ndarray], " train_size"],
        Integer[Union[Tensor, np.ndarray], " val_size"],
        Integer[Union[Tensor, np.ndarray], " test_size"],
    ]:
        """Create reproducible train/val/test splits."""

        dataset_index = dataset.index

        (
            train_size,
            val_size,
            test_size,
        ) = self.absolut_split_sizes(len(dataset_index))

        train_index = dataset_index[:train_size]
        val_index = dataset_index[train_size : train_size + val_size]
        test_index = dataset_index[train_size + val_size :]

        return (
            train_index,
            val_index,
            test_index,
        )


@config_dataclass
class StratifiedSystemsSplit(DatasetSplit):
    """Uses stratified sampling based on actions_idx to split the dataset into train/val/test sets."""

    @jaxtyped(typechecker=typechecked)
    def _create_split(
        self,
        dataset: "BaseDataset",
    ) -> Tuple[
        Integer[Union[Tensor, np.ndarray], " train_size"],
        Integer[Union[Tensor, np.ndarray], " val_size"],
        Integer[Union[Tensor, np.ndarray], " test_size"],
    ]:
        """Create reproducible train/val/test splits."""

        dataset_index = dataset.index.to(dataset.device)
        actions_idx = dataset.get_action_idx(dataset_index)
        assert actions_idx is not None, "actions_idx is not set"
        unique_actions_idx, counts = torch.unique(actions_idx, return_counts=True)

        rand_permutation = np.arange(len(dataset_index))
        self.rng.shuffle(rand_permutation)

        shuffled_dataset_index = dataset_index[rand_permutation]
        shuffled_actions_idx = actions_idx[rand_permutation]

        train_idx = []
        val_idx = []
        test_idx = []
        for system_id, count in zip(unique_actions_idx, counts):
            system_mask = shuffled_actions_idx == system_id
            system_dataset_index = shuffled_dataset_index[system_mask]
            assert count == len(system_dataset_index), "Count mismatch"
            (
                train_size,
                val_size,
                test_size,
            ) = self.absolut_split_sizes(count)

            train_idx.append(system_dataset_index[:train_size])
            val_idx.append(system_dataset_index[train_size : train_size + val_size])
            test_idx.append(system_dataset_index[train_size + val_size :])

        train_idx = torch.cat(train_idx)
        val_idx = torch.cat(val_idx)
        test_idx = torch.cat(test_idx)

        return (
            train_idx,
            val_idx,
            test_idx,
        )


@config_dataclass
class OODSystemsSplit(DatasetSplit):
    """Splits into train/val/test sets, such that every type of system is only in one of the sets."""

    @jaxtyped(typechecker=typechecked)
    def _create_split(
        self,
        dataset: "BaseDataset",
    ) -> Tuple[
        Integer[Union[Tensor, np.ndarray], " train_size"],
        Integer[Union[Tensor, np.ndarray], " val_size"],
        Integer[Union[Tensor, np.ndarray], " test_size"],
    ]:
        """Create reproducible train/val/test splits."""

        dataset_index = dataset.index.to(dataset.device)
        actions_idx = dataset.get_action_idx(dataset_index)

        unique_actions_idx, counts = torch.unique(actions_idx, return_counts=True)

        rand_permutation = np.arange(len(unique_actions_idx))
        self.rng.shuffle(rand_permutation)

        shuffled_unique_actions_idx = unique_actions_idx[rand_permutation]

        (
            train_size,
            val_size,
            test_size,
        ) = self.absolut_split_sizes(len(unique_actions_idx))

        # seperate shuffled actions into train, val, test sets
        train_unique_actions_idx = shuffled_unique_actions_idx[:train_size]
        val_unique_actions_idx = shuffled_unique_actions_idx[
            train_size : train_size + val_size
        ]
        test_unique_actions_idx = shuffled_unique_actions_idx[train_size + val_size :]

        # find indices of actions in train, val, test sets
        train_idx = torch.where(
            actions_idx.unsqueeze(1) == train_unique_actions_idx.unsqueeze(0)
        )[0]
        val_idx = torch.where(
            actions_idx.unsqueeze(1) == val_unique_actions_idx.unsqueeze(0)
        )[0]
        test_idx = torch.where(
            actions_idx.unsqueeze(1) == test_unique_actions_idx.unsqueeze(0)
        )[0]

        # compute rations of train, val, test sets
        effective_train_ratio = len(train_idx) / len(dataset_index)
        effective_val_ratio = len(val_idx) / len(dataset_index)
        effective_test_ratio = len(test_idx) / len(dataset_index)

        print(
            f"Effective train ratio: {effective_train_ratio} (desired: {self.train_ratio})"
        )
        print(
            f"Effective val ratio: {effective_val_ratio} (desired: {1 - self.train_ratio - self.test_ratio})"
        )
        print(
            f"Effective test ratio: {effective_test_ratio} (desired: {self.test_ratio})"
        )

        return (
            train_idx,
            val_idx,
            test_idx,
        )

    def create_split(
        self,
        dataset: T,
    ) -> Tuple[T, T, T]:
        """Create reproducible train/val/test splits."""
        train_dataset, val_dataset, test_dataset = super().create_split(dataset)
        # now we want to assert these datasets reaaally don't share any actions

        train_actions_idx = train_dataset.get_action_idx(train_dataset.index)
        val_actions_idx = val_dataset.get_action_idx(val_dataset.index)
        test_actions_idx = test_dataset.get_action_idx(test_dataset.index)

        # get unique actions
        train_unique_actions_idx = torch.unique(train_actions_idx)
        val_unique_actions_idx = torch.unique(val_actions_idx)
        test_unique_actions_idx = torch.unique(test_actions_idx)

        # assert no overlap
        assert not torch.any(
            val_unique_actions_idx.unsqueeze(1) == test_unique_actions_idx.unsqueeze(0)
        ), "Validation and test sets share actions"
        assert not torch.any(
            train_unique_actions_idx.unsqueeze(1) == val_unique_actions_idx.unsqueeze(0)
        ), "Train and validation sets share actions"
        assert not torch.any(
            train_unique_actions_idx.unsqueeze(1)
            == test_unique_actions_idx.unsqueeze(0)
        ), "Train and test sets share actions"

        return train_dataset, val_dataset, test_dataset


import numpy as np
import sklearn


def split_slices(direction, split_no=2, **kwargs):
    """Split the dataset into 3-fold nested cross validation scheme.

    The recordings are parsed into trials and split into a train, valid, test set with 3-fold nested cross validation scheme.

    Args:
        split: The split to use. Choose among 'train', 'valid', 'test', 'all', and 'wo_test'(all trials except test split).

    """

    # direction_change_idx = np.where(self.index[1:, 1] != self.index[:-1, 1])[0]
    direction_change_idx = np.where(direction[1:] != direction[:-1])[0]
    trial_change_idx = np.append(
        np.insert(direction_change_idx[1::2], 0, 0), len(direction)
    )
    total_trials_num = len(trial_change_idx) - 1

    outer_folds = np.array_split(
        np.arange(total_trials_num), 3
    )  ## Divide data into 3 equal trial-sized array
    inner_folds = sklearn.model_selection.KFold(
        n_splits=3, random_state=None, shuffle=False
    )
    ## in each outer fold array, make train, valid, test split

    train_trials = []
    valid_trials = []
    test_trials = []

    for out_fold in outer_folds:
        train_trial, val_test_trial = list(inner_folds.split(out_fold))[split_no]
        test_trial, valid_trial = np.array_split(val_test_trial, 2)
        train_trials.extend(np.array(out_fold)[train_trial])
        valid_trials.extend(np.array(out_fold)[valid_trial])
        test_trials.extend(np.array(out_fold)[test_trial])

    # create selected indices for each split

    get_selected_slices = lambda the_trials: tuple(
        slice(trial_change_idx[i], trial_change_idx[i + 1]) for i in the_trials
    )

    train_slices = get_selected_slices(train_trials)
    valid_slices = get_selected_slices(valid_trials)
    test_slices = get_selected_slices(test_trials)

    return train_slices, valid_slices, test_slices


@config_dataclass
class CebraRatSplit(DatasetSplit):
    """
    For splitting cebra rat dataset into train/val/test splits
    according to the 3-fold nested cross validations schema defined in the cebra paper.
    """

    @jaxtyped(typechecker=typechecked)
    def _create_split(
        self,
        dataset: "HippcampusRatDataset",
    ) -> Tuple[
        Integer[Union[Tensor, np.ndarray], " train_size"],
        Integer[Union[Tensor, np.ndarray], " val_size"],
        Integer[Union[Tensor, np.ndarray], " test_size"],
    ]:
        """Create reproducible train/val/test splits."""

        dataset_index = dataset.index

        # call the cebra dataset split function
        direction = dataset._data["direction_right"]
        train_slices, valid_slices, test_slices = split_slices(direction.cpu())

        select_slices = lambda x, slices: torch.cat([x[slice] for slice in slices])

        train_index = select_slices(dataset_index, train_slices)
        val_index = select_slices(dataset_index, valid_slices)
        test_index = select_slices(dataset_index, test_slices)

        return (
            train_index,
            val_index,
            test_index,
        )
