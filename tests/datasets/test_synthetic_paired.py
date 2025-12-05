import multiprocessing
import traceback
from pathlib import Path

import torch
from groupcl import datasets
from groupcl.models import mixing
from groupcl.models.dynamics import linear_dynamics
from groupcl.models.dynamics import utils

from typing import List


def split_seed(seed: int, num_splits: int) -> List[int]:
    import random
    # Set the random seed for reproducibility
    random_generator = random.Random(seed)
    # Generate num_splits random seeds
    return [random_generator.randint(1, 100000) for _ in range(num_splits)]


def get_syn_dataset_config(
    *,
    obs_dim=10,
    latent_dim=5,
    num_samples=1_000,
    num_systems=10,
    seed=42,
):

    num_groups = 2
    min_group_size = 2
    max_group_size = num_systems

    dataset_seed, lds_seed, mixing_seed = split_seed(seed, 3)

    dataset = datasets.SyntheticPairedRotationsDataset(
        lazy=True,
        seed=dataset_seed,
        num_samples=num_samples,
        mixing_model=mixing.LinearNonlinearMixingModel(
            seed=mixing_seed,
            output_dim=obs_dim,
            input_dim=latent_dim,
        ),
        dynamics_model=linear_dynamics.LinearDynamicsModel(
            seed=lds_seed,
            dim=latent_dim,
            num_systems=num_systems,
            initializer=utils.RotationGroupLDSParameters(
                min_group_size=min_group_size,
                max_group_size=max_group_size,
                num_groups=num_groups,
            ),
        ),
    )
    return dataset.to_dict()


def assert_datasets_equal(dataset1: datasets.PairedActionsDataset,
                          dataset2: datasets.PairedActionsDataset):

    assert torch.all(dataset1.index == dataset2.index), "Indices are not equal"
    observed_data1 = dataset1.get_observed_data(dataset1.index)
    observed_data2 = dataset2.get_observed_data(dataset2.index)
    assert observed_data1 == observed_data2, "Data is not equal"

    assert dataset1.auxilary_variables == dataset2.auxilary_variables, "Auxilary variables are not equal"


def assert_datasets_model_equal(dataset1: datasets.BaseSyntheticPairedDataset,
                                dataset2: datasets.BaseSyntheticPairedDataset):
    assert dataset1.dynamics_model == dataset2.dynamics_model, "Dynamics models are not equal"
    assert dataset1.mixing_model == dataset2.mixing_model, "Mixing models are not equal"


def test_dataset_consistent_storage(tmp_path):
    tmp_data_root = Path(tmp_path) / "data"
    tmp_data_root.mkdir()

    dataset_config = get_syn_dataset_config(seed=1958, num_systems=6)
    # try loading the same dataset again:
    dataset_1 = datasets.SyntheticPairedRotationsDataset.from_dict(
        dataset_config,
        kwargs={
            "SyntheticPairedRotationsDataset":
                dict(
                    root=tmp_data_root,
                    force_regenerate=True,
                )
        },
    )

    # try loading the same dataset again:
    dataset_2 = datasets.SyntheticPairedRotationsDataset.from_dict(
        dataset_config,
        kwargs={"SyntheticPairedRotationsDataset": dict(root=tmp_data_root,)},
    )

    assert_datasets_equal(dataset_1, dataset_2)
    assert_datasets_model_equal(dataset_1, dataset_2)


def test_dataset_consistent_generation(tmp_path):
    tmp_data_root = Path(tmp_path) / "data"
    tmp_data_root.mkdir()
    # clean up any existing data
    if tmp_data_root.exists():
        import shutil
        shutil.rmtree(tmp_data_root,)
    tmp_data_root.mkdir()

    dataset_config = get_syn_dataset_config(seed=123, num_systems=5)
    # try loading the same dataset again:
    dataset_1 = datasets.SyntheticPairedRotationsDataset.from_dict(
        dataset_config,
        kwargs={
            "SyntheticPairedRotationsDataset":
                dict(
                    root=tmp_data_root,
                    force_regenerate=True,
                )
        },
    )

    # this time we force regeneration to check the generation is consistent
    dataset_2 = datasets.SyntheticPairedRotationsDataset.from_dict(
        dataset_config,
        kwargs={
            "SyntheticPairedRotationsDataset":
                dict(
                    root=tmp_data_root,
                    force_regenerate=True,
                )
        },
    )

    assert_datasets_equal(dataset_1, dataset_2)
    assert_datasets_model_equal(dataset_1, dataset_2)


def _create_dataset(tmp_data_root, config_dict, event=None):
    # Wait for event if provided (used in staggered test)
    if event is not None:
        event.wait()

    # Create dataset from the same config dict
    dataset = datasets.SyntheticPairedRotationsDataset.from_dict(
        config_dict,
        kwargs={"SyntheticPairedRotationsDataset": dict(root=tmp_data_root,)})
    # Verify the dataset was created properly
    assert len(dataset) > 0
    return dataset


def _create_dataset_wrapper(tmp_data_root, config_dict, results):
    try:
        dataset = _create_dataset(tmp_data_root, config_dict)
        results.append(dataset)
    except Exception as e:
        print(f"Process failed with exception: {e}")
        # print full traceback
        traceback.print_exc()
        results.append(None)


def test_concurrent_dataset_generation(tmp_path):
    """Test creating multiple dataset instances concurrently.

    This test creates n dataset instances in parallel and verifies that
    they all initialize properly without race conditions.

    All instances use the EXACT SAME configuration, including the same seed,
    to test how the system handles concurrent access to the same dataset files.
    """

    num_processes = 20  # Number of concurrent dataset instantiations

    # Create a shared data directory
    tmp_data_root = Path(tmp_path) / "data"
    tmp_data_root.mkdir()

    # Get the configuration dictionary
    config_dict = get_syn_dataset_config(seed=864, num_systems=5)

    # Start multiple processes to create datasets concurrently
    processes = []
    results = multiprocessing.Manager().list()

    for i in range(num_processes):
        p = multiprocessing.Process(target=_create_dataset_wrapper,
                                    args=(tmp_data_root, config_dict, results))
        processes.append(p)
        p.start()

    # Wait for all processes to complete
    for p in processes:
        p.join()

    # Verify all processes successfully created their datasets
    assert len(results) == num_processes
    assert all(r is not None for r in results), "Some dataset creations failed"

    # Compare all datasets to ensure they are equal
    for i in range(1, len(results)):
        assert_datasets_equal(results[0], results[i])
        assert_datasets_model_equal(results[0], results[i])


def _create_dataset_with_result(tmp_data_root, config_dict, event, results):
    try:
        dataset = _create_dataset(tmp_data_root, config_dict, event)
        results.append(dataset)
    except Exception as e:
        print(f"Process failed with exception: {e}")
        results.append(None)


def test_staggered_dataset_generation(tmp_path):
    """Test creating multiple dataset instances with staggered starts.

    This test creates n dataset instances in parallel but starts each one
    with a small delay to test how the system handles near-concurrent access.

    All instances use the EXACT SAME configuration, including the same seed,
    to test how the system handles concurrent access to the same dataset files.
    """
    import multiprocessing
    import time

    num_processes = 15  # Number of dataset instantiations
    stagger_delay = 1  # Seconds between starts

    # Create a shared data directory
    tmp_data_root = Path(tmp_path) / "data"
    tmp_data_root.mkdir()

    # Get the configuration dictionary
    config_dict = get_syn_dataset_config(seed=346, num_systems=5)

    # Create events for staggered starts
    events = [multiprocessing.Event() for _ in range(num_processes)]
    processes = []
    results = multiprocessing.Manager().list()

    # Start multiple processes
    for i in range(num_processes):
        p = multiprocessing.Process(target=_create_dataset_with_result,
                                    args=(tmp_data_root, config_dict, events[i],
                                          results))
        processes.append(p)
        p.start()

    # Trigger the events with staggered delays
    for i, event in enumerate(events):
        time.sleep(stagger_delay)
        event.set()

    # Wait for all processes to complete
    for p in processes:
        p.join(timeout=60)  # Set a timeout for safety

    # Verify all processes completed
    for i, p in enumerate(processes):
        assert not p.is_alive(), f"Process {i} did not complete in time"

    # Verify all datasets were created successfully
    assert len(results) == num_processes
    assert all(d is not None for d in results), "Some dataset creations failed"

    # Compare all datasets against the first one
    reference_dataset = results[0]
    for i, dataset in enumerate(results[1:], 1):
        assert_datasets_equal(reference_dataset, dataset)
        assert_datasets_model_equal(reference_dataset, dataset)


if __name__ == "__main__":
    import pytest
    pytest.main([__file__])
