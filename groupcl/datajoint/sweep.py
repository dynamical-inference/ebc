from tqdm import tqdm

from groupcl.datajoint import schema
from groupcl.datajoint.utils import get_experiment_configs_for_sweep
from groupcl.experiments.experiments import Experiment
from typing import List


def flatten_dict(d, parent_key='', sep='__'):
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def unflatten_dict(d, sep='__'):
    result_dict = {}
    for k, v in d.items():
        parts = k.split(sep)
        d = result_dict
        for p in parts[:-1]:
            if p not in d:
                d[p] = {}
            d = d[p]
        d[parts[-1]] = v
    return result_dict


def reset_sweeps(existing_configs, class_sweep_name):
    print(
        f"Resetting sweep '{class_sweep_name}'... Deleting {len(existing_configs)} configs"
    )
    # before deleting, let's check if some of them were already trained
    existing_checkpoints = schema.ExperimentTable.Checkpoint(
    ) & existing_configs
    if len(existing_checkpoints) > 0:
        print(
            f"Warning: There exists {len(existing_checkpoints)} checkpoints associated with this sweep. "
        )
        confirmation = input(
            "Are you sure you want to delete these configs and their results? [y/N] "
        )
        if confirmation.lower() != 'y':
            print("Aborting sweep reset")
            return

    # sometimes, the existing_configs that are passed are not the correct ExperimentConfigTable instance (because of a join with another table)
    # so we get the arg_hashs from existing_configs and filter the ExperimentConfigTable explictly again
    arg_hashes = existing_configs.fetch("arg_hash")
    configs_to_delete = (schema.ExperimentConfigTable() &
                         [dict(arg_hash=h) for h in arg_hashes])
    configs_to_delete.delete()


def populate_exp_list(
    sweep_name: str,
    experiments: List[Experiment],
    reset_sweep: bool = False,
):

    existing_configs = get_experiment_configs_for_sweep(
        sweep_name=sweep_name,
        partial=False,
    )

    if reset_sweep:
        reset_sweeps(existing_configs, sweep_name)

    n_experiments_start = len(schema.ExperimentConfigTable())
    num_configs = len(experiments)
    print(
        f"Start populating {num_configs} configurations for sweep '{sweep_name}'"
    )
    print(
        f"Sweep '{sweep_name}' alredy contains {len(existing_configs)} configs")

    for experiment in tqdm(experiments, total=num_configs):

        schema.ExperimentConfigTable().insert_from_config(
            experiment.to_dict(),
            sweep_name=sweep_name,
            skip_duplicates=True,
        )

    n_experiments = len(schema.ExperimentConfigTable()) - n_experiments_start
    print(
        f"Finished populating {n_experiments} experiments for sweep '{sweep_name}'"
    )

    return n_experiments
