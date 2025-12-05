import json
import os
from pathlib import Path
from typing import Any, Dict

import datajoint as dj
from dj_ml_core import fields as db_fields
from dj_ml_core.core import Schema
from dj_ml_core.git_check import GitChecker
from dj_ml_core.login import connect_to_database
from dj_ml_core.table import BaseConfigTable

from groupcl.experiments.experiments import Experiment
from groupcl.utils.checkpoints import DJCheckpointSavingCallback

env_vars = connect_to_database()

# get schema_name from env variable
schema_name = os.getenv("DATAJOINT_SCHEMA_NAME")
assert schema_name is not None, "environment variable DATAJOINT_SCHEMA_NAME is not set"

schema = Schema(schema_name, locals(), create_tables=True, create_schema=True)


@schema
class ModelConfigTable(BaseConfigTable):
    pass


@schema
class SolverConfigTable(BaseConfigTable):
    model = db_fields.ProjectedTableField(table=ModelConfigTable,
                                          attribute_name="model",
                                          key_name="arg_hash")


@schema
class DatasetConfigTable(BaseConfigTable):
    pass


@schema
class DataLoaderConfigTable(BaseConfigTable):
    pass


@schema
class ExperimentConfigTable(BaseConfigTable):

    dataset = db_fields.ProjectedTableField(table=DatasetConfigTable,
                                            attribute_name="dataset",
                                            key_name="arg_hash")
    solver = db_fields.ProjectedTableField(table=SolverConfigTable,
                                           attribute_name="solver",
                                           key_name="arg_hash")

    train_loader = db_fields.ProjectedTableField(table=DataLoaderConfigTable,
                                                 attribute_name="train_loader",
                                                 key_name="arg_hash")
    val_loader = db_fields.ProjectedTableField(table=DataLoaderConfigTable,
                                               attribute_name="val_loader",
                                               key_name="arg_hash")
    test_loader = db_fields.ProjectedTableField(table=DataLoaderConfigTable,
                                                attribute_name="test_loader",
                                                key_name="arg_hash")

    save_step = db_fields.IntField(default=1000)
    eval_frequency = db_fields.IntField(default=1000)

    def insert_from_config(self,
                           config_dict,
                           row_dict: Dict[str, Any] = {},
                           sweep_name: str = "default",
                           **kwargs):

        exp_arg_hash = super().insert_from_config(
            config_dict,
            row_dict=row_dict,
            **kwargs,
        )

        SweepExperimentTable().insert1(
            dict(sweep=sweep_name, experiment=exp_arg_hash),
            **kwargs,
        )

        return exp_arg_hash


@schema
class SweepTable(dj.Manual):
    name = db_fields.CharField(length=128, primary_key=True)


@schema
class SweepExperimentTable(dj.Manual):
    sweep = db_fields.ProjectedTableField(table=SweepTable,
                                          attribute_name="sweep",
                                          key_name="name",
                                          primary_key=True)
    experiment = db_fields.ProjectedTableField(table=ExperimentConfigTable,
                                               attribute_name="experiment",
                                               key_name="arg_hash",
                                               primary_key=True)

    def insert1(self, row, **kwargs):
        # automatically create the sweep if it doesn't exist
        SweepTable().insert1(
            dict(name=row["sweep"]),
            skip_duplicates=True,
        )
        super().insert1(row, **kwargs)


@schema
class ExperimentTable(dj.Computed):
    experiment_config = db_fields.TableField(table=ExperimentConfigTable,
                                             primary_key=True)

    logdir = db_fields.VarcharField(length=128)
    logs = db_fields.JSONField()
    code_version = db_fields.CharField(length=40)
    state = db_fields.CharField(length=32)

    class Checkpoint(dj.Part):
        definition = """
        -> ExperimentTable
        checkpoint_base_path : varchar(128)
        checkpoint_name : varchar(128)
        ---
        """

    def make(self,
             key,
             unsafe_hash=False,
             save_step=2_000,
             eval_frequency=None,
             **kwargs):
        print("\nMake Experiment for key\n", key, flush=True)
        print(f"With make_kwargs: {kwargs}\n", flush=True)
        try:
            # 1: get arguments necessary to initialize Experiment()
            config_dict = ExperimentConfigTable().get_config_dict(key)
            print(json.dumps(config_dict, indent=4), flush=True)

            # 2: initialize exp & get git hash
            print("Initalizing exp...", flush=True)
            exp = Experiment.from_dict(config_dict)
            print("Experiment intialized!", flush=True)
            repo_hash = GitChecker(
                unsafe_hash=unsafe_hash,
                verbose=True,
                include_patterns=["groupcl/"],
            ).check_repo_status()

            print(f"Repo hash: {repo_hash}", flush=True)
            print(f"Length of repo hash: {len(repo_hash)}", flush=True)

            # print which sweeps this experiment belongs to
            sweeps = (SweepExperimentTable().proj(arg_hash="experiment") & key)
            print(f"Experiment {key} present in sweeps:", flush=True)
            print(sweeps.fetch())

            def dj_callback(ckpt_path):
                return ExperimentTable.Checkpoint.insert1(
                    dict(
                        checkpoint_base_path=ckpt_path.parent,
                        checkpoint_name=ckpt_path.name,
                        **key,
                    ))

            dj_checkpoints = DJCheckpointSavingCallback(
                dj_callback=dj_callback,
                save_step=save_step,
            )

            # run the experiment
            if eval_frequency is None:
                eval_frequency = (ExperimentConfigTable() &
                                  key).fetch1()["eval_frequency"]
            print(
                f"Running experiment with eval frequency {eval_frequency} and save frequency {save_step}",
                flush=True)

            # To be able to populate the checkpoint table, the experiment table entry must already exist
            # so we already insert it here, and update it again later
            try:
                self.insert1(
                    dict(
                        **key,
                        logdir=exp.log_dir,
                        code_version=repo_hash,
                        logs=exp.solver.logs,
                        state="running",
                    ))
            except Exception:
                # try again without the logs
                self.insert1(
                    dict(
                        **key,
                        logdir=exp.log_dir,
                        code_version=repo_hash,
                        logs={},
                        state="running",
                    ))
            exp.run(
                save_hook=dj_checkpoints,
                eval_frequency=eval_frequency,
                **kwargs,
            )
            try:
                self.update1(
                    dict(
                        **key,
                        logs=exp.solver.logs,
                        state="finished",
                    ))
            except Exception:
                # try again without the logs
                self.update1(dict(
                    **key,
                    logs={},
                    state="finished",
                ))

        except Exception as e:
            import traceback
            traceback.print_exc()
            print(e)
            raise e


class MetricsTable(dj.Computed):

    checkpoint = db_fields.PartTableField(table=ExperimentTable.Checkpoint,
                                          main=ExperimentTable,
                                          primary_key=True)

    metrics_code_version = db_fields.CharField(length=40)

    results = db_fields.JSONField()

    def compute_metrics(self, exp: Experiment):
        pass

    def make(self, key, unsafe_hash=False, **kwargs):
        print(f"Make Metrics for key: {key}", flush=True)
        print(f"With make_kwargs: {kwargs}", flush=True)
        try:
            repo_hash = GitChecker(
                unsafe_hash=unsafe_hash,
                verbose=True,
                include_patterns=["groupcl/"],
            ).check_repo_status()

            logdir = (ExperimentTable & key).fetch1("logdir")
            ckpt_base_path = key["checkpoint_base_path"]
            ckpt_name = key["checkpoint_name"]
            ckpt_path = Path(ckpt_base_path) / ckpt_name
            exp = Experiment.load(logdir)
            exp.load_checkpoint(ckpt_path)
            metrics = self.compute_metrics(exp, **kwargs)
            self.insert1(
                dict(
                    **key,
                    metrics_code_version=repo_hash,
                    results=metrics,
                ))

        except Exception as e:
            import traceback
            traceback.print_exc()
            raise e


@schema
class TrainMetricsTable(MetricsTable):

    def compute_metrics(self, exp: Experiment, **kwargs):
        return exp.evaluate_train(**kwargs)


@schema
class ValMetricsTable(MetricsTable):

    def compute_metrics(self, exp: Experiment, **kwargs):
        return exp.evaluate_val(**kwargs)


@schema
class TestMetricsTable(MetricsTable):

    def compute_metrics(self, exp: Experiment, **kwargs):
        return exp.evaluate_test(**kwargs)
