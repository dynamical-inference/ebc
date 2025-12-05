import argparse
import os
import traceback

from groupcl.datajoint import schema
from groupcl.datajoint.utils import get_sweep_partial
from groupcl.metrics.decoding import (
    DirectionKNNClassification,
    DirectionLogisticRegression,
    PositionKNNRegression,
    PositionLinearRegression,
    TimeIndexKNNRegression,
)
from groupcl.metrics.dynamics import LogisticRegressionClassification, NNActionAccuracy
from groupcl.metrics.identifiability import CCA, R2, GroupHomomorphismR2, VanillaR2

os.environ["JAXTYPING_DISABLE"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

train_metrics = [
    # R2(group="x", bias=False),
    # R2(group="x", true_subspace="group", bias=False),
    # R2(group="x", true_subspace="content", bias=False),
    # R2(group="x", true_subspace="group", pred_subspace="group", bias=False),
    # R2(group="x", true_subspace="content", pred_subspace="content", bias=False),
    # R2(group="x", true_subspace="shape"),
    # R2(group="x", true_subspace="scale"),
    # R2(group="x", true_subspace="orientation"),
    # R2(group="x", true_subspace="posX"),
    # R2(group="x", true_subspace="posY"),
    # R2(group="x_prime", bias=False),
    # R2(group="x_prime", true_subspace="group", bias=False),
    # R2(group="x_prime", true_subspace="content", bias=False),
    # R2(group="x_prime_pred", bias=False),
    # R2(group="x_prime_pred", true_subspace="group", bias=False),
    # R2(group="x_prime_pred", true_subspace="content", bias=False),
    # VanillaR2(group="x_prime_pred_emb"),
    # VanillaR2(group="x_prime_pred_emb",
    #           true_subspace="group",
    #           pred_subspace="group"),
    # VanillaR2(group="x_prime_pred_emb",
    #           true_subspace="content",
    #           pred_subspace="content"),
]
spaces = [
    (None, None),
    ("group", "group"),
    ("content", "content"),
    # ("group", "content"),
    # ("content", "group"),
    # ("shape", None),
    # ("scale", None),
    # ("orientation", None),
    # ("posX", None),
    # ("posY", None),
]
val_metrics = []
for group in ["x", "x_prime", "x_prime_pred"]:
    for bias in [True, False]:
        for hold_out_split in [
            True,
            # False,
        ]:
            for true, pred in spaces:
                for fitting_method in [
                    "lsqt",
                    "ortho-procrustes",
                ]:
                    if fitting_method == "ortho-procrustes" and bias:
                        continue
                    if fitting_method == "ortho-procrustes" and true != pred:
                        continue
                    val_metrics += [
                        R2(group=group, true_subspace=true, pred_subspace=pred, bias=bias, hold_out_split=hold_out_split, fitting_method=fitting_method),
                    ]

            if not bias and true == pred:
                val_metrics += [
                    CCA(
                        group=group,
                        true_subspace=true,
                        pred_subspace=pred,
                    ),
                ]

for true, pred in spaces:
    if true == pred:
        val_metrics += [
            VanillaR2(
                group="x_prime_pred_emb",
                true_subspace=true,
                pred_subspace=pred,
            ),
        ]

val_metrics += [
    GroupHomomorphismR2(num_samples=10_000),
]

# val_metrics += [
#     NNActionPrediction(batch_size=100,
#                        epochs=10,
#                        seed=0,
#                        action_names=[[latent_name]])
#     for latent_name in ["posX", "posY", "orientation"]
# ]
# val_metrics += [
#     NNActionPrediction(batch_size=100,
#                        epochs=10,
#                        seed=0,
#                        action_names=[["posX"], ["posY"], ["orientation"]])
# ]

# val_metrics += [
#     NNActionPrediction(batch_size=100,
#                        epochs=10,
#                        action_update_step_sizes=5,
#                        seed=0,
#                        action_names=[latent_combinations])
#     for latent_combinations in [
#         ["posX", "posY"],
#         ["posX", "orientation"],
#         ["posY", "orientation"],
#     ]
# ]

val_metrics += [
    # NNActionPrediction(
    #     action_update_step_sizes=5,
    #     batch_size=10,
    #     epochs=100,
    #     action_names=[
    #         ["orientation", "posX", "posY"],
    #     ],
    # )
]

# for k in [
#         1,
#         5,
#         10,
# ]:
#     val_metrics += [
#         KNNClassification(group="x", k=k),
#         KNNClassification(group="x_prime", k=k),
#         KNNClassification(group="x", k=k, pred_subspace="content"),
#         KNNClassification(group="x_prime", k=k, pred_subspace="content"),
#     ]

# val_metrics += [
#     KNNClassification(group="x", k=1, pred_subspace="group"),
#     KNNClassification(group="x_prime", k=1, pred_subspace="group"),
# ]

val_metrics += [
    LogisticRegressionClassification(group="x"),
    LogisticRegressionClassification(group="x_prime"),
]

val_metrics += [
    NNActionAccuracy(k=k, batch_size=8, max_samples=20_000)
    for k in [
        1,
        # 2,
        3,
        # 4,
        5,
        # 6,
        # 7,
        # 8,
        # 9,
        10,
    ]
]


partial_metrics = [
    # partial(
    #     LinearPositionPrediction,
    #     prediction_step_size=step_size,
    #     prediction_steps_ahead=steps_ahead,
    #     groupby=groupby,
    # )
    # for groupby in [None, "direction"]
    # for (step_size, steps_ahead) in [
    #     # (0.1, 5),
    #     (0.05, 10),
    #     # (0.01, 50),
    # ]
]
# reset val_metrics
val_metrics = []
for subspace in [None, "group", "content"]:
    for method in [
        PositionKNNRegression,
        PositionLinearRegression,
        DirectionKNNClassification,
        DirectionLogisticRegression,
        TimeIndexKNNRegression,
    ] + partial_metrics:
        metric = method(subspace=subspace)
        val_metrics.append(metric)
# check that all the names are unique
metric_names = [metric.name for metric in val_metrics]
if len(metric_names) != len(set(metric_names)):
    raise ValueError("Some metric names are not unique")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--suppress-errors", action="store_true", help="Ignore exceptions during populations.")
    parser.add_argument("--order", help="Job execution order", choices=["original", "reverse", "random"], default="random")
    parser.add_argument("--sweep", help="The sweep name to run", default=None, required=True)
    parser.add_argument("--eval_only", action="store_true", help="Only evaluate the models, don't train any new ones.", default=False, required=False)
    parser.add_argument("--limit", help="Max number of experiments to run, default is no limit", default=None, required=False)

    parser.add_argument("--all_checkpoint", action="store_true", help="Only evaluate the last checkpoint for each experiment", default=False, required=False)

    parser.add_argument("--skip_train", action="store_true", help="Skip training metric computation", default=False, required=False)
    parser.add_argument("--skip_val", action="store_true", help="Skip validation metric computation", default=False, required=False)
    parser.add_argument("--skip_test", action="store_true", help="Skip test metric computation", default=False, required=False)

    args = parser.parse_args()

    print(f"CUDA_VISIBLE_DEVICES={os.getenv('CUDA_VISIBLE_DEVICES')}")

    restriction = schema.ExperimentConfigTable()
    if args.sweep is not None:
        print(f"Filter for sweep name: {args.sweep}", flush=True)
        sweep_names = get_sweep_partial(args.sweep).proj(sweep="name")
        print(sweep_names, flush=True)
        sweep_restriction = schema.SweepExperimentTable().proj(arg_hash="experiment") & sweep_names
        restriction = restriction & sweep_restriction

    print(f"# Experiments to choose from: {len(restriction)}", flush=True)
    assert len(restriction) > 0
    max_calls = args.limit
    max_calls = int(max_calls) if max_calls is not None else None

    # max_calls in datajoint is implemented such that it doesn't take into account reserved jobs...
    # so we need to double check this ourselves and add it as part of the restrictions

    if not args.eval_only:
        errors = schema.ExperimentTable.populate(
            restriction,
            reserve_jobs=True,
            suppress_errors=args.suppress_errors,
            order=args.order,
            max_calls=max_calls,
            display_progress=True,
            make_kwargs=dict(
                disable_jaxtyping=True,
                save_step=10_000,
                eval_frequency=10_000,
                eval_kwargs=dict(eval_batches=10, metrics=train_metrics),
            ),
        )

        if errors is not None and len(errors) > 0:
            for error in errors:
                print(error, flush=True)
                traceback.print_exc()
            print("Showed errors", flush=True)
        else:
            print("Successfully populated Model", flush=True)

    # populate metrics
    print("Populating metrics", flush=True)
    metric_restriction = schema.ExperimentTable.Checkpoint() & restriction
    if not args.all_checkpoint:
        metric_restriction = metric_restriction & dict(checkpoint_name="last_checkpoint")

    print(f"Checkpoints to populate metrics for: {len(metric_restriction)}", flush=True)

    metrics_kwargs = dict(
        metrics=val_metrics,
        eval_batches=10,
    )

    if not args.skip_train:
        schema.TrainMetricsTable.populate(
            metric_restriction,
            reserve_jobs=True,
            suppress_errors=args.suppress_errors,
            order=args.order,
            display_progress=True,
            make_kwargs=metrics_kwargs,
        )

    if not args.skip_val:
        schema.ValMetricsTable.populate(
            metric_restriction,
            reserve_jobs=True,
            suppress_errors=args.suppress_errors,
            order=args.order,
            display_progress=True,
            make_kwargs=metrics_kwargs,
        )
    if not args.skip_test:
        schema.TestMetricsTable.populate(
            metric_restriction,
            reserve_jobs=True,
            suppress_errors=args.suppress_errors,
            order=args.order,
            display_progress=True,
            make_kwargs=metrics_kwargs,
        )
