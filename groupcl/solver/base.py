import datetime
import json
import math
import random
import time
import traceback
from abc import ABC, abstractmethod
from collections import OrderedDict, defaultdict
from dataclasses import field, fields
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

import numpy as np
import torch
from config_dataclass import Configurable, config_dataclass, config_field, state_field
from jaxtyping import jaxtyped
from jaxtyping._typeguard import typechecked
from tqdm.auto import tqdm

from groupcl.loader.base import BaseDataLoader
from groupcl.metrics import EmbeddingMetric, GroupMetric, Metric
from groupcl.models.base import BaseModel
from groupcl.utils.checkpoints import CheckpointSavingCallback
from groupcl.utils.datatypes import (
    Batch,
    EmbeddingPrediction,
    GroupSolverPrediction,
    PairedGroupData,
    SinglePairedGroupData,
)


@config_dataclass
class BaseSolver(Configurable, ABC):
    """Base class for all solvers."""

    model: BaseModel = config_field(default_factory=BaseModel)
    # Training parameters
    num_epochs: Optional[int] = config_field(default=1)
    max_iterations: Optional[int] = config_field(default=None, skip_default=True)
    seed: int = config_field(default=42)

    # Best metric to monitor
    best_metric_key: Optional[str] = config_field(default=None, skip_default=True)
    best_metric_mode: Literal["min", "max"] = config_field(
        default="min", skip_default=True
    )

    # Version of the solver
    version: str = config_field(default="1", skip_default=True)

    # State
    logs: Dict[str, List[float]] = state_field(
        default_factory=lambda: defaultdict(list),
        repr=False,
    )
    current_step: int = state_field(default=0)
    current_epoch: int = state_field(default=0)
    last_best_metric: float = state_field(default=float("inf"))

    use_tqdm: bool = field(default=True, repr=False)
    silence_metric_errors: bool = field(default=True)
    skip_metrics: List[str] = field(default_factory=list, init=False, repr=False)

    def __post_init__(self):
        super().__post_init__()
        self.last_best_metric = (
            float("inf") if self.best_metric_mode == "min" else -float("inf")
        )

    def validate_config(self):
        super().validate_config()
        if self.num_epochs is not None and self.max_iterations is not None:
            raise ValueError(
                "num_epochs and max_iterations cannot be set at the same time"
            )
        if self.num_epochs is None and self.max_iterations is None:
            raise ValueError("num_epochs or max_iterations must be set")

    @classmethod
    def config_prefix(cls) -> str:
        """Prefix for the configuration file."""
        return "solver"

    def __lazy_post_init__(self):
        super().__lazy_post_init__()
        """Initialize solver."""
        self.reset()

    def reset(self):
        """Reset the solver."""
        self.logs = defaultdict(list)
        self.current_step = 0

    @property
    def state(self) -> Dict[str, Any]:
        """Get the current state of the solver."""
        return {
            f.name: getattr(self, f.name)
            for f in fields(self)
            if f.metadata.get("type") == "state"
        }

    def _save_state(self, path: Path) -> None:
        """Save the current state of the solver."""
        state_path = path / "solver_state.json"
        state_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(state_path, "w") as f:
                json.dump(self.state, f, indent=4)
        except Exception as e:
            traceback.print_exc()
            print(f"Error saving solver state: {e}")
            # write empty state
            with open(state_path, "w") as f:
                json.dump({}, f, indent=4)

    def _load_state(self, path: Path) -> bool:
        """Load the current state of the solver."""
        state_path = path / "solver_state.json"
        if not state_path.exists():
            return False
        with open(state_path, "r") as f:
            state = json.load(f)
            for f in fields(self):
                if f.name in state:
                    setattr(self, f.name, state[f.name])

    def _save_additional(self, path: Path) -> None:
        """Save additional state to disk."""
        self._save_state(path)

    def _load_additional(self, path: Path) -> bool:
        """Load additional state from disk."""
        return self._load_state(path)

    @abstractmethod
    def train_step(self, batch: Batch) -> Dict[str, Any]:
        """Perform a single training step.

        Args:
            batch: Dictionary containing the current batch of data

        Returns:
            Dictionary containing training metrics
        """

    @torch.no_grad()
    def validate(
        self,
        loader: BaseDataLoader,
        eval_batches: int = 1,
        metrics: List[Metric] = [],
    ) -> Dict[str, Any]:
        """Validate the model.

        Args:
            loader: data loader
            eval_batches: number of batches to evaluate loss metricson
        Returns:
            Dictionary containing metrics
        """

        self.to(loader.device)
        self.set_eval()

        loss_metrics = defaultdict(list)
        for i, batch in enumerate(loader):
            if i >= eval_batches:
                break
            losses = self.validate_step(batch=batch)
            for k, v in losses.items():
                loss_metrics[k].append(v)

        loss_metrics = {
            k: torch.mean(torch.tensor(v)).item() for k, v in loss_metrics.items()
        }

        global_metrics = self.compute_group_metrics(
            loader=loader,
            metrics=metrics,
        )
        embedding_metrics = self.compute_embedding_metrics(
            loader=loader,
            metrics=metrics,
        )
        return {**global_metrics, **loss_metrics, **embedding_metrics}

    @abstractmethod
    def predictions(
        self,
        loader: BaseDataLoader,
    ) -> GroupSolverPrediction:
        pass

    @jaxtyped(typechecker=typechecked)
    def compute_group_metrics(
        self,
        loader: BaseDataLoader,
        metrics: List[Metric] = [],
    ) -> Dict[str, Any]:
        """Compute global metrics."""
        metrics = [m for m in metrics if isinstance(m, GroupMetric)]
        if len(metrics) == 0:
            return {}
        predictions = self.predictions(loader)
        ground_truth = loader.ground_truth_data
        metric_results = {}
        for m in metrics:
            try:
                if m.name in self.skip_metrics:
                    continue
                print(f"Computing metric {m.name}", flush=True)
                metric_results[m.name] = m.compute(
                    predictions=predictions,
                    ground_truth=ground_truth,
                    solver=self,
                    loader=loader,
                )
            except Exception as e:
                # skiping metrics so we don't print the same error over and over during training
                self.skip_metrics.append(m.name)
                if self.silence_metric_errors:
                    print(f"Error computing metric {m.name}:")
                    print(e)
                    traceback.print_exc()
                else:
                    raise e
        return metric_results

    @jaxtyped(typechecker=typechecked)
    def compute_embedding_metrics(
        self,
        loader: BaseDataLoader,
        metrics: List[Metric] = [],
    ) -> Dict[str, Any]:
        """Compute embedding metrics."""
        metrics = [m for m in metrics if isinstance(m, EmbeddingMetric)]
        if len(metrics) == 0:
            return {}
        dataset = loader.dataset
        if hasattr(dataset, "get_observed_variable"):
            observed_data = dataset.get_observed_variable(dataset.index)
        elif hasattr(dataset, "get_observed_data"):
            observed_data = dataset.get_observed_data(dataset.index)
        else:
            raise ValueError(
                f"Dataset {dataset} does not have a get_observed_variable or get_observed_data method"
            )
        if isinstance(observed_data, (SinglePairedGroupData, PairedGroupData)):
            observed_data = observed_data.x
        elif isinstance(observed_data, torch.Tensor):
            pass
        else:
            raise ValueError(f"Unsupported observed data type: {type(observed_data)}")

        predictions = EmbeddingPrediction(
            x=self.embeddings(observed_data),
        )
        metric_results = {}
        for m in metrics:
            try:
                if m.name in self.skip_metrics:
                    continue
                print(f"Computing metric {m.name}", flush=True)
                metric_results[m.name] = m.compute(
                    predictions=predictions,
                    solver=self,
                    dataset=dataset,
                )
            except Exception as e:
                # skiping metrics so we don't print the same error over and over during training
                self.skip_metrics.append(m.name)
                if self.silence_metric_errors:
                    print(f"Error computing metric {m.name}:")
                    print(e)
                    traceback.print_exc()
                else:
                    raise e
        return metric_results

    @abstractmethod
    def validate_step(
        self, batch: Batch, gt_batch: Batch
    ) -> Tuple[Batch, Dict[str, Any]]:
        """Does a single prediction step, computes metrics for the batch and returns both the predictions and the metrics"""

    def evaluate(
        self,
        loader: BaseDataLoader,
        metrics: List[Metric] = [],
        **kwargs,
    ) -> Dict[str, Any]:
        """Evaluate the model after training is complete.
        This may be used to compute additional metrics, besides the one computed by validate.

        Args:
            loader: data loader

        Returns:
            Dictionary containing metrics
        """
        # by default, just call validate
        metrics = self.validate(loader, metrics=metrics, **kwargs)

        # for compatibility with datajoint all metrics need to be json serializable, which means they need to be regular dataytpes
        # iterate over metrics and convert any tensor or numpy arrays to lists
        def recursive_convert_to_list(obj):
            if isinstance(obj, torch.Tensor):
                return obj.tolist()
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: recursive_convert_to_list(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [recursive_convert_to_list(item) for item in obj]
            else:
                return obj

        for metric_name, metric_value in metrics.items():
            metrics[metric_name] = recursive_convert_to_list(metric_value)

        # now do a test conversion to json
        # try each metric separately and throw out any that fail
        del_metrics = []
        for metric_name, metric_value in metrics.items():
            try:
                json.dumps(metric_value)
            except Exception as e:
                print(f"Error converting metric {metric_name} to json: {e}", flush=True)
                traceback.print_exc()
                del_metrics.append(metric_name)
        for metric_name in del_metrics:
            del metrics[metric_name]

        return metrics

    def filter_tqdm_stats(self, stats: Dict[str, Any]) -> Dict[str, Any]:
        """Filter out keys that are not meant to be displayed in the progress bar."""
        return stats

    @property
    def priority_keys(self) -> List[str]:
        """Prioritiy keys for tqdm progress bar"""
        return ["epoch", "train_loss", "val_loss", "train_accuracy", "val_accuracy"]

    def update_tqdm_stats(
        self, old_postfix: Dict[str, Any], **kwargs
    ) -> Dict[str, Any]:
        """Update progress bar statistics by combining old and new data.

        Args:
            old_postfix: Previous postfix dictionary
            **kwargs: Keyword arguments to update postfix with. Special keys:
                - epoch: Current epoch number
                Other keys will be formatted with default formatting

        Returns:
            Updated postfix dictionary with formatted strings
        """
        postfix = old_postfix.copy()

        for k, v in kwargs.items():
            try:
                if isinstance(v, torch.Tensor):
                    kwargs[k] = v.item()
            except Exception as e:
                print(f"Error converting tensor for key {k}: {e}")

        # Handle epoch/num_epochs if present
        epoch = kwargs.pop("train_epoch", None)
        if epoch is not None:
            postfix["epoch"] = f"{epoch}/{self.num_epochs}"

        loss_keys = [k for k in kwargs.keys() if "loss" in k]
        for loss_key in loss_keys:
            loss_value = kwargs.pop(loss_key)
            postfix[loss_key] = f"{loss_value:.4e}"

        # Handle remaining kwargs with default formatting
        for k, v in kwargs.items():
            if isinstance(v, (int, float)):
                postfix[k] = f"{v:.4f}"
            else:
                postfix[k] = str(v)

        # Create ordered dict with priority keys first
        ordered_postfix = OrderedDict()
        for key in self.priority_keys:
            if key in postfix:
                ordered_postfix[key] = postfix[key]

        # Add remaining keys
        for key in postfix:
            if key not in self.priority_keys:
                ordered_postfix[key] = postfix[key]

        return ordered_postfix

    @jaxtyped(typechecker=typechecked)
    def fit(
        self,
        train_loader: BaseDataLoader,
        val_loader: Optional[BaseDataLoader] = None,
        save_hook: Optional[CheckpointSavingCallback] = None,
        eval_frequency: Optional[int] = None,
        eval_kwargs: Optional[Dict[str, Any]] = None,
    ):
        """Train the model.

        Args:
            train_loader: Training data loader
            val_loader: Optional validation data loader
            save_hook: Optional checkpoint saving callback
            eval_frequency: Optional evaluation frequency
            eval_batches: Optional number of batches to evaluate loss metrics on
        """
        eval_kwargs = eval_kwargs or {}
        print(f"Training on device: {train_loader.device}")
        self.to(train_loader.device)
        if val_loader is not None:
            val_loader.to(train_loader.device)

        print(f"Training with random seed: {self.seed}")
        # Set random seeds for reproducibility
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)

        if self.num_epochs is None:
            num_epochs = math.ceil(self.max_iterations / len(train_loader))
        else:
            num_epochs = self.num_epochs

        if self.max_iterations is None:
            total_iters = num_epochs * len(train_loader)
        else:
            total_iters = self.max_iterations

        # Single progress bar for all iterations
        progress_bar = (
            tqdm(
                initial=self.current_step,
                total=total_iters,
                desc="Training",
                unit="iter",
                # min and max control how often the pbar is updated (in seconds)
                miniters=200,
                maxinterval=float("inf"),
            )
            if self.use_tqdm
            else None
        )
        postfix = {}

        val_metrics = None
        done_training = False
        best_state_dict = self.state_dict
        for epoch in range(self.current_epoch, num_epochs):
            if done_training:
                break
            self.start_epoch(epoch)
            for batch in train_loader:
                self.set_train()
                done_training = self.current_step >= total_iters
                if done_training:
                    break
                # Training step
                metrics = self.train_step(batch)
                metrics["epoch"] = epoch
                self._log_metrics(metrics, prefix="train_")
                # Logging
                postfix = self.update_tqdm_stats(postfix, **metrics)

                # Validation
                if (
                    eval_frequency is not None
                    and val_loader is not None
                    and self.current_step % eval_frequency == 0
                ):
                    val_metrics = self.validate(val_loader, **eval_kwargs)
                    val_metrics["step"] = self.current_step
                    val_metrics["epoch"] = epoch
                    self._log_metrics(val_metrics, prefix="val_")
                    # Update progress bar with validation metrics
                    postfix = self.update_tqdm_stats(postfix, **val_metrics)

                if self.use_tqdm:
                    progress_bar.update(1)
                    progress_bar.set_postfix(**postfix, refresh=False)
                self.current_step += 1

                if save_hook is not None:
                    save_hook.maybe_save(
                        solver=self,
                        step=self.current_step,
                    )

            # check if we have a new best metric, only if there's a validation set
            if val_loader is not None and self.best_metric_key is not None:
                # recompute validation metrics
                best_eval_kwargs = dict(eval_batches=10)
                # if the metric to monitor is not in the validation metrics, check if it's in the metrics of eval_kwargs
                if "loss" not in self.best_metric_key:
                    available_metrics = [m.name for m in eval_kwargs["metrics"]]
                    best_eval_kwargs["metrics"] = [
                        m for m in available_metrics if m == self.best_metric_key
                    ]
                    assert len(best_eval_kwargs["metrics"]) == 1, (
                        "The best_metric_key must match exactly one metric"
                    )
                val_metrics = self.validate(val_loader, **best_eval_kwargs)
                new_best_metric = val_metrics[f"{self.best_metric_key}"]
                if (
                    self.best_metric_mode == "min"
                    and new_best_metric < self.last_best_metric
                ) or (
                    self.best_metric_mode == "max"
                    and new_best_metric > self.last_best_metric
                ):
                    self.last_best_metric = new_best_metric
                    # get state dict
                    best_state_dict = self.state_dict

            self.current_epoch += 1
            self.end_epoch(epoch)

        # final validation after training end
        if val_loader is not None and eval_frequency is not None:
            val_metrics = self.validate(val_loader, **eval_kwargs)
            val_metrics["step"] = self.current_step
            self._log_metrics(val_metrics, prefix="val_")
            postfix = self.update_tqdm_stats(postfix, **val_metrics)
        postfix = self.update_tqdm_stats(postfix, epoch=epoch + 1)
        if save_hook is not None:
            save_hook(
                solver=self,
                ckpt_dir_name="last_checkpoint",
            )
        if self.use_tqdm:
            progress_bar.close()

        if self.best_metric_key is not None:
            # load best state dict
            print(
                f"Loading best state dict from epoch={epoch}, step={self.current_step} with metric '{self.best_metric_key}'={self.last_best_metric:.4e}"
            )
            self.load_state_dict(best_state_dict)

    def start_epoch(self, epoch: int):
        """Hook for start of epoch."""
        # log start time stamp
        timestamp = time.time()
        self.logs["epoch_start_time"].append(timestamp)
        self.logs["epoch_start_time_str"].append(
            datetime.datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S")
        )

    def end_epoch(self, epoch: int):
        """Hook for end of epoch."""
        # log end time stamp
        timestamp = time.time()
        self.logs["epoch_end_time"].append(timestamp)
        self.logs["epoch_end_time_str"].append(
            datetime.datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S")
        )
        # log duration
        self.logs["epoch_duration"].append(
            timestamp - self.logs["epoch_start_time"][-1]
        )
        # log duration in hours, minutes, seconds
        duration = datetime.timedelta(seconds=self.logs["epoch_duration"][-1])
        self.logs["epoch_duration_str"].append(str(duration))

    def _log_metrics(self, metrics: Dict[str, Any], prefix: str = ""):
        """Log metrics to the logs dictionary.

        Args:
            metrics: Dictionary of metrics to log
            prefix: Optional prefix for metric names
        """
        for name, value in metrics.items():
            log_name = f"{prefix}{name}"
            self.logs[log_name].append(value)

    def to(self, device: torch.device):
        """Move the model to a specific device."""
        self.model.to(device)
        return self

    def set_eval(self):
        """Set the model to evaluation mode."""
        self.model.eval()
        return self

    def set_train(self):
        """Set the model to training mode."""
        self.model.train()

    @property
    def model_fields(self):
        return [f for f in fields(self) if isinstance(getattr(self, f.name), BaseModel)]

    @property
    def state_dict(self):
        state_dict = dict(
            current_epoch=self.current_epoch, current_step=self.current_step
        )
        for field in self.model_fields:
            state_dict[field.name] = getattr(self, field.name).state_dict()
        return state_dict

    def load_state_dict(self, state_dict: Dict[str, Any]):
        for field in self.model_fields:
            getattr(self, field.name).load_state_dict(state_dict[field.name])
