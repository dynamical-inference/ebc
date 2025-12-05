from dataclasses import dataclass
from typing import Any, Dict, Tuple

import torch
from jaxtyping import Float
from jaxtyping import jaxtyped
from torch import Tensor
from jaxtyping._typeguard import typechecked

from groupcl.loader.dynamics import GroupDataLoader
from groupcl.models.dynamics.slds import GumbelSLDS
from groupcl.solver.base import BaseSolver
from groupcl.solver.optimizer import Optimizer
from groupcl.solver.optimizer import AdamOptimizer
from groupcl.utils.datatypes import PairedGroupData
from groupcl.utils.datatypes import GumbelPairedPredictions
from groupcl.utils.datatypes import GroupSolverPrediction

from groupcl.utils.temperature_scheduler import ConstantTemperatureScheduler
from groupcl.utils.temperature_scheduler import TemperatureScheduler
from groupcl.criterions.dynamics import MSECriterion
from config_dataclass import config_dataclass, config_field, state_field


@jaxtyped(typechecker=typechecked)
@config_dataclass
class GroupDynamicsSolver(BaseSolver):
    """Solver for training SLDS models using Gumbel-Softmax relaxation."""

    # Model
    model: GumbelSLDS = config_field(default_factory=GumbelSLDS)
    num_gumbel_samples: int = config_field(default=1)

    # Temperature scheduler
    temperature_scheduler: TemperatureScheduler = config_field(
        default_factory=ConstantTemperatureScheduler)

    # Optimizer
    optimizer: Optimizer = config_field(default_factory=AdamOptimizer)

    # Criterion
    criterion: MSECriterion = config_field(default_factory=MSECriterion)

    # state
    current_temp: float = state_field(default=0.0)

    @property
    def dynamics_model(self) -> GumbelSLDS:
        return self.model

    def __lazy_post_init__(self):
        """Initialize solver with optimizer and temperature."""
        super().__lazy_post_init__()
        self.optimizer.lazy_init(self.model.parameters())
        self.current_temp = self.temperature_scheduler.initial_temp
        self.model.tau = self.current_temp

    def reset(self):
        """Reset the solver."""
        super().reset()
        self.current_temp = self.temperature_scheduler.initial_temp
        self.model.tau = self.current_temp

    def _update_temperature(self, epoch: int):
        """Update Gumbel-Softmax temperature using scheduler."""
        self.current_temp = self.temperature_scheduler.get_temperature(
            epoch=epoch)
        self.model.tau = self.current_temp

    @jaxtyped(typechecker=typechecked)
    def _compute_loss(
        self,
        x_prime_pred: Float[Tensor, "batch num_samples dim"],
        y_prime_pred: Float[Tensor, "batch num_samples dim"],
        batch: PairedGroupData,
    ) -> Tuple[Float[Tensor, ""], Dict[str, float]]:
        """Computes the MSE loss between predictions and targets.

        Flattens the prediction sample dimension and repeats targets before comparison.
        """
        batch_size, num_samples, dim = x_prime_pred.shape

        # Flatten predictions: (batch, num_samples, dim) -> (batch * num_samples, dim)
        x_prime_pred_flat = x_prime_pred.view(-1, dim)
        y_prime_pred_flat = y_prime_pred.view(-1, dim)

        # Repeat targets to match flattened predictions
        x_prime_target_rep = batch.x_prime.repeat_interleave(
            repeats=num_samples,
            dim=0,
        )
        y_prime_target_rep = batch.y_prime.repeat_interleave(
            repeats=num_samples,
            dim=0,
        )

        # Compute MSE loss using the criterion
        loss_x = self.criterion(input=x_prime_pred_flat,
                                target=x_prime_target_rep)
        loss_y = self.criterion(input=y_prime_pred_flat,
                                target=y_prime_target_rep)
        loss = (loss_x + loss_y) / 2

        return loss, {"loss_x": loss_x.item(), "loss_y": loss_y.item()}

    @jaxtyped(typechecker=typechecked)
    def train_step(self, batch: PairedGroupData) -> Dict[str, float]:
        """Perform single training step using Gumbel-Softmax relaxation.

        Args:
            batch: PairedGroupData containing original and transformed pairs (x, y) and (x_prime, y_prime).

        Returns:
            Dictionary containing training metrics
        """
        self.model.train()
        self.optimizer.zero_grad()

        # Forward pass
        # x_prime_pred/y_prime_pred shape: (batch, num_samples, dim)
        # selected_modes shape: (batch, num_samples, num_modes)
        x_prime_pred, y_prime_pred, selected_modes = self.model(
            x=batch.x,
            y=batch.y,
            x_prime=batch.x_prime,
            y_prime=batch.y_prime,
            num_samples=self.num_gumbel_samples,
        )

        # Compute losses
        loss, loss_metrics = self._compute_loss(x_prime_pred, y_prime_pred,
                                                batch)

        # Backward pass
        loss.backward()
        self.optimizer.step()
        return {
            **loss_metrics,
            "loss": loss.item(),
            "temperature": self.current_temp,
        }

    def start_epoch(self, epoch: int):
        """At start of epoch, update temperature."""
        super().start_epoch(epoch=epoch)
        self._update_temperature(epoch=epoch)

    @jaxtyped(typechecker=typechecked)
    def validate_step(self, batch: PairedGroupData) -> Dict[str, float]:
        """Perform a single validation step.

        Args:
            batch: PairedGroupData containing the current batch of data

        Returns:
            Dictionary containing validation metrics
        """
        # Forward pass
        # x_prime_pred/y_prime_pred shape: (batch, 1, dim)
        # selected_modes shape: (batch, 1, num_modes)
        x_prime_pred, y_prime_pred, selected_modes = self.model(
            x=batch.x,
            y=batch.y,
            x_prime=batch.x_prime,
            y_prime=batch.y_prime,
            num_gumbel_samples=1,
        )

        # Compute losses
        loss, loss_metrics = self._compute_loss(x_prime_pred, y_prime_pred,
                                                batch)

        return {
            **loss_metrics,
            "loss": loss.item(),
        }

    @jaxtyped(typechecker=typechecked)
    def predictions(self, loader: GroupDataLoader) -> GroupSolverPrediction:
        """Perform predictions for metrics using validation data.

        Args:
            loader: GroupDataLoader containing validation data

        Returns:
            GroupSolverPrediction containing predictions and embeddings
        """
        self.set_eval()
        val_data = loader.validation_data

        # Forward pass with num_samples=1 for validation/prediction
        x_prime_pred, y_prime_pred, mode_samples = self.model(
            x=val_data.x,
            y=val_data.y,
            x_prime=val_data.x_prime,
            y_prime=val_data.y_prime,
            num_samples=1,  # Use 1 sample for deterministic prediction
        )

        dynamics_preds = GumbelPairedPredictions(x_prime=x_prime_pred,
                                                 y_prime=y_prime_pred,
                                                 mode_samples=mode_samples)

        # Note: GumbelDynamicsSolver doesn't have an encoder, so 'embeddings'
        # here will just contain the original input data from the validation set.
        return GroupSolverPrediction(embeddings=val_data,
                                     dynamics=dynamics_preds)
