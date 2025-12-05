from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union, Literal

import torch
from jaxtyping import Float
from jaxtyping import Integer
from jaxtyping import jaxtyped
from torch import Tensor
from jaxtyping._typeguard import typechecked

from groupcl.loader.contrastive import GroupContrastiveDataLoader, GCLDataLoader
from groupcl.models.dynamics.slds import GumbelSLDS
from groupcl.models.dynamics.bivector_dynamics import BivectorDynamicsModel
from groupcl.models.dynamics.utils import freeze_model
from groupcl.models.groups import GroupDynamicsModel
from groupcl.models.encoder import EncoderModel
from groupcl.models.encoder import MLP
from groupcl.solver.base import BaseSolver
from groupcl.solver.optimizer import DynCLAdamOptimizer
from groupcl.solver.optimizer import AdamOptimizer
from groupcl.solver.optimizer import Optimizer
from groupcl.utils.temperature_scheduler import ConstantTemperatureScheduler
from groupcl.utils.temperature_scheduler import TemperatureScheduler
from groupcl.models.dynamics.identitiy import IdentitySLDSModel
from groupcl.utils import datatypes as dt
from groupcl import criterions
from config_dataclass import config_dataclass, config_field, state_field


@jaxtyped(typechecker=typechecked)
@config_dataclass
class GroupContrastiveLearningSolver(BaseSolver):
    """Solver for training encoder model jointly with group dynamics model using contrastive learning."""

    # bump version, after fixing bug in loss computation (ref and pos were swapped)
    version: str = config_field(default="2", skip_default=False)

    # Models
    # encoder model
    model: EncoderModel = config_field(default_factory=MLP)
    dynamics_model: Union[GumbelSLDS, IdentitySLDSModel] = config_field(
        default_factory=GumbelSLDS)
    num_gumbel_samples: int = config_field(default=1)

    # Temperature scheduler
    temperature_scheduler: TemperatureScheduler = config_field(
        default_factory=ConstantTemperatureScheduler)

    # Optimizer
    optimizer: Optimizer = config_field(default_factory=DynCLAdamOptimizer)

    # Criterion
    criterion: criterions.ContrastiveCriterion = config_field(
        default_factory=criterions.MseInfoNCE)

    # state
    current_epoch: int = state_field(default=0)
    current_temp: float = state_field(default=0.0)

    # Whether to freeze the dynamics model
    freeze_dynamics_model: bool = config_field(default=False)

    @property
    def dynamics_model_offset(self) -> int:
        return self.dynamics_model.time_offset

    def __lazy_post_init__(self):
        """Initialize solver with optimizer and temperature."""
        super().__lazy_post_init__()
        self.optimizer.lazy_init(
            parameters=dict(encoder_model=self.model.parameters(),
                            dynamics_model=self.dynamics_model.parameters()))
        self.reset()

        if self.freeze_dynamics_model:
            print("Freezing dynamics model for training.")
            freeze_model(self.dynamics_model)

    def reset(self):
        """Reset the solver."""
        super().reset()
        self.current_temp = self.temperature_scheduler.initial_temp
        self.dynamics_model.tau = self.current_temp

    def _update_temperature(self, epoch: int):
        """Update Gumbel-Softmax temperature using scheduler."""
        self.current_temp = self.temperature_scheduler.get_temperature(
            epoch=epoch)
        self.dynamics_model.tau = self.current_temp

    def start_epoch(self, epoch: int):
        """At start of epoch, update temperature."""
        super().start_epoch(epoch=epoch)
        self._update_temperature(epoch=epoch)

    @jaxtyped(typechecker=typechecked)
    def train_step(self, batch: dt.ContrastiveGroupBatch) -> Dict[str, Any]:
        """Perform single training step using contrastive learning.

        Args:
            batch: dt.ContrastiveGroupBatch

        Returns:
            Dictionary containing training metrics
        """
        self.set_train()
        self.optimizer.zero_grad()

        embedding_batch = self.predict_step(batch)

        loss, loss_metrics = self.compute_loss(embedding_batch)

        # Backward pass
        loss.backward()
        self.optimizer.step()

        return dict(**loss_metrics,
                    loss=loss.item(),
                    temperature=self.current_temp)

    def predict_step(self, batch: dt.ContrastiveGroupBatch):
        """Takes an input batch and returns a batch of embeddings with predictions."""
        embedding_batch = self.encode_batch(batch)
        dynamics_prediction = self.dynamics_step(embedding_batch)
        return self.to_contrastive_loss_batch(
            embedding_batch=embedding_batch,
            dynamics_prediction=dynamics_prediction,
        )

    @jaxtyped(typechecker=typechecked)
    def to_contrastive_loss_batch(
        self,
        embedding_batch: dt.ContrastiveGroupBatch,
        dynamics_prediction: dt.GumbelPairedPredictions,
    ) -> dt.PairedContrastiveLossBatch:

        # Flatten first two dimensions of predictions (batch, gumbel_samples, dim) -> (batch*gumbel_samples, dim)
        x_prime_pred = dynamics_prediction.x_prime.view(-1,
                                                        self.dynamics_model.dim)
        y_prime_pred = dynamics_prediction.y_prime.view(-1,
                                                        self.dynamics_model.dim)

        # Repeat transformed (_prime) samples to match flattened predictions
        x_prime_repeated = embedding_batch.positives.x_prime.repeat_interleave(
            repeats=dynamics_prediction.gumbel_samples,
            dim=0,
        )
        y_prime_repeated = embedding_batch.positives.y_prime.repeat_interleave(
            repeats=dynamics_prediction.gumbel_samples,
            dim=0,
        )

        # Create contrastive loss batch
        x = dt.ContrastiveLossBatch(
            reference=x_prime_pred,
            positive=x_prime_repeated,
            negative=embedding_batch.negatives.x,
        )
        y = dt.ContrastiveLossBatch(
            reference=y_prime_pred,
            positive=y_prime_repeated,
            negative=embedding_batch.negatives.y,
        )

        return dt.PairedContrastiveLossBatch(x=x, y=y)

    def encode_batch(
            self, batch: dt.ContrastiveGroupBatch) -> dt.ContrastiveGroupBatch:
        """Encode the batch into embeddings."""

        # positive pair embeddings
        # contrastive_group_batch = dt.ContrastiveGroupBatch(
        #     positives=dt.PairedGroupData(
        #         x=self.model(batch.positives.x),
        #         y=self.model(batch.positives.y),
        #         x_prime=self.model(batch.positives.x_prime),
        #         y_prime=self.model(batch.positives.y_prime),
        #         indices=batch.positives.indices,
        #     ),
        #     negatives=dt.PairedData(
        #         x=self.model(batch.negatives.x),
        #         y=self.model(batch.negatives.y),
        #         indices=batch.negatives.indices,
        #     ),
        # )

        # To improve performance, we should instead concatenate all
        # of these tensors and then pass them through the model in one forward pass
        concatenated_data = torch.cat([
            batch.positives.x,
            batch.positives.y,
            batch.positives.x_prime,
            batch.positives.y_prime,
            batch.negatives.x,
            batch.negatives.y,
        ],
                                      dim=0)
        embeddings = self.embeddings(concatenated_data)

        # define the slices to split the embeddings into contrastive_group_batch
        pos_x_slice = slice(0, batch.positives.x.shape[0])
        pos_y_slice = slice(
            pos_x_slice.stop,
            pos_x_slice.stop + batch.positives.y.shape[0],
        )
        pos_x_prime_slice = slice(
            pos_y_slice.stop,
            pos_y_slice.stop + batch.positives.x_prime.shape[0],
        )
        pos_y_prime_slice = slice(
            pos_x_prime_slice.stop,
            pos_x_prime_slice.stop + batch.positives.y_prime.shape[0],
        )
        neg_x_slice = slice(
            pos_y_prime_slice.stop,
            pos_y_prime_slice.stop + batch.negatives.x.shape[0],
        )
        neg_y_slice = slice(
            neg_x_slice.stop,
            neg_x_slice.stop + batch.negatives.y.shape[0],
        )

        # create contrastive_group_batch
        embedding_batch = dt.ContrastiveGroupBatch(
            positives=dt.PairedGroupData(
                x=embeddings[pos_x_slice],
                y=embeddings[pos_y_slice],
                x_prime=embeddings[pos_x_prime_slice],
                y_prime=embeddings[pos_y_prime_slice],
                indices=batch.positives.indices,
            ),
            negatives=dt.PairedData(
                x=embeddings[neg_x_slice],
                y=embeddings[neg_y_slice],
                indices=batch.negatives.indices,
            ),
        )
        return embedding_batch

    def embeddings(
        self,
        batch: Float[Tensor, "batch obs_dim"],
    ) -> Float[Tensor, "batch latent_dim"]:
        """Encode the batch into embeddings."""
        return self.model(batch)

    @jaxtyped(typechecker=typechecked)
    def dynamics_step(
        self,
        batch: dt.ContrastiveGroupBatch,
    ) -> dt.GumbelPairedPredictions:
        """Perform a single dynamics step."""

        x_prime_pred, y_prime_pred, mode_samples = self.dynamics_model(
            x=batch.positives.x,
            y=batch.positives.y,
            x_prime=batch.positives.x_prime,
            y_prime=batch.positives.y_prime,
            num_samples=self.num_gumbel_samples)
        return dt.GumbelPairedPredictions(x_prime=x_prime_pred,
                                          y_prime=y_prime_pred,
                                          mode_samples=mode_samples)

    def _compute_individual_loss(
        self,
        batch: dt.ContrastiveLossBatch,
    ) -> Tuple[Float[Tensor, ""], Dict[str, Any]]:
        """Compute loss for contrastive learning."""
        total, align, uniformity = self.criterion(batch)
        return total, dict(
            loss_align=align.item(),
            loss_uniformity=uniformity.item(),
            loss_total=total.item(),
        )

    @jaxtyped(typechecker=typechecked)
    def compute_loss(
        self, batch: dt.PairedContrastiveLossBatch
    ) -> Tuple[Float[Tensor, ""], Dict[str, Any]]:
        """Compute loss for contrastive learning."""
        x_loss, x_loss_metrics = self._compute_individual_loss(batch.x)
        y_loss, y_loss_metrics = self._compute_individual_loss(batch.y)
        loss = (x_loss + y_loss) / 2
        return loss, {
            **{
                f"{k}_x": v for k, v in x_loss_metrics.items()
            },
            **{
                f"{k}_y": v for k, v in y_loss_metrics.items()
            },
        }

    @jaxtyped(typechecker=typechecked)
    def filter_tqdm_stats(self, stats: Dict[str, Any]) -> Dict[str, Any]:
        """Filter out keys that are not meant to be displayed in the progress bar."""
        key_to_keep = [
            "loss",
            "loss_mse_x",
            "loss_align_x",
            "loss_align_y",
            "loss_uniformity_x",
            "loss_uniformity_y",
        ]
        return {k: v for k, v in stats.items() if k not in key_to_keep}

    @jaxtyped(typechecker=typechecked)
    def validate_step(
        self,
        batch: dt.ContrastiveGroupBatch,
    ) -> Dict[str, Any]:
        """Does a single prediction step, computes metrics for the batch and returns both the predictions and the metrics"""

        self.set_eval()
        embedding_batch = self.predict_step(batch)
        loss, loss_metrics = self.compute_loss(embedding_batch)

        return dict(**loss_metrics, loss=loss.item())

    @jaxtyped(typechecker=typechecked)
    def predictions(
        self,
        loader: GroupContrastiveDataLoader,
    ) -> dt.GroupSolverPrediction:
        """Perform predictions for metrics."""

        val_data = loader.validation_data

        embeddings = dt.PairedGroupData(
            x=self.embeddings(val_data.x),
            y=self.embeddings(val_data.y),
            x_prime=self.embeddings(val_data.x_prime),
            y_prime=self.embeddings(val_data.y_prime),
            indices=val_data.indices,
        )
        x_prime_pred, y_prime_pred, mode_samples = self.dynamics_model(
            x=embeddings.x,
            y=embeddings.y,
            x_prime=embeddings.x_prime,
            y_prime=embeddings.y_prime,
            num_samples=1,
        )
        dynamics = dt.GumbelPairedPredictions(x_prime=x_prime_pred,
                                              y_prime=y_prime_pred,
                                              mode_samples=mode_samples)
        return dt.GroupSolverPrediction(embeddings=embeddings,
                                        dynamics=dynamics)

    def to(self, device: torch.device):
        """Move the model to a specific device."""
        super().to(device)
        self.dynamics_model.to(device)
        return self

    def set_eval(self):
        """Set the model to evaluation mode."""
        super().set_eval()
        self.dynamics_model.eval()
        return self

    def set_train(self):
        """Set the model to training mode."""
        super().set_train()
        self.dynamics_model.train()
        return self


@jaxtyped(typechecker=typechecked)
@config_dataclass
class GroupCLSolverCustomCriterion(GroupContrastiveLearningSolver):
    """
    Variante of the GroupCLSolver that allows us to move more complexity into the criterion class.
    This makes it easier to experiment with different criteria    
    """

    criterion: criterions.GroupCLCriterion = config_field(
        default_factory=criterions.GroupCLMseInfoNCE)

    @jaxtyped(typechecker=typechecked)
    def to_contrastive_loss_batch(
        self,
        embedding_batch: dt.ContrastiveGroupBatch,
        dynamics_prediction: dt.GumbelPairedPredictions,
    ) -> dt.GroupSolverContrastivePrediction:
        return dt.GroupSolverContrastivePrediction(
            embeddings=embedding_batch,
            dynamics=dynamics_prediction,
        )

    @jaxtyped(typechecker=typechecked)
    def compute_loss(
        self, batch: dt.GroupSolverContrastivePrediction
    ) -> Tuple[Float[Tensor, ""], Dict[str, Any]]:
        """Compute loss for contrastive learning."""
        loss, loss_metrics = self.criterion(batch)
        return loss, loss_metrics


@jaxtyped(typechecker=typechecked)
@config_dataclass
class BivectorCLSolver(BaseSolver):

    # Models
    # encoder model
    model: EncoderModel = config_field(default_factory=MLP)
    bivector_model: BivectorDynamicsModel = config_field(
        default_factory=BivectorDynamicsModel)

    # Optimizer
    optimizer: Optimizer = config_field(default_factory=AdamOptimizer)

    # Criterion
    criterion: criterions.ContrastiveCriterion = config_field(
        default_factory=criterions.MseInfoNCE)

    # state
    current_epoch: int = state_field(default=0)
    current_temp: float = state_field(default=0.0)

    def __lazy_post_init__(self):
        """Initialize solver with optimizer and temperature."""
        super().__lazy_post_init__()
        self.optimizer.lazy_init(parameters=self.model.parameters())
        self.reset()

    @jaxtyped(typechecker=typechecked)
    def train_step(self, batch: dt.ContrastiveGroupBatch) -> Dict[str, Any]:
        """Perform single training step using contrastive learning.

        Args:
            batch: dt.ContrastiveGroupBatch

        Returns:
            Dictionary containing training metrics
        """
        self.set_train()
        self.optimizer.zero_grad()

        embedding_batch = self.predict_step(batch)

        loss, loss_metrics = self.compute_loss(embedding_batch)

        # Backward pass
        loss.backward()
        self.optimizer.step()

        return dict(**loss_metrics,
                    loss=loss.item(),
                    temperature=self.current_temp)

    def predict_step(self, batch: dt.ContrastiveGroupBatch):
        """Takes an input batch and returns a batch of embeddings with predictions."""
        embedding_batch = self.encode_batch(batch)
        return self.to_contrastive_loss_batch(embedding_batch=embedding_batch)

    @jaxtyped(typechecker=typechecked)
    def to_contrastive_loss_batch(
        self,
        embedding_batch: dt.ContrastiveGroupBatch,
    ) -> dt.PairedContrastiveLossBatch:

        yp, yp_prime, y_prime_pred = self.bivector_model(
            x=embedding_batch.positives.x,
            y=embedding_batch.positives.y,
            x_prime=embedding_batch.positives.x_prime,
            y_prime=embedding_batch.positives.y_prime,
        )

        # swap x and y in model inputs to compute other direction
        xp, xp_prime, x_prime_pred = self.bivector_model(
            x=embedding_batch.positives.y,
            y=embedding_batch.positives.x,
            x_prime=embedding_batch.positives.y_prime,
            y_prime=embedding_batch.positives.x_prime,
            y_negs=embedding_batch.negatives.x,
        )

        return_dtype = torch.float32

        # Create contrastive loss batch
        x = dt.ContrastiveLossBatch(
            reference=x_prime_pred.to(dtype=return_dtype),
            positive=xp_prime.to(dtype=return_dtype),
            negative=embedding_batch.negatives.x.to(dtype=return_dtype),
        )
        y = dt.ContrastiveLossBatch(
            reference=y_prime_pred.to(dtype=return_dtype),
            positive=yp_prime.to(dtype=return_dtype),
            negative=embedding_batch.negatives.y.to(dtype=return_dtype),
        )

        return dt.PairedContrastiveLossBatch(x=x, y=y)

    def encode_batch(
            self, batch: dt.ContrastiveGroupBatch) -> dt.ContrastiveGroupBatch:
        """Encode the batch into embeddings."""

        # To improve performance, we should instead concatenate all
        # of these tensors and then pass them through the model in one forward pass
        concatenated_data = torch.cat([
            batch.positives.x,
            batch.positives.y,
            batch.positives.x_prime,
            batch.positives.y_prime,
            batch.negatives.x,
            batch.negatives.y,
        ],
                                      dim=0)
        embeddings = self.embeddings(concatenated_data)

        # define the slices to split the embeddings into contrastive_group_batch
        pos_x_slice = slice(0, batch.positives.x.shape[0])
        pos_y_slice = slice(
            pos_x_slice.stop,
            pos_x_slice.stop + batch.positives.y.shape[0],
        )
        pos_x_prime_slice = slice(
            pos_y_slice.stop,
            pos_y_slice.stop + batch.positives.x_prime.shape[0],
        )
        pos_y_prime_slice = slice(
            pos_x_prime_slice.stop,
            pos_x_prime_slice.stop + batch.positives.y_prime.shape[0],
        )
        neg_x_slice = slice(
            pos_y_prime_slice.stop,
            pos_y_prime_slice.stop + batch.negatives.x.shape[0],
        )
        neg_y_slice = slice(
            neg_x_slice.stop,
            neg_x_slice.stop + batch.negatives.y.shape[0],
        )

        # create contrastive_group_batch
        embedding_batch = dt.ContrastiveGroupBatch(
            positives=dt.PairedGroupData(
                x=embeddings[pos_x_slice],
                y=embeddings[pos_y_slice],
                x_prime=embeddings[pos_x_prime_slice],
                y_prime=embeddings[pos_y_prime_slice],
                indices=batch.positives.indices,
            ),
            negatives=dt.PairedData(
                x=embeddings[neg_x_slice],
                y=embeddings[neg_y_slice],
                indices=batch.negatives.indices,
            ),
        )
        return embedding_batch

    def embeddings(
        self,
        batch: Float[Tensor, "batch obs_dim"],
    ) -> Float[Tensor, "batch latent_dim"]:
        """Encode the batch into embeddings."""
        return self.model(batch)

    def _compute_individual_loss(
        self,
        batch: dt.ContrastiveLossBatch,
    ) -> Tuple[Float[Tensor, ""], Dict[str, Any]]:
        """Compute loss for contrastive learning."""
        total, align, uniformity = self.criterion(batch)
        return total, dict(
            loss_align=align.item(),
            loss_uniformity=uniformity.item(),
            loss_total=total.item(),
        )

    @jaxtyped(typechecker=typechecked)
    def compute_loss(
        self, batch: dt.PairedContrastiveLossBatch
    ) -> Tuple[Float[Tensor, ""], Dict[str, Any]]:
        """Compute loss for contrastive learning."""
        x_loss, x_loss_metrics = self._compute_individual_loss(batch.x)
        y_loss, y_loss_metrics = self._compute_individual_loss(batch.y)
        loss = (x_loss + y_loss) / 2
        return loss, {
            **{
                f"{k}_x": v for k, v in x_loss_metrics.items()
            },
            **{
                f"{k}_y": v for k, v in y_loss_metrics.items()
            },
        }

    @jaxtyped(typechecker=typechecked)
    def filter_tqdm_stats(self, stats: Dict[str, Any]) -> Dict[str, Any]:
        """Filter out keys that are not meant to be displayed in the progress bar."""
        key_to_keep = [
            "loss",
            "loss_mse_x",
            "loss_align_x",
            "loss_align_y",
            "loss_uniformity_x",
            "loss_uniformity_y",
        ]
        return {k: v for k, v in stats.items() if k not in key_to_keep}

    @jaxtyped(typechecker=typechecked)
    def validate_step(
        self,
        batch: dt.ContrastiveGroupBatch,
    ) -> Dict[str, Any]:
        """Does a single prediction step, computes metrics for the batch and returns both the predictions and the metrics"""

        self.set_eval()
        embedding_batch = self.predict_step(batch)
        loss, loss_metrics = self.compute_loss(embedding_batch)

        return dict(**loss_metrics, loss=loss.item())

    @jaxtyped(typechecker=typechecked)
    def predictions(
        self,
        loader: GroupContrastiveDataLoader,
    ) -> dt.GroupSolverPrediction:
        """Perform predictions for metrics."""

        val_data = loader.validation_data

        embeddings = dt.PairedGroupData(
            x=self.embeddings(val_data.x),
            y=self.embeddings(val_data.y),
            x_prime=self.embeddings(val_data.x_prime),
            y_prime=self.embeddings(val_data.y_prime),
            indices=val_data.indices,
        )
        return dt.GroupSolverPrediction(embeddings=embeddings,)


from groupcl.models.dynamics.orthogonal_procurstes import OrthogonalProcrustesModel


@jaxtyped(typechecker=typechecked)
@config_dataclass
class OrthProcrustesCLSolver(BaseSolver):

    # Models
    # encoder model
    model: EncoderModel = config_field(default_factory=MLP)
    dynamics_model: OrthogonalProcrustesModel = config_field(
        default_factory=OrthogonalProcrustesModel)

    # Optimizer
    optimizer: Optimizer = config_field(default_factory=AdamOptimizer)

    # Criterion
    criterion: criterions.ContrastiveCriterion = config_field(
        default_factory=criterions.MseInfoNCE)
    # either compute losses on both x and y pairs or only on y pairs
    compute_loss_on: Literal["both", "y"] = config_field(default="y")

    # state
    current_epoch: int = state_field(default=0)
    current_temp: float = state_field(default=0.0)

    def __lazy_post_init__(self):
        """Initialize solver with optimizer and temperature."""
        super().__lazy_post_init__()
        self.optimizer.lazy_init(parameters=self.model.parameters())
        self.reset()

    @jaxtyped(typechecker=typechecked)
    def train_step(self, batch: dt.ContrastiveGroupBatch) -> Dict[str, Any]:
        """Perform single training step using contrastive learning.

        Args:
            batch: dt.ContrastiveGroupBatch

        Returns:
            Dictionary containing training metrics
        """
        self.set_train()
        self.optimizer.zero_grad()

        embedding_batch = self.predict_step(batch)

        loss, loss_metrics = self.compute_loss(embedding_batch)

        # Backward pass
        loss.backward()
        self.optimizer.step()

        return dict(**loss_metrics,
                    loss=loss.item(),
                    temperature=self.current_temp)

    def predict_step(self, batch: dt.ContrastiveGroupBatch):
        """Takes an input batch and returns a batch of embeddings with predictions."""
        embedding_batch = self.encode_batch(batch)
        dynamics_predictions = self.dynamics_step(embedding_batch)
        return self.to_contrastive_loss_batch(
            embedding_batch=embedding_batch,
            dynamics_predictions=dynamics_predictions,
        )

    @jaxtyped(typechecker=typechecked)
    def dynamics_step(
        self,
        batch: dt.ContrastiveGroupBatch,
    ) -> dt.DynamicsPredictions:
        """Perform a single dynamics step."""

        self.dynamics_model.fit_parameters(
            x=batch.positives.x,
            x_prime=batch.positives.x_prime,
            system_idx=batch.positives.actions_idx,
        )
        x_prime_pred = self.dynamics_model(
            batch.positives.x, system_idx=batch.positives.actions_idx)
        y_prime_pred = self.dynamics_model(
            batch.positives.y, system_idx=batch.positives.actions_idx)

        return dt.DynamicsPredictions(
            x_prime=x_prime_pred,
            y_prime=y_prime_pred,
        )

    @jaxtyped(typechecker=typechecked)
    def to_contrastive_loss_batch(
        self,
        embedding_batch: dt.ContrastiveGroupBatch,
        dynamics_predictions: dt.DynamicsPredictions,
    ) -> dt.PairedContrastiveLossBatch:

        x = dt.ContrastiveLossBatch(
            reference=dynamics_predictions.x_prime,
            positive=embedding_batch.positives.x_prime,
            negative=embedding_batch.negatives.x,
        )

        y = dt.ContrastiveLossBatch(
            reference=dynamics_predictions.y_prime,
            positive=embedding_batch.positives.y_prime,
            negative=embedding_batch.negatives.y,
        )

        return dt.PairedContrastiveLossBatch(x=x, y=y)

    def encode_batch(
            self, batch: dt.ContrastiveGroupBatch) -> dt.ContrastiveGroupBatch:
        """Encode the batch into embeddings."""

        # To improve performance, we should instead concatenate all
        # of these tensors and then pass them through the model in one forward pass
        concatenated_data = torch.cat([
            batch.positives.x,
            batch.positives.y,
            batch.positives.x_prime,
            batch.positives.y_prime,
            batch.negatives.x,
            batch.negatives.y,
        ],
                                      dim=0)
        embeddings = self.embeddings(concatenated_data)

        # define the slices to split the embeddings into contrastive_group_batch
        pos_x_slice = slice(0, batch.positives.x.shape[0])
        pos_y_slice = slice(
            pos_x_slice.stop,
            pos_x_slice.stop + batch.positives.y.shape[0],
        )
        pos_x_prime_slice = slice(
            pos_y_slice.stop,
            pos_y_slice.stop + batch.positives.x_prime.shape[0],
        )
        pos_y_prime_slice = slice(
            pos_x_prime_slice.stop,
            pos_x_prime_slice.stop + batch.positives.y_prime.shape[0],
        )
        neg_x_slice = slice(
            pos_y_prime_slice.stop,
            pos_y_prime_slice.stop + batch.negatives.x.shape[0],
        )
        neg_y_slice = slice(
            neg_x_slice.stop,
            neg_x_slice.stop + batch.negatives.y.shape[0],
        )

        # create contrastive_group_batch
        embedding_batch = dt.ContrastiveGroupBatch(
            positives=dt.PairedGroupData(
                x=embeddings[pos_x_slice],
                y=embeddings[pos_y_slice],
                x_prime=embeddings[pos_x_prime_slice],
                y_prime=embeddings[pos_y_prime_slice],
                indices=batch.positives.indices,
                actions_idx=batch.positives.actions_idx,
            ),
            negatives=dt.PairedData(
                x=embeddings[neg_x_slice],
                y=embeddings[neg_y_slice],
                indices=batch.negatives.indices,
            ),
        )
        return embedding_batch

    def embeddings(
        self,
        batch: Float[Tensor, "batch obs_dim"],
    ) -> Float[Tensor, "batch latent_dim"]:
        """Encode the batch into embeddings."""
        return self.model(batch)

    def _compute_individual_loss(
        self,
        batch: dt.ContrastiveLossBatch,
    ) -> Tuple[Float[Tensor, ""], Dict[str, Any]]:
        """Compute loss for contrastive learning."""
        total, align, uniformity = self.criterion(batch)
        return total, dict(
            loss_align=align.item(),
            loss_uniformity=uniformity.item(),
            loss_total=total.item(),
        )

    @jaxtyped(typechecker=typechecked)
    def compute_loss(
        self, batch: dt.PairedContrastiveLossBatch
    ) -> Tuple[Float[Tensor, ""], Dict[str, Any]]:
        """Compute loss for contrastive learning."""
        if self.compute_loss_on == "both":
            x_loss, x_loss_metrics = self._compute_individual_loss(batch.x)
            y_loss, y_loss_metrics = self._compute_individual_loss(batch.y)
            loss = (x_loss + y_loss) / 2
        elif self.compute_loss_on == "y":
            x_loss_metrics = {}
            loss, y_loss_metrics = self._compute_individual_loss(batch.y)
        else:
            raise ValueError(f"Unknown compute_loss_on: {self.compute_loss_on}")

        # in any case, we compute both mse and dot procut between x ref and pos
        with torch.no_grad():
            mse_x = (batch.x.reference - batch.x.positive).pow(2).mean()
            dot_x = (torch.einsum('nd,nd->n', batch.x.reference,
                                  batch.x.positive)).mean()
            # compute determinant of Rs
            det_Rs = torch.det(self.dynamics_model.R).mean()
            RRT = torch.einsum('sij,sjk->sik', self.dynamics_model.R,
                               self.dynamics_model.R.transpose(1, 2))
            I = torch.eye(
                self.dynamics_model.dim,
                device=self.dynamics_model.R.device).unsqueeze(0).repeat(
                    self.dynamics_model.num_systems, 1, 1)
            mse_RRT_I = (RRT - I).pow(2).mean()

        return loss, dict(
            **{
                f"{k}_x": v for k, v in x_loss_metrics.items()
            },
            **{
                f"{k}_y": v for k, v in y_loss_metrics.items()
            },
            mse_x=mse_x.item(),
            dot_x=dot_x.item(),
            det_Rs=det_Rs.item(),
            mse_RRT_I=mse_RRT_I.item(),
        )

    @jaxtyped(typechecker=typechecked)
    def filter_tqdm_stats(self, stats: Dict[str, Any]) -> Dict[str, Any]:
        """Filter out keys that are not meant to be displayed in the progress bar."""
        key_to_keep = [
            "loss",
            "mse_x",
            "dot_x",
            "loss_mse_x",
            "loss_align_x",
            "loss_align_y",
            "loss_uniformity_x",
            "loss_uniformity_y",
        ]
        return {k: v for k, v in stats.items() if k not in key_to_keep}

    @jaxtyped(typechecker=typechecked)
    def validate_step(
        self,
        batch: dt.ContrastiveGroupBatch,
    ) -> Dict[str, Any]:
        """Does a single prediction step, computes metrics for the batch and returns both the predictions and the metrics"""

        self.set_eval()
        embedding_batch = self.predict_step(batch)
        loss, loss_metrics = self.compute_loss(embedding_batch)

        return dict(**loss_metrics, loss=loss.item())

    @jaxtyped(typechecker=typechecked)
    def predictions(
        self,
        loader: GroupContrastiveDataLoader,
    ) -> dt.GroupSolverPrediction:
        """Perform predictions for metrics."""

        val_data = loader.validation_data

        embeddings = dt.PairedGroupData(
            x=self.embeddings(val_data.x),
            y=self.embeddings(val_data.y),
            x_prime=self.embeddings(val_data.x_prime),
            y_prime=self.embeddings(val_data.y_prime),
            indices=val_data.indices,
            actions_idx=val_data.actions_idx,
        )
        x_prime_pred = self.dynamics_model(
            x=embeddings.x,
            x_prime=embeddings.x_prime,
            system_idx=embeddings.actions_idx,
        )
        y_prime_pred = self.dynamics_model(
            x=embeddings.y,
            x_prime=embeddings.y_prime,
            system_idx=embeddings.actions_idx,
        )
        return dt.GroupSolverPrediction(embeddings=embeddings,
                                        dynamics=dt.DynamicsPredictions(
                                            x_prime=x_prime_pred,
                                            y_prime=y_prime_pred,
                                        ))

    def to(self, device: torch.device):
        super().to(device)
        self.dynamics_model.to(device)


@jaxtyped(typechecker=typechecked)
@config_dataclass
class GCLSolver(BaseSolver):

    # Models
    # encoder model
    model: EncoderModel = config_field(default_factory=MLP)

    # Optimizer
    optimizer: Optimizer = config_field(default_factory=AdamOptimizer)

    # Criterion
    loss: criterions.GCLLoss = config_field(default_factory=criterions.GCLLoss)
    # state
    current_epoch: int = state_field(default=0)
    current_temp: float = state_field(default=0.0)

    action_counts: Dict[int, int] = state_field(default_factory=dict)

    def __lazy_post_init__(self):
        """Initialize solver with optimizer and temperature."""
        super().__lazy_post_init__()
        self.optimizer.lazy_init(parameters=self.model.parameters())
        self.reset()

    def reset(self):
        super().reset()
        self.action_counts = dict()

    @property
    def group_model(self) -> GroupDynamicsModel:
        return self.loss.group_model

    @jaxtyped(typechecker=typechecked)
    def train_step(self, batch: dt.ContrastiveGroupBatch) -> Dict[str, Any]:
        """Perform single training step using contrastive learning.

        Args:
            batch: dt.ContrastiveGroupBatch

        Returns:
            Dictionary containing training metrics
        """
        self.set_train()
        self.optimizer.zero_grad()

        embedding_batch = self.encode_batch(batch)

        loss, loss_metrics = self.loss(embedding_batch)

        # Backward pass
        loss.backward()
        self.optimizer.step()

        # log the number of unique actions used for training
        unique_actions, counts = torch.unique(
            embedding_batch.positives.actions_idx, return_counts=True)
        for action, count in zip(unique_actions.tolist(), counts.tolist()):
            if action in self.action_counts:
                self.action_counts[action] += count
            else:
                self.action_counts[action] = count

        return dict(**loss_metrics,
                    loss=loss.item(),
                    temperature=self.current_temp)

    def encode_batch(
            self, batch: dt.ContrastiveGroupBatch) -> dt.ContrastiveGroupBatch:
        """Encode the batch into embeddings."""

        # To improve performance, we should instead concatenate all
        # of these tensors and then pass them through the model in one forward pass

        # Reshape positives to handle batch, num_samples, obs_dim
        batch_size, num_samples, obs_dim = batch.positives.x.shape
        pos_x_flat = batch.positives.x.reshape(-1, obs_dim)
        pos_x_prime_flat = batch.positives.x_prime.reshape(-1, obs_dim)

        # Concatenate all tensors
        concatenated_data = torch.cat([
            pos_x_flat,
            pos_x_prime_flat,
            batch.negatives.x,
        ],
                                      dim=0)

        embeddings = self.embeddings(concatenated_data)

        # Define the slices to split the embeddings
        total_pos_x_samples = batch_size * num_samples
        pos_x_slice = slice(0, total_pos_x_samples)
        pos_x_prime_slice = slice(
            pos_x_slice.stop,
            pos_x_slice.stop + total_pos_x_samples,
        )
        neg_x_slice = slice(
            pos_x_prime_slice.stop,
            pos_x_prime_slice.stop + batch.negatives.x.shape[0],
        )

        # Reshape embeddings back to batch, num_samples, latent_dim
        latent_dim = embeddings.shape[1]
        pos_x_emb = embeddings[pos_x_slice].reshape(batch_size, num_samples,
                                                    latent_dim)
        pos_x_prime_emb = embeddings[pos_x_prime_slice].reshape(
            batch_size, num_samples, latent_dim)

        # Create contrastive_group_batch
        embedding_batch = dt.ContrastiveGroupBatch(
            positives=dt.SinglePairedGroupData(
                x=pos_x_emb,
                x_prime=pos_x_prime_emb,
                indices=batch.positives.indices,
                actions_idx=batch.positives.actions_idx,
            ),
            negatives=dt.SingleData(
                x=embeddings[neg_x_slice],
                indices=batch.negatives.indices,
            ),
        )
        return embedding_batch

    @jaxtyped(typechecker=typechecked)
    def embeddings(
        self,
        batch: Float[Tensor, " *batch obs_dim"],
    ) -> Float[Tensor, " *batch latent_dim"]:
        """Encode the batch into embeddings."""
        # model expects batch, dim.
        # in case batch contains multiple dimensions, we need to reshape a bit
        batch_shape = batch.shape[:-1]
        batch = batch.reshape(-1, batch.shape[-1])
        embeddings = self.model(batch)
        return embeddings.reshape(*batch_shape, -1)

    @jaxtyped(typechecker=typechecked)
    def filter_tqdm_stats(self, stats: Dict[str, Any]) -> Dict[str, Any]:
        """Filter out keys that are not meant to be displayed in the progress bar."""
        key_to_keep = [
            "loss",
            "loss_align",
            "loss_uniformity",
        ]
        return {k: v for k, v in stats.items() if k not in key_to_keep}

    @jaxtyped(typechecker=typechecked)
    def validate_step(
        self,
        batch: dt.ContrastiveGroupBatch,
    ) -> Dict[str, Any]:
        """Does a single prediction step, computes metrics for the batch and returns both the predictions and the metrics"""

        self.set_eval()
        embedding_batch = self.encode_batch(batch)
        loss, loss_metrics = self.loss(embedding_batch)

        return dict(**loss_metrics, loss=loss.item())

    @jaxtyped(typechecker=typechecked)
    def predictions(
        self,
        loader: GCLDataLoader,
    ) -> dt.GroupSolverPrediction:
        """Perform predictions for metrics."""

        val_data = loader.validation_data

        embeddings = dt.SinglePairedGroupData(
            x=self.embeddings(val_data.x),
            x_prime=self.embeddings(val_data.x_prime),
            indices=val_data.indices,
            actions_idx=val_data.actions_idx,
            class_idx=val_data.class_idx,
        )

        support_indices = loader.support_indicies_from_positive(
            positive_indices=embeddings.indices)
        support_embeddings = embeddings[support_indices]

        embeddings_for_fitting = dt.SinglePairedGroupData(
            x=torch.cat([embeddings.x.unsqueeze(1), support_embeddings.x],
                        dim=1),
            x_prime=torch.cat(
                [embeddings.x_prime.unsqueeze(1), support_embeddings.x_prime],
                dim=1),
            indices=torch.cat([
                embeddings.indices.unsqueeze(1),
                support_indices,
            ],
                              dim=1),
            actions_idx=torch.cat([
                embeddings.actions_idx.unsqueeze(1),
                support_embeddings.actions_idx
            ],
                                  dim=1),
            class_idx=torch.cat([
                embeddings.class_idx.unsqueeze(1), support_embeddings.class_idx
            ],
                                dim=1)
            if embeddings.class_idx is not None else None,
        )
        target, reference, targets_x_prime_pred, Q = self.loss.group_predictions(
            embeddings_for_fitting,
            target_dim_selector=0,
            reference_dim_selector=slice(1, None),
        )

        assert torch.allclose(target.x, embeddings.x)
        assert torch.allclose(target.x_prime, embeddings.x_prime)
        dynamics = dt.DynamicsPredictions(x_prime=targets_x_prime_pred, Qx=Q)

        return dt.GroupSolverPrediction(embeddings=embeddings,
                                        dynamics=dynamics)


@jaxtyped(typechecker=typechecked)
@config_dataclass
class InfoNCECLSolver(GCLSolver):

    # Criterion
    loss: criterions.InfoNCELoss = config_field(
        default_factory=criterions.InfoNCELoss)

    @jaxtyped(typechecker=typechecked)
    def predictions(
        self,
        loader: GCLDataLoader,
    ) -> dt.GroupSolverPrediction:
        """Perform predictions for metrics."""

        val_data = loader.validation_data

        embeddings = dt.SinglePairedGroupData(
            x=self.embeddings(val_data.x),
            x_prime=self.embeddings(val_data.x_prime),
            indices=val_data.indices,
            actions_idx=val_data.actions_idx,
            class_idx=val_data.class_idx,
        )

        dynamics = dt.DynamicsPredictions(x_prime=embeddings.x, Qx=None)

        assert dynamics.x_prime.shape == embeddings.x_prime.shape

        return dt.GroupSolverPrediction(embeddings=embeddings,
                                        dynamics=dynamics)
