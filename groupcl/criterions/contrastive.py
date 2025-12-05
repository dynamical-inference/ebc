from abc import ABC
from abc import abstractmethod
from dataclasses import dataclass, field
from typing import Literal, Tuple, Dict, Any

import torch
from torch import Tensor
from jaxtyping import jaxtyped, Float
from jaxtyping._typeguard import typechecked

from config_dataclass import Configurable, torch_dataclass, config_field, check_initialized
from groupcl.utils import datatypes as dt


@torch.jit.script
def dot_similarity(ref: Tensor, pos: Tensor,
                   neg: Tensor) -> Tuple[Tensor, Tensor]:
    """Cosine similarity the ref, pos and negative pairs

    Args:
        ref: The reference samples of shape `(n, d)`.
        pos: The positive samples of shape `(n, d)`.
        neg: The negative samples of shape `(n, d)`.

    Returns:
        The similarity between reference samples and positive samples of shape `(n,)`, and
        the similarities between reference samples and negative samples of shape `(n, n)`.
    """
    pos_dist = torch.einsum("ni,ni->n", ref, pos)
    neg_dist = torch.einsum("ni,mi->nm", ref, neg)
    return pos_dist, neg_dist


@torch.jit.script
def euclidean_similarity(
    ref: Tensor,
    pos: Tensor,
    neg: Tensor,
) -> Tuple[Tensor, Tensor]:
    """Negative L2 distance between the ref, pos and negative pairs

    Args:
        ref: The reference samples of shape `(n, d)`.
        pos: The positive samples of shape `(n, d)`.
        neg: The negative samples of shape `(n, d)`.

    Returns:
        The similarity between reference samples and positive samples of shape `(n,)`, and
        the similarities between reference samples and negative samples of shape `(n, n)`.
    """
    ref_sq = torch.einsum("ni->n", ref**2)
    pos_sq = torch.einsum("ni->n", pos**2)
    neg_sq = torch.einsum("ni->n", neg**2)

    pos_cosine, neg_cosine = dot_similarity(ref, pos, neg)
    pos_dist = -(ref_sq + pos_sq - 2 * pos_cosine)
    neg_dist = -(ref_sq[:, None] + neg_sq[None] - 2 * neg_cosine)

    return pos_dist, neg_dist


@torch.jit.script
def dot_similarity_dual_ref(
    ref_pos: Tensor,
    ref_neg: Tensor,
    pos: Tensor,
    neg: Tensor,
) -> Tuple[Tensor, Tensor]:
    """Cosine similarity using separate reference tensors for positive and negative pairs.

    Args:
        ref_pos: The reference samples for positive pairs of shape `(n, d)`.
        ref_neg: The reference samples for negative pairs of shape `(n, d)`.
        pos: The positive samples of shape `(n, d)`.
        neg: The negative samples of shape `(n, d)`.

    Returns:
        The similarity between ref_pos and positive samples of shape `(n,)`, and
        the similarities between ref_neg and negative samples of shape `(n, n)`.
    """
    pos_dist = torch.einsum("ni,ni->n", ref_pos, pos)
    neg_dist = torch.einsum("ni,mi->nm", ref_neg, neg)
    return pos_dist, neg_dist


@torch.jit.script
def euclidean_similarity_dual_ref(
    ref_pos: Tensor,
    ref_neg: Tensor,
    pos: Tensor,
    neg: Tensor,
) -> Tuple[Tensor, Tensor]:
    """Negative L2 distance using separate reference tensors for positive and negative pairs.

    Args:
        ref_pos: The reference samples for positive pairs of shape `(n, d)`.
        ref_neg: The reference samples for negative pairs of shape `(n, d)`.
        pos: The positive samples of shape `(n, d)`.
        neg: The negative samples of shape `(n, d)`.

    Returns:
        The similarity between ref_pos and positive samples of shape `(n,)`, and
        the similarities between ref_neg and negative samples of shape `(n, n)`.
    """
    ref_pos_sq = torch.einsum("ni->n", ref_pos**2)
    ref_neg_sq = torch.einsum("ni->n", ref_neg**2)
    pos_sq = torch.einsum("ni->n", pos**2)
    neg_sq = torch.einsum("ni->n", neg**2)

    pos_cosine, neg_cosine = dot_similarity_dual_ref(ref_pos, ref_neg, pos, neg)
    pos_dist = -(ref_pos_sq + pos_sq - 2 * pos_cosine)
    neg_dist = -(ref_neg_sq[:, None] + neg_sq[None] - 2 * neg_cosine)

    return pos_dist, neg_dist


@torch.jit.script
def infonce(
        pos_dist: Tensor,  # nxd
        neg_dist: Tensor,  # nxd
):

    with torch.no_grad():
        c, _ = neg_dist.max(dim=1, keepdim=True)
    c = c.detach()

    pos_dist = pos_dist - c.squeeze(1)
    neg_dist = neg_dist - c

    pos = (-pos_dist).mean()
    neg = torch.logsumexp(neg_dist, dim=1).mean()

    c_mean = c.mean()
    numerator = pos - c_mean
    denominator = neg + c_mean
    return numerator + denominator, numerator, denominator


@torch.jit.script
def infonce_ratio(pos_dist: Tensor, neg_dist: Tensor, log_q_pos: Tensor,
                  log_q_neg: Tensor):

    with torch.no_grad():
        c, _ = neg_dist.max(dim=1, keepdim=True)
    c = c.detach()

    pos_dist = pos_dist - c.squeeze(1)
    neg_dist = neg_dist - c

    log_ratio = log_q_pos.unsqueeze(1) - log_q_neg.unsqueeze(0)

    pos = (-pos_dist).mean()
    neg = torch.logsumexp(log_ratio + neg_dist, dim=1).mean()

    c_mean = c.mean()
    numerator = pos - c_mean
    denominator = neg + c_mean
    return numerator + denominator, numerator, denominator


@torch.jit.script
def infonce_full_denominator(pos_dist, neg_dist):

    with torch.no_grad():
        c, _ = neg_dist.max(dim=1, keepdim=True)
    c = c.detach()

    pos_dist = pos_dist - c.squeeze(1)
    neg_dist = neg_dist - c

    numerator = (-pos_dist).mean()
    denominator = torch.logsumexp(
        torch.concatenate([
            pos_dist.unsqueeze(1),
            neg_dist,
        ], dim=1),
        dim=1,
    ).mean()

    c_mean = c.mean()
    numerator = numerator - c_mean
    denominator = denominator + c_mean

    return numerator + denominator, numerator, denominator


@torch_dataclass
class ContrastiveCriterion(Configurable, torch.nn.Module, ABC):
    """Contrastive criterion for training encoder model jointly with slds model using contrastive learning."""

    temperature: float = config_field(default=1.0)
    infonce_type: Literal["infonce", "infonce_full_denominator"] = config_field(
        default="infonce")

    def __new__(cls, *args, **k):
        inst = super().__new__(cls)
        torch.nn.Module.__init__(inst)
        return inst

    @torch.jit.export
    @abstractmethod
    def _distance(self,
                  batch: dt.ContrastiveLossBatch) -> Tuple[Tensor, Tensor]:
        """Compute distances between reference, positive and negative samples.

        Args:
            batch: Batch containing reference, positive and negative samples

        Returns:
            Tuple of:
                pos_dist: Distance between reference and positive samples
                neg_dist: Distance between reference and negative samples
        """

    def forward(
        self,
        batch: dt.ContrastiveLossBatch,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Compute the InfoNCE loss.

        Args:
            ref: The reference samples of shape `(n, d)`.
            pos: The positive samples of shape `(n, d)`.
            neg: The negative samples of shape `(n, d)`.
        """
        pos_dist, neg_dist = self._distance(batch)

        if self.infonce_type == "infonce":
            return infonce(pos_dist, neg_dist)
        elif self.infonce_type == "infonce_full_denominator":
            return infonce_full_denominator(pos_dist, neg_dist)
        else:
            raise ValueError


@torch_dataclass
class DotInfoNCE(ContrastiveCriterion):

    @torch.jit.export
    def _distance(self, batch: dt.ContrastiveLossBatch):
        pos_dist, neg_dist = dot_similarity(batch.reference, batch.positive,
                                            batch.negative)
        return pos_dist / self.temperature, neg_dist / self.temperature


@torch_dataclass
class MseInfoNCE(ContrastiveCriterion):

    @torch.jit.export
    def _distance(self, batch: dt.ContrastiveLossBatch):
        pos_dist, neg_dist = euclidean_similarity(
            ref=batch.reference,
            pos=batch.positive,
            neg=batch.negative,
        )
        return pos_dist / self.temperature, neg_dist / self.temperature


@torch_dataclass
class GroupCLCriterion(Configurable, torch.nn.Module, ABC):
    """Criterions that take the predictions of encoder model and dynamics model to compute the loss.
    Allows for more complex losses that may be compositions of multiple contrastive losses."""

    def __new__(cls, *args, **k):
        inst = super().__new__(cls)
        torch.nn.Module.__init__(inst)
        return inst

    @abstractmethod
    def forward(
        self,
        batch: dt.GroupSolverContrastivePrediction,
    ) -> Tuple[Float[Tensor, ""], Dict[str, Any]]:
        """Compute the loss.

        Args:
            batch: Batch containing encoder embeddings and dynamics predictions

        Returns:
            Tuple of:
                loss: The loss value
                loss_metrics: Dictionary of loss metrics
        """
        pass


@torch_dataclass
class GroupCLMseInfoNCE(GroupCLCriterion):
    """Wrapper around MSE InfoNCE criterion."""

    temperature: float = config_field(default=1.0)
    infonce_type: Literal[
        "infonce",
        "infonce_full_denominator",
    ] = config_field(default="infonce")

    def to_contrastive_loss_batch(
        self,
        batch: dt.GroupSolverContrastivePrediction,
    ) -> dt.PairedContrastiveLossBatch:

        embedding_batch = batch.embeddings
        dynamics_prediction = batch.dynamics
        # Flatten first two dimensions of predictions (batch, gumbel_samples, dim) -> (batch*gumbel_samples, dim)
        x_prime_pred = dynamics_prediction.x_prime.view(-1, batch.dynamics.dim)
        y_prime_pred = dynamics_prediction.y_prime.view(-1, batch.dynamics.dim)

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

    def _compute_individual_loss(
        self,
        batch: dt.ContrastiveLossBatch,
    ) -> Tuple[Float[Tensor, ""], Dict[str, Any]]:
        """Compute loss for contrastive learning."""
        if self.infonce_type == "infonce":
            infonce_fn = infonce
        elif self.infonce_type == "infonce_full_denominator":
            infonce_fn = infonce_full_denominator
        else:
            raise ValueError(f"Invalid infonce type: {self.infonce_type}")

        pos_dist, neg_dist = euclidean_similarity(batch.reference,
                                                  batch.positive,
                                                  batch.negative)
        pos_dist, neg_dist = apply_temperature(pos_dist, neg_dist,
                                               self.temperature)

        total, align, uniformity = infonce_fn(pos_dist, neg_dist)
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

    def forward(self, batch: dt.GroupSolverContrastivePrediction):
        loss_batch = self.to_contrastive_loss_batch(batch)
        loss, loss_metrics = self.compute_loss(loss_batch)
        return loss, loss_metrics


def apply_temperature(postive_dist: Tensor, negative_dist: Tensor,
                      temperature: float) -> Tuple[Tensor, Tensor]:
    return postive_dist / temperature, negative_dist / temperature


@torch_dataclass
class GroupCLMseInfoNCEDualReference(GroupCLCriterion):
    """MSE InfoNCE criterion with dual reference.
    Using dynamics prediction only for positive pairs. 
    """

    temperature: float = config_field(default=1.0)
    infonce_type: Literal[
        "infonce",
        "infonce_full_denominator",
    ] = config_field(default="infonce")

    @jaxtyped(typechecker=typechecked)
    def forward(self, batch: dt.GroupSolverContrastivePrediction):
        # Flatten first two dimensions of predictions (batch, gumbel_samples, dim) -> (batch*gumbel_samples, dim)
        x_prime_pred = batch.dynamics.x_prime.view(-1, batch.dynamics.dim)
        y_prime_pred = batch.dynamics.y_prime.view(-1, batch.dynamics.dim)

        # Repeat transformed (_prime) samples to match flattened predictions
        x_prime_repeated = batch.embeddings.positives.x_prime.repeat_interleave(
            repeats=batch.dynamics.gumbel_samples,
            dim=0,
        )
        y_prime_repeated = batch.embeddings.positives.y_prime.repeat_interleave(
            repeats=batch.dynamics.gumbel_samples,
            dim=0,
        )

        dist_batch_x = dict(
            ref_pos=x_prime_pred,
            pos=x_prime_repeated,
            ref_neg=batch.embeddings.positives.x,
            neg=batch.embeddings.negatives.x,
        )
        dist_batch_y = dict(
            ref_pos=y_prime_pred,
            pos=y_prime_repeated,
            ref_neg=batch.embeddings.positives.y,
            neg=batch.embeddings.negatives.y,
        )

        pos_dist_x, neg_dist_x = euclidean_similarity_dual_ref(**dist_batch_x)
        pos_dist_y, neg_dist_y = euclidean_similarity_dual_ref(**dist_batch_y)

        pos_dist_x, neg_dist_x = apply_temperature(pos_dist_x, neg_dist_x,
                                                   self.temperature)
        pos_dist_y, neg_dist_y = apply_temperature(pos_dist_y, neg_dist_y,
                                                   self.temperature)

        if self.infonce_type == "infonce":
            infonce_fn = infonce
        elif self.infonce_type == "infonce_full_denominator":
            infonce_fn = infonce_full_denominator
        else:
            raise ValueError(f"Invalid infonce type: {self.infonce_type}")

        total_x, align_x, uniformity_x = infonce_fn(
            pos_dist=pos_dist_x,
            neg_dist=neg_dist_x,
        )
        total_y, align_y, uniformity_y = infonce_fn(
            pos_dist=pos_dist_y,
            neg_dist=neg_dist_y,
        )

        loss = (total_x + total_y) / 2
        return loss, {
            "loss_align_x": align_x.item(),
            "loss_uniformity_x": uniformity_x.item(),
            "loss_align_y": align_y.item(),
            "loss_uniformity_y": uniformity_y.item(),
            "loss_total": loss.item(),
            "loss_total_x": total_x.item(),
            "loss_total_y": total_y.item(),
        }


@torch_dataclass
class SingleGroupMseInfoNCE(GroupCLCriterion):
    """
    Only applies a InfoNCE loss to the y samples. 
    For the x samples, we only apply a standard MSE loss.
    """

    temperature: float = config_field(default=1.0)
    infonce_type: Literal[
        "infonce",
        "infonce_full_denominator",
    ] = config_field(default="infonce")

    def forward(self, batch: dt.GroupSolverContrastivePrediction):
        # Flatten first two dimensions of predictions (batch, gumbel_samples, dim) -> (batch*gumbel_samples, dim)
        x_prime_pred = batch.dynamics.x_prime.view(-1, batch.dynamics.dim)
        y_prime_pred = batch.dynamics.y_prime.view(-1, batch.dynamics.dim)

        # Repeat transformed (_prime) samples to match flattened predictions
        x_prime_repeated = batch.embeddings.positives.x_prime.repeat_interleave(
            repeats=batch.dynamics.gumbel_samples,
            dim=0,
        )
        y_prime_repeated = batch.embeddings.positives.y_prime.repeat_interleave(
            repeats=batch.dynamics.gumbel_samples,
            dim=0,
        )

        ## Compute InfoNCE loss for y samples

        dist_batch_y = dict(
            ref_pos=y_prime_pred,
            pos=y_prime_repeated,
            ref_neg=batch.embeddings.positives.y,
            neg=batch.embeddings.negatives.y,
        )

        pos_dist_y, neg_dist_y = euclidean_similarity_dual_ref(**dist_batch_y)
        pos_dist_y, neg_dist_y = apply_temperature(pos_dist_y, neg_dist_y,
                                                   self.temperature)

        if self.infonce_type == "infonce":
            infonce_fn = infonce
        elif self.infonce_type == "infonce_full_denominator":
            infonce_fn = infonce_full_denominator
        else:
            raise ValueError(f"Invalid infonce type: {self.infonce_type}")

        total_y, align_y, uniformity_y = infonce_fn(
            pos_dist=pos_dist_y,
            neg_dist=neg_dist_y,
        )

        ## Compute MSE loss for x samples

        mse_loss_x = (x_prime_pred - x_prime_repeated).pow(2).mean()

        loss = mse_loss_x + total_y
        return loss, {
            "loss_mse_x": mse_loss_x.item(),
            "loss_align_y": align_y.item(),
            "loss_uniformity_y": uniformity_y.item(),
            "loss_total_y": total_y.item(),
            "loss_total": loss.item(),
        }
