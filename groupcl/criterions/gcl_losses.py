import groupcl.utils.datatypes as dt
from config_dataclass import Configurable, config_dataclass, config_field, check_initialized
from groupcl.models.groups import GroupDynamicsModel
from groupcl.models.dynamics.slds import GumbelSLDS
from groupcl import criterions
import torch
from jaxtyping import jaxtyped
from jaxtyping._typeguard import typechecked


@jaxtyped(typechecker=typechecked)
@config_dataclass
class InfoNCELoss(Configurable):

    criterion: criterions.ContrastiveCriterion = config_field(
        default_factory=criterions.MseInfoNCE)

    def __call__(self, embeddings: dt.ContrastiveGroupBatch):

        loss_batch = dt.ContrastiveLossBatch(
            reference=embeddings.positives.x[:, 0],
            positive=embeddings.positives.x_prime[:, 0],
            negative=embeddings.negatives.x,
        )

        loss, loss_align, loss_uniformity = self.criterion(loss_batch)
        metrics = dict(
            loss_total=loss.item(),
            loss_align=loss_align.item(),
            loss_uniformity=loss_uniformity.item(),
        )

        return loss, metrics

    def group_predictions(self,
                          positives: dt.SinglePairedGroupData,
                          target_dim_selector: int = 0,
                          reference_dim_selector=slice(1, None)):
        """
        Placeholder function for group predicitons, which we don't really do here, but we implement the same interface as the other group losses.
        As a predictor we simply return Q=Identity matrix.
        """
        # separate across samples_per_system dimension
        target = positives[:, target_dim_selector]
        reference = positives[:, reference_dim_selector]

        targets_x_prime_pred = target.x
        Q = torch.eye(targets_x_prime_pred.shape[-1],
                      device=targets_x_prime_pred.device)
        return target, reference, targets_x_prime_pred, Q


@jaxtyped(typechecker=typechecked)
@config_dataclass
class GCLLoss(Configurable):

    group_model: GroupDynamicsModel = config_field(
        default_factory=GroupDynamicsModel)

    criterion: criterions.ContrastiveCriterion = config_field(
        default_factory=criterions.MseInfoNCE)

    def __call__(self, embeddings: dt.ContrastiveGroupBatch):
        return self.group_loss(embeddings)

    def group_predictions(self,
                          positives: dt.SinglePairedGroupData,
                          target_dim_selector: int = 0,
                          reference_dim_selector=slice(1, None)):
        # separate across samples_per_system dimension
        target = positives[:, target_dim_selector]
        reference = positives[:, reference_dim_selector]

        # check action indices matches for target and reference
        if positives.actions_idx is not None:
            target_actions_idx = positives.actions_idx[:, :1]
            reference_actions_idx = positives.actions_idx[:, 1:]
            assert torch.all(target_actions_idx == reference_actions_idx
                            ), "target and reference action indices must match"

        targets_x_prime_pred, Q = self.group_model(
            ref=reference.x,
            ref_prime=reference.x_prime,
            target=target.x,
        )
        return target, reference, targets_x_prime_pred, Q

    def group_loss(self,
                   embeddings: dt.ContrastiveGroupBatch,
                   target_dim_selector: int = 0,
                   reference_dim_selector=slice(1, None)):

        target, reference, targets_x_prime_pred, Q = self.group_predictions(
            embeddings.positives,
            target_dim_selector,
            reference_dim_selector,
        )

        loss_batch = dt.ContrastiveLossBatch(
            reference=targets_x_prime_pred,
            positive=target.x_prime,
            negative=embeddings.negatives.x,
        )

        loss, loss_align, loss_uniformity = self.criterion(loss_batch)

        metrics = dict(
            loss_total=loss.item(),
            loss_align=loss_align.item(),
            loss_uniformity=loss_uniformity.item(),
        )

        # compute additional metrics
        with torch.no_grad():
            ref_x_prime_pred = torch.einsum("...ij,...kj->...ki", Q,
                                            reference.x)
            # compute mse and dot product between ref_x_prime_pred and reference.x_prime
            mse_x_prime_pred = (ref_x_prime_pred -
                                reference.x_prime).pow(2).mean()
            dot_x_prime_pred = (torch.einsum('...d,...d->...', ref_x_prime_pred,
                                             reference.x_prime)).mean()
            det_Q = torch.det(Q).mean()

            additional_metrics = dict(
                mse_x_prime_pred=mse_x_prime_pred.item(),
                dot_x_prime_pred=dot_x_prime_pred.item(),
                det_Q=det_Q.item(),
            )

            # compute matrix norm of Q - I
            Q_minus_I = Q - torch.eye(Q.shape[-1], device=Q.device).expand(
                Q.shape)
            Q_minus_I_norm = torch.norm(Q_minus_I, p="fro", dim=(-2, -1))
            additional_metrics["Q_minus_I_norm"] = Q_minus_I_norm.mean().item()

        return loss, {**metrics, **additional_metrics}


@jaxtyped(typechecker=typechecked)
@config_dataclass
class SymmetricGCLLoss(GCLLoss):
    """
    In the symmetric GCL loss, we compute the loss both for x and x_prime and then again for the swapped x and x_prime.
    We then average the losses.
    """

    def __call__(self, embeddings: dt.ContrastiveGroupBatch):
        # probably don't need the clone, but just to be safe
        embeddings_swapped = embeddings.clone()
        x = embeddings.positives.x
        x_prime = embeddings.positives.x_prime
        embeddings_swapped.positives.x = x_prime
        embeddings_swapped.positives.x_prime = x
        # we'll also multiply actions_idx by -1 to indicate the swap
        if embeddings.positives.actions_idx is not None:
            embeddings_swapped.positives.actions_idx = -1 * embeddings.positives.actions_idx

        loss_1, metrics_1 = self.group_loss(embeddings)
        loss_2, metrics_2 = self.group_loss(embeddings_swapped)

        loss = torch.stack([loss_1, loss_2]).mean()
        all_metrics = [metrics_1, metrics_2]
        metrics = {
            k: torch.tensor([m[k] for m in all_metrics]).mean().item()
            for k in all_metrics[0].keys()
        }
        return loss, metrics


@jaxtyped(typechecker=typechecked)
@config_dataclass
class DCLLoss(GCLLoss):

    group_model: GumbelSLDS = config_field(default_factory=GumbelSLDS)

    def group_predictions(self,
                          positives: dt.SinglePairedGroupData,
                          target_dim_selector: int = 0,
                          reference_dim_selector=slice(1, None)):
        # separate across samples_per_system dimension
        target = positives[:, target_dim_selector]
        reference = positives[:, reference_dim_selector]
        self.group_model.to(target.x.device)

        targets_x_prime_pred, Q = self.group_model(
            x=target.x,
            x_prime=target.x_prime,
        )
        return target, reference, targets_x_prime_pred, Q

    def group_loss(self,
                   embeddings: dt.ContrastiveGroupBatch,
                   target_dim_selector: int = 0,
                   reference_dim_selector=slice(1, None)):

        target, _, targets_x_prime_pred, Q = self.group_predictions(
            embeddings.positives,
            target_dim_selector,
            reference_dim_selector,
        )

        loss_batch = dt.ContrastiveLossBatch(
            reference=targets_x_prime_pred,
            positive=target.x_prime,
            negative=embeddings.negatives.x,
        )

        loss, loss_align, loss_uniformity = self.criterion(loss_batch)

        metrics = dict(
            loss_total=loss.item(),
            loss_align=loss_align.item(),
            loss_uniformity=loss_uniformity.item(),
        )

        return loss, metrics


@jaxtyped(typechecker=typechecked)
@config_dataclass
class GCLLossV2(GCLLoss):

    def __call__(self, embeddings: dt.ContrastiveGroupBatch):
        # iterate over all possible target and reference combinations where the target is a hold-out example.

        num_samples = embeddings.positives.x.shape[1]
        all_samples_idx = torch.arange(num_samples)
        combinations = []
        for i in range(num_samples):
            combinations.append(
                dict(target_dim_selector=i,
                     reference_dim_selector=all_samples_idx[all_samples_idx !=
                                                            i]))

        all_losses = []
        all_metrics = []
        for combination in combinations:
            loss, metrics = self.group_loss(embeddings, **combination)
            all_losses.append(loss)
            all_metrics.append(metrics)

        loss = torch.stack(all_losses).mean()
        metrics = {
            k: torch.tensor([m[k] for m in all_metrics]).mean().item()
            for k in all_metrics[0].keys()
        }
        return loss, metrics


@jaxtyped(typechecker=typechecked)
@config_dataclass
class SymmetricGCLLossV2(GCLLoss):

    def __call__(self, embeddings: dt.ContrastiveGroupBatch):
        # iterate over all possible target and reference combinations where the target is a hold-out example.

        num_samples = embeddings.positives.x.shape[1]
        all_samples_idx = torch.arange(num_samples)
        combinations = []
        for i in range(num_samples):
            combinations.append(
                dict(target_dim_selector=i,
                     reference_dim_selector=all_samples_idx[all_samples_idx !=
                                                            i]))

        # probably don't need the clone, but just to be safe
        embeddings_swapped = embeddings.clone()
        x = embeddings.positives.x
        x_prime = embeddings.positives.x_prime
        embeddings_swapped.positives.x = x_prime
        embeddings_swapped.positives.x_prime = x
        # we'll also multiply actions_idx by -1 to indicate the swap
        if embeddings.positives.actions_idx is not None:
            embeddings_swapped.positives.actions_idx = -1 * embeddings.positives.actions_idx

        all_losses = []
        all_metrics = []
        for combination in combinations:
            # again just to be safe clone the embeddings
            embeddings_tmp = embeddings.clone()
            embeddings_swapped_tmp = embeddings_swapped.clone()

            loss_1, metrics_1 = self.group_loss(embeddings_tmp, **combination)
            loss_2, metrics_2 = self.group_loss(embeddings_swapped_tmp,
                                                **combination)

            loss = torch.stack([loss_1, loss_2]).mean()
            all_metrics = [metrics_1, metrics_2]
            metrics = {
                k: torch.tensor([m[k] for m in all_metrics]).mean().item()
                for k in all_metrics[0].keys()
            }
            all_losses.append(loss)
            all_metrics.append(metrics)

        loss = torch.stack(all_losses).mean()
        metrics = {
            k: torch.tensor([m[k] for m in all_metrics]).mean().item()
            for k in all_metrics[0].keys()
        }
        return loss, metrics


@jaxtyped(typechecker=typechecked)
@config_dataclass
class GCLLossV3(GCLLoss):

    def __call__(self, embeddings: dt.ContrastiveGroupBatch):
        # iterate over all possible target and reference combinations where the target is a hold-out example.

        num_samples = embeddings.positives.x.shape[1]
        all_samples_idx = torch.arange(num_samples)
        combinations = []
        for i in range(num_samples):
            combinations.append(
                dict(target_dim_selector=i,
                     reference_dim_selector=all_samples_idx[all_samples_idx !=
                                                            i]))

        # for V3, additionally also iterate over samples we used as references
        num_samples = embeddings.positives.x.shape[1]
        all_samples_idx = torch.arange(num_samples)
        for i in range(num_samples):
            combinations.append(
                dict(target_dim_selector=i,
                     reference_dim_selector=all_samples_idx[:]))

        all_losses = []
        all_metrics = []
        for combination in combinations:
            loss, metrics = self.group_loss(embeddings, **combination)
            all_losses.append(loss)
            all_metrics.append(metrics)

        loss = torch.stack(all_losses).mean()
        metrics = {
            k: torch.tensor([m[k] for m in all_metrics]).mean().item()
            for k in all_metrics[0].keys()
        }
        return loss, metrics


@jaxtyped(typechecker=typechecked)
@config_dataclass
class GCLDualLoss(GCLLoss):

    # first dimensions are dynamics, last are content
    content_dims: int = config_field(default=2)

    def content_loss(self, embeddings: dt.ContrastiveGroupBatch):
        # separate across samples_per_system dimension
        # NOTE: technically we woudn't need to separate into target and ref here
        # but for the purpose of using equal amount of ref/pos/negs f
        # or both the group and content loss we do so here

        target = embeddings.positives[:, 0]

        # Content loss doesn't use any prediction model
        loss_batch = dt.ContrastiveLossBatch(
            reference=target.x,
            positive=target.x_prime,
            negative=embeddings.negatives.x,
        )

        loss, loss_align, loss_uniformity = self.criterion(loss_batch)

        metrics = dict(
            loss_total=loss.item(),
            loss_align=loss_align.item(),
            loss_uniformity=loss_uniformity.item(),
        )
        return loss, metrics

    def __call__(self, embeddings: dt.ContrastiveGroupBatch):

        latent_dim = embeddings.positives.dim
        # separate across samples_per_system dimension
        content_start = latent_dim - self.content_dims
        content_end = latent_dim
        content_slice = slice(content_start, content_end)
        content_embeddings = slice_embeddings(embeddings, content_slice)

        group_loss, group_metrics = self.group_loss(embeddings)
        content_loss, content_metrics = self.content_loss(content_embeddings)

        return self.combine_losses(
            group_loss,
            group_metrics,
            content_loss,
            content_metrics,
        )

    def combine_losses(self, group_loss, group_metrics, content_loss,
                       content_metrics):
        loss = (group_loss + content_loss) / 2

        # add prefixes to the content and group metrics
        content_metrics = {
            f"content_{k}": v for k, v in content_metrics.items()
        }
        group_metrics = {f"group_{k}": v for k, v in group_metrics.items()}

        metrics = {**content_metrics, **group_metrics}

        return loss, metrics


@jaxtyped(typechecker=typechecked)
@config_dataclass
class GCLDualLossSplitSpace(GCLDualLoss):

    def __call__(self, embeddings: dt.ContrastiveGroupBatch):

        latent_dim = embeddings.positives.dim
        # separate across samples_per_system dimension
        content_start = latent_dim - self.content_dims
        content_end = latent_dim
        content_slice = slice(content_start, content_end)
        content_embeddings = embeddings.slice_latent_dim(content_slice)

        group_start = 0
        group_end = content_start
        group_slice = slice(group_start, group_end)
        group_embeddings = embeddings.slice_latent_dim(group_slice)

        group_loss, group_metrics = self.group_loss(group_embeddings)
        content_loss, content_metrics = self.content_loss(content_embeddings)

        return self.combine_losses(
            group_loss,
            group_metrics,
            content_loss,
            content_metrics,
        )


@jaxtyped(typechecker=typechecked)
@config_dataclass
class GCLDualLossContentOnly(GCLDualLoss):

    def __call__(self, embeddings: dt.ContrastiveGroupBatch):

        latent_dim = embeddings.positives.dim
        # separate across samples_per_system dimension
        content_start = latent_dim - self.content_dims
        content_end = latent_dim
        content_slice = slice(content_start, content_end)
        content_embeddings = embeddings.slice_latent_dim(content_slice)

        content_loss, content_metrics = self.content_loss(content_embeddings)

        return self.combine_losses(
            group_loss=content_loss,
            group_metrics={},
            content_loss=content_loss,
            content_metrics=content_metrics,
        )
