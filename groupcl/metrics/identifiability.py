from abc import ABC, abstractmethod
from typing import Literal, Optional, Tuple, Union

import numpy as np
import scipy as sp
import torch
from config_dataclass import config_dataclass, config_field
from jaxtyping import Float, jaxtyped
from jaxtyping._typeguard import typechecked
from scipy.optimize import linear_sum_assignment
from sklearn.cross_decomposition import CCA as cca_skl
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from torch import Tensor

from groupcl.metrics import GroupMetric
from groupcl.metrics.utils import apply_identifiability_constant, fit_linear_regression, fit_ortho_procrustes
from groupcl.utils import datatypes as dt


def slice_latents(true_latents, embeddings, true_slice, pred_slice):
    if true_slice is not None:
        true_latents = true_latents[..., slice(*true_slice)]
    if pred_slice is not None:
        embeddings = embeddings[..., slice(*pred_slice)]
    return true_latents, embeddings


def guess_true_space(dataset):
    """
    Based on the dataset, determine the subspaces of the group and content in the ground truth latent space.

    ## For GT Latent Space -> we use dataset information
    latent_dim is always available
    dynamics_dim is always available
    from that we can infer the content_dim
    """
    data_dynamics_dim = dataset.dynamics_model.dim
    data_latent_dim = dataset.latent_dim
    data_content_dim = data_latent_dim - data_dynamics_dim

    return (
        data_latent_dim,
        data_dynamics_dim,
        data_content_dim,
    )


def guess_embedding_space(solver, dataset):
    """
    Based on the solver, determine the subspaces of the group and content in the embedding space.

    ## For Embedding Space -> we need to use solver information
    latent_dim is always available (model aka encoder output dim)

    content and dynamics dim may not be available
    dynamics dim may be in solver.loss.group_model.dynamics_dim
    content dim may be in solver.model.content_dims

    if neither is available,
    a) if latent dim is the same for data and solver, use content_dim from data
    b) otherwise, assume the ratio is the same and check if we can extrapolate
    """

    model_latent_dim = solver.model.output_dim
    model_content_dim = None
    model_dynamics_dim = None

    if hasattr(solver.model, "content_dims"):
        model_content_dim = solver.model.content_dims
        model_dynamics_dim = model_latent_dim - model_content_dim

    if hasattr(solver.loss, "group_model"):
        model_dynamics_dim = solver.loss.group_model.dynamics_dim
        model_content_dim = model_latent_dim - model_dynamics_dim

    if model_content_dim is None:
        data_latent_dim, data_dynamics_dim, data_content_dim = guess_true_space(dataset)
        if model_latent_dim == data_latent_dim:
            model_content_dim = data_content_dim
            model_dynamics_dim = data_dynamics_dim
        else:
            # assume the ratio is the same
            ratio = data_content_dim / data_latent_dim
            model_content_dim = model_latent_dim * ratio
            # only accept model_content_dim as correct if it is an integer
            if not model_content_dim.is_integer():
                model_content_dim = None

    return (
        model_latent_dim,
        model_dynamics_dim,
        model_content_dim,
    )


def guess_subspaces(solver, dataset):
    """
    Based on the solver and dataset, determine the subspaces of the group and content subspaces in the embeddings.
    """

    # assuming SyntheticContentDataset
    true_space = guess_true_space(dataset)
    embedding_space = guess_embedding_space(solver, dataset)

    return (true_space, embedding_space)


def get_subspace_slice(subspace_name, latent_dim, dynamics_dim, content_dim):
    """
    Get the slice for the subspace.
    """
    if subspace_name == "all":
        return slice(None, None)
    elif subspace_name == "group":
        return slice(0, dynamics_dim)
    elif subspace_name == "content":
        return slice(dynamics_dim, latent_dim)
    else:
        raise ValueError(f"Invalid subspace name: {subspace_name}")


@jaxtyped(typechecker=typechecked)
@config_dataclass
class IdentifiabilityMetric(GroupMetric, ABC):
    group: Literal[
        "x",
        "y",
        "x_prime",
        "y_prime",
        "x_prime_pred",
        "y_prime_pred",
        "x_prime_pred_emb",
        "all",
    ] = config_field(default="x")

    true_subspace: Optional[
        Literal[
            "group",
            "content",
            "all",
            "shape",
            "scale",
            "orientation",
            "posX",
            "posY",
        ]
    ] = config_field(default=None)

    pred_subspace: Optional[
        Literal[
            "group",
            "content",
            "all",
        ]
    ] = config_field(default=None)

    # allow to manually slices across latent dims.
    true_slice: Optional[Tuple[int, int]] = config_field(default=None)
    pred_slice: Optional[Tuple[int, int]] = config_field(default=None)

    hold_out_split: bool = config_field(default=False)

    @jaxtyped(typechecker=typechecked)
    def _get_data(
        self,
        predictions: dt.GroupSolverPrediction,
        ground_truth: dt.GroundTruthData,
        loader=None,
        solver=None,
    ) -> Tuple[
        Float[Tensor, "samples dim1"],
        Float[Tensor, "samples dim2"],
    ]:
        if self.group == "x":
            return ground_truth.latents.x, predictions.embeddings.x
        elif self.group == "x_prime":
            return ground_truth.latents.x_prime, predictions.embeddings.x_prime
        elif self.group == "x_prime_pred":
            return ground_truth.latents.x_prime, predictions.dynamics.x_prime.squeeze(1)
        elif self.group == "x_prime_pred_emb":
            return predictions.embeddings.x_prime, predictions.dynamics.x_prime.squeeze(1)
        elif self.group == "all":
            latents = torch.cat(
                [
                    ground_truth.latents.x,
                    ground_truth.latents.y,
                    ground_truth.latents.x_prime,
                    ground_truth.latents.y_prime,
                ],
                dim=0,
            )
            embeddings = torch.cat(
                [
                    predictions.embeddings.x,
                    predictions.embeddings.y,
                    predictions.embeddings.x_prime,
                    predictions.embeddings.y_prime,
                ],
                dim=0,
            )
            return latents, embeddings

    def slice_space(self, true_space, pred_space, loader, solver):
        # first true space
        true_space = self.slice_true_space(true_space, loader, solver)
        # then pred space
        pred_space = self.slice_pred_space(pred_space, loader, solver)

        return true_space, pred_space

    def slice_true_space(self, true_space, loader, solver):
        if self.true_subspace is None and self.true_slice is None:
            return true_space

        manual_slice = self.true_slice is not None
        subspace_slice = self.true_subspace is not None
        assert not (manual_slice and subspace_slice), "Cannot slice by both subspace and manual slice, one must be None"

        if manual_slice:
            return true_space[..., slice(*self.true_slice)]

        # we can't import dataset classes because of circular imports so we check class names
        if loader.dataset.__class__.__name__ == "DSpritesDataset" or loader.dataset.__class__.__name__ == "InfiniteDSpritesDataset":
            latent_names = loader.dataset.latent_names
            if self.true_subspace in latent_names:
                space_idx = [latent_names.index(self.true_subspace)]
            elif self.true_subspace == "all":
                space_idx = list(range(len(latent_names)))
            elif self.true_subspace in ["content", "group"]:
                name_map = {"content": ["shape", "scale"], "group": ["orientation", "posX", "posY"]}
                space_idx = [latent_names.index(name) for name in name_map[self.true_subspace]]

            else:
                raise ValueError(f"Invalid subspace name: {self.true_subspace}, must be one of {latent_names}")
            true_space = true_space[..., space_idx]
            return true_space

        else:
            # assume SyntheticContentDataset
            true_dims = guess_true_space(loader.dataset)
            true_subspace_slice = get_subspace_slice(self.true_subspace, *true_dims)
            true_space = true_space[..., true_subspace_slice]

            return true_space

    def slice_pred_space(self, pred_space, loader, solver):
        if self.pred_subspace is None and self.pred_slice is None:
            return pred_space

        manual_slice = self.pred_slice is not None
        subspace_slice = self.pred_subspace is not None
        assert not (manual_slice and subspace_slice), "Cannot slice by both subspace and manual slice, one must be None"

        if manual_slice:
            pred_space = pred_space[..., slice(*self.pred_slice)]
        elif subspace_slice:
            emb_dims = guess_embedding_space(solver, loader.dataset)
            pred_subspace_slice = get_subspace_slice(self.pred_subspace, *emb_dims)
            pred_space = pred_space[..., pred_subspace_slice]

        return pred_space

    @jaxtyped(typechecker=typechecked)
    def compute(
        self,
        predictions: dt.GroupSolverPrediction,
        ground_truth: dt.GroundTruthData,
        loader=None,
        solver=None,
        **kwargs,
    ) -> float:
        assert torch.all(ground_truth.index == predictions.embeddings.indices), "Ground truth and predictions indices do not match"

        latents, embeddings = self._get_data(
            predictions,
            ground_truth,
            loader=loader,
            solver=solver,
        )
        latents, embeddings = self.slice_space(latents, embeddings, loader, solver)
        return self._compute(z_true=latents.cpu(), z_pred=embeddings.cpu())

    @jaxtyped(typechecker=typechecked)
    @abstractmethod
    def _compute(
        self,
        z_true: Float[Tensor, "batch dim1"],
        z_pred: Float[Tensor, "batch dim2"],
    ) -> float:
        pass

    @property
    def name(self):
        if self.true_slice is None and self.pred_slice is None:
            t_default_name = ""
            pred_default_name = ""
        else:
            t_default_name = "_t[:]"
            pred_default_name = "_p[:]"
        true_slice_name = f"_t[{self.true_slice[0]}:{self.true_slice[1]}]" if self.true_slice is not None else t_default_name
        pred_slice_name = f"_p[{self.pred_slice[0]}:{self.pred_slice[1]}]" if self.pred_slice is not None else pred_default_name

        if self.true_subspace is None and self.pred_subspace is None:
            t_default_name = ""
            pred_default_name = ""
        else:
            t_default_name = "_t[all]"
            pred_default_name = "_p[all]"
        true_subspace_name = f"_t[{self.true_subspace}]" if self.true_subspace is not None else t_default_name
        pred_subspace_name = f"_p[{self.pred_subspace}]" if self.pred_subspace is not None else pred_default_name

        hold_out_name = "_hold_out" if self.hold_out_split else ""

        return f"{self.__class__.__name__}_{self.group.upper()}{true_slice_name}{true_subspace_name}{pred_slice_name}{pred_subspace_name}{hold_out_name}"


@jaxtyped(typechecker=typechecked)
@config_dataclass
class MCC(IdentifiabilityMetric):
    corr_method: Literal["Pearson", "Spearman"] = config_field(default="Pearson")

    @property
    def name(self):
        corr_method_name = f"_{self.corr_method.capitalize()}" if self.corr_method == "Spearman" else ""
        return f"{super().name}{corr_method_name}"

    @jaxtyped(typechecker=typechecked)
    def _compute(
        self,
        z_true: Float[Tensor, "batch dim1"],
        z_pred: Float[Tensor, "batch dim2"],
    ) -> float:
        x = z_true.numpy().copy().T
        y = z_pred.numpy().copy().T
        dim = x.shape[0]

        if self.corr_method == "Pearson":
            corr = np.corrcoef(y, x)
            corr = corr[0:dim, dim:]
        elif self.corr_method == "Spearman":
            corr, _ = sp.stats.spearmanr(y.T, x.T)
            corr = corr[0:dim, dim:]
        else:
            raise ValueError(f"Invalid correlation method: {self.corr_method}")

        row, cols = linear_sum_assignment(-np.absolute(corr))
        indexes = list(zip(row.tolist(), cols.tolist()))

        sort_idx = np.zeros(dim, dtype=int)
        for i in range(dim):
            sort_idx[i] = indexes[i][1]

        x_sort = x[sort_idx]
        if self.corr_method == "Pearson":
            corr_sort = np.corrcoef(y, x_sort)
            corr_sort = corr_sort[0:dim, dim:]
        elif self.corr_method == "Spearman":
            corr_sort, _ = sp.stats.spearmanr(y.T, x_sort.T)
            corr_sort = corr_sort[0:dim, dim:]

        mcc = np.mean(np.abs(np.diag(corr_sort)))
        return float(mcc)


@jaxtyped(typechecker=typechecked)
@config_dataclass
class CCA(IdentifiabilityMetric):
    n_components: Optional[int] = config_field(default=None)

    @jaxtyped(typechecker=typechecked)
    def _compute(
        self,
        z_true: Float[Tensor, "batch dim"],
        z_pred: Float[Tensor, "batch dim"],
    ) -> float:
        if self.hold_out_split:
            z_true_train, z_true_test, z_pred_train, z_pred_test = train_test_split(z_true, z_pred, test_size=0.5, random_state=42)
        else:
            z_true_train, z_pred_train = z_true, z_pred
            z_true_test, z_pred_test = z_true, z_pred

        n_components = self.n_components or z_true.shape[-1]
        cca = cca_skl(n_components=n_components)
        cca.fit(z_true_train, z_pred_train)

        z_true_c, z_pred_c = cca.transform(z_true_test, z_pred_test)
        correlation_matrix = np.corrcoef(z_true_c.T, z_pred_c.T)[:n_components, n_components:]
        mean_corr = correlation_matrix.diagonal().mean()
        return float(mean_corr)


@jaxtyped(typechecker=typechecked)
@config_dataclass
class R2(IdentifiabilityMetric):
    bias: bool = config_field(default=True)
    use_inverse: bool = config_field(default=False)
    direction: Literal["forward", "backward"] = config_field(default="backward")
    fitting_method: Literal["lsqt", "ortho-procrustes"] = config_field(default="lsqt", skip_default=True)

    def validate_config(self):
        super().validate_config()
        if self.bias and self.use_inverse:
            raise ValueError("bias=True and use_inverse=True is not supported for R2 metric")
        if self.fitting_method == "ortho-procrustes" and self.bias:
            raise ValueError("use_bias=True is not supported for R2 metric when fitting_method='ortho-procrustes'")

    @property
    def name(self):
        bias_name = "_bias" if self.bias else ""
        direction_name = f"_{self.direction.capitalize()}"
        use_inverse_name = "_ViaInverse" if self.use_inverse else ""

        fitting_method_name = ""
        if self.fitting_method == "ortho-procrustes":
            fitting_method_name = "_ortho"
        return f"{super().name}{direction_name}{bias_name}{use_inverse_name}{fitting_method_name}"

    @jaxtyped(typechecker=typechecked)
    def _directional_params(
        self,
        z_true: Float[Tensor, "batch dim1"],
        z_pred: Float[Tensor, "batch dim2"],
    ) -> Union[
        Tuple[Float[Tensor, "batch dim1"], Float[Tensor, "batch dim2"]],
        Tuple[Float[Tensor, "batch dim2"], Float[Tensor, "batch dim1"]],
    ]:
        if self.direction == "forward":
            return z_true, z_pred
        elif self.direction == "backward":
            return z_pred, z_true
        else:
            raise ValueError(f"Invalid direction: {self.direction}")

    @jaxtyped(typechecker=typechecked)
    def _compute(
        self,
        z_true: Float[Tensor, "batch dim1"],
        z_pred: Float[Tensor, "batch dim2"],
    ) -> float:
        x, y = self._directional_params(z_true, z_pred)
        if self.hold_out_split:
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.5, random_state=42)
        else:
            x_train, y_train = x, y
            x_test, y_test = x, y

        if self.fitting_method == "lsqt":
            if self.use_inverse:
                lr_model = fit_linear_regression(y_train, x_train, self.bias)
                y_pred = apply_identifiability_constant(A=np.linalg.inv(lr_model.coef_), x=x_test)
                r2 = r2_score(y_test, y_pred)
            else:
                lr_model = fit_linear_regression(x_train, y_train, self.bias)
                r2 = lr_model.score(x_test, y_test)
        elif self.fitting_method == "ortho-procrustes":
            if self.use_inverse:
                raise NotImplementedError("Orthogonal Procrustes fitting method does not support use_inverse=True")
            else:
                A_hat = fit_ortho_procrustes(x_train, y_train)
                y_test_pred = apply_identifiability_constant(A_hat, x_test)
                r2 = r2_score(y_test, y_test_pred)
        else:
            raise ValueError(f"Invalid fitting method: {self.fitting_method}")

        return float(r2)


@jaxtyped(typechecker=typechecked)
@config_dataclass
class VanillaR2(IdentifiabilityMetric):
    group: Literal["x_prime_pred_emb",] = config_field(default="x")

    true_subspace: Optional[
        Literal[
            "group",
            "content",
            "all",
        ]
    ] = config_field(default=None)

    pred_subspace: Optional[
        Literal[
            "group",
            "content",
            "all",
        ]
    ] = config_field(default=None)

    @jaxtyped(typechecker=typechecked)
    def _get_data(
        self,
        predictions: dt.GroupSolverPrediction,
        ground_truth: dt.GroundTruthData,
        loader=None,
        solver=None,
    ) -> Tuple[
        Float[Tensor, "samples dim1"],
        Float[Tensor, "samples dim2"],
    ]:
        if self.group == "x_prime_pred_emb":
            return predictions.embeddings.x_prime, predictions.dynamics.x_prime.squeeze(1)
        else:
            raise ValueError(f"Invalid group: {self.group} for VanillaR2 metric")

    def slice_true_space(self, true_space, loader, solver):
        if self.true_subspace is None and self.true_slice is None:
            return true_space

        manual_slice = self.true_slice is not None
        subspace_slice = self.true_subspace is not None
        assert not (manual_slice and subspace_slice), "Cannot slice by both subspace and manual slice, one must be None"

        if manual_slice:
            true_space = true_space[..., slice(*self.true_slice)]
        elif subspace_slice:
            emb_dims = guess_embedding_space(solver, loader.dataset)
            true_subspace_slice = get_subspace_slice(self.true_subspace, *emb_dims)
            print(f"VanillaR2: {subspace_slice} slice translated to true_subspace_slice: {true_subspace_slice}")
            true_space = true_space[..., true_subspace_slice]

        return true_space

    def slice_pred_space(self, pred_space, loader, solver):
        if self.pred_subspace is None and self.pred_slice is None:
            return pred_space

        manual_slice = self.pred_slice is not None
        subspace_slice = self.pred_subspace is not None
        assert not (manual_slice and subspace_slice), "Cannot slice by both subspace and manual slice, one must be None"

        if manual_slice:
            pred_space = pred_space[..., slice(*self.pred_slice)]
        elif subspace_slice:
            emb_dims = guess_embedding_space(solver, loader.dataset)
            pred_subspace_slice = get_subspace_slice(self.pred_subspace, *emb_dims)
            print(f"VanillaR2: {subspace_slice} slice translated to pred_subspace_slice: {pred_subspace_slice}")
            pred_space = pred_space[..., pred_subspace_slice]

        return pred_space

    @jaxtyped(typechecker=typechecked)
    def _compute(
        self,
        z_true: Float[Tensor, "batch dim1"],
        z_pred: Float[Tensor, "batch dim2"],
    ) -> float:
        r2 = r2_score(z_true, z_pred)

        return float(r2)


@jaxtyped(typechecker=typechecked)
def predict_foward(
    target: Float[Tensor, " *batch j"],
    Q: Float[Tensor, " *batch i j"],
) -> Float[Tensor, " *batch i"]:
    return torch.einsum("...ij,...j->...i", Q, target)


@jaxtyped(typechecker=typechecked)
@config_dataclass
class GroupHomomorphismR2(GroupMetric):
    """
    R2 score to measure the homomorphism property is retained in the embedding space.

    To achieve this we do the following:
    1. Generate true latents according to the homomorphism property.
        a) Sample two action matrices L, K from the dataset
        b) Compute G = L @ K
        c) sample random starting points x
        d) compute
            x_prime_k = K @ x,
            x_prime_l = L @ x_prime_k, --> should be equal to x_prime_l=(L @ K) @ x  = G @ x
            x_prime_g = G @ x
        e) optionally: double check that x_prime_g == L @ x_prime_k
        f) mix latents to get y, y_prime_l, y_prime_k, y_prime_g
    2. Compute embeddings, estiamte L, K, G
        a) use encoder to get emebddings emb_x, emb_x_prime_l, emb_x_prime_k, emb_x_prime_g
        b) compute L_hat = lstsq(emb_x_prime_l, emb_x)
        c) compute K_hat = lstsq(emb_x_prime_k, emb_x)
        d) compute G_hat = lstsq(emb_x_prime_g, emb_x)
    3. Apply G_hat to emb_x and L_hat@K_hat@emb_x
    4. compute R2 Score between these two vectors
    """

    num_samples: int = config_field(default=100)
    seed: int = config_field(default=42)

    @property
    def name(self):
        return f"{super().name}_n{self.num_samples}"

    def compute(
        self,
        *,
        solver=None,
        loader=None,
        **kwargs,
    ) -> float:
        # 1. generate data
        (
            y,
            y_prime_l_k,
            y_prime_k,
            y_prime_g,
        ) = self._generate_data(loader=loader)

        # 2. compute embeddings and estimate L, K, G
        (
            emb_x,
            emb_x_prime_l_k,
            emb_x_prime_k,
            emb_x_prime_g,
        ) = self._compute_embeddings(
            solver=solver,
            y=y,
            y_prime_l_k=y_prime_l_k,
            y_prime_k=y_prime_k,
            y_prime_g=y_prime_g,
        )

        # estimate L, K, G
        (
            (
                emb_x_hold_out,
                _,
            ),
            (
                G_hat,
                L_hat,
                K_hat,
            ),
        ) = self._estimate_group_matrices(
            solver=solver,
            emb_x=emb_x,
            emb_x_prime_l_k=emb_x_prime_l_k,
            emb_x_prime_k=emb_x_prime_k,
            emb_x_prime_g=emb_x_prime_g,
        )

        # 3. apply G_hat to emb_x_hold_out and L_hat@K_hat@emb_x_hold_out

        emb_x_prime_g_hat = predict_foward(emb_x_hold_out, Q=G_hat)
        emb_x_prime_lk_hat = predict_foward(emb_x_hold_out, Q=L_hat @ K_hat)

        emb_x_prime_l_k_hat = predict_foward(predict_foward(emb_x_hold_out, Q=K_hat), Q=L_hat)

        # sanity checking, emb_x_prime_lk_hat should equal emb_x_prime_l_k_hat
        assert torch.allclose(emb_x_prime_lk_hat, emb_x_prime_l_k_hat, atol=1e-5)

        # 3. compute R2 score
        r2 = r2_score(emb_x_prime_g_hat.detach().cpu().numpy(), emb_x_prime_lk_hat.detach().cpu().numpy())

        return float(r2)

    @jaxtyped(typechecker=typechecked)
    def _generate_data(
        self, loader
    ) -> Tuple[
        Float[Tensor, "batch samples_per_action dim"],
        Float[Tensor, "batch samples_per_action dim"],
        Float[Tensor, "batch samples_per_action dim"],
        Float[Tensor, "batch samples_per_action dim"],
    ]:
        dataset = loader.dataset
        device = dataset.device
        samples_per_action = loader.samples_per_action
        # seed
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        # a) Sample two action matrices L, K from the dataset
        unique_action_idx = torch.unique(dataset.get_action_idx(dataset.index))
        L_rand_indices = torch.randint(
            low=0,
            high=unique_action_idx.shape[0],
            size=(self.num_samples,),
        )
        K_rand_indices = torch.randint(
            low=0,
            high=unique_action_idx.shape[0],
            size=(self.num_samples,),
        )
        L_action_idx = unique_action_idx[L_rand_indices]
        K_action_idx = unique_action_idx[K_rand_indices]

        matrices = dataset.dynamics_model.A
        # the following gives us tensors of shape (num_samples, dim, dim)
        L = matrices[L_action_idx]
        K = matrices[K_action_idx]

        # b) Compute G = L @ K
        G = L @ K

        # c) sample random starting points x
        latents_x = dataset.starting_points_sampler.sample(
            num_samples=self.num_samples * samples_per_action,
            dim=dataset.dynamics_model.dim,
        ).to(device)
        # shape to (num_samples, samples_per_action, dim)
        latents_x = latents_x.reshape(self.num_samples, samples_per_action, -1)

        # d) compute
        #     x_prime_k = K @ x,
        #     x_prime_l = L @ x_prime_k, --> should be equal to x_prime_l=(L @ K) @ x  = G @ x
        #     x_prime_g = G @ x

        # also don't forget to repeat K, L, Q to introduce the samples_per_action dimension
        latents_x_prime_k = predict_foward(
            latents_x,
            Q=K.unsqueeze(1).repeat(1, samples_per_action, 1, 1),
        )
        latents_x_prime_l_k = predict_foward(
            latents_x_prime_k,
            Q=L.unsqueeze(1).repeat(1, samples_per_action, 1, 1),
        )
        latents_x_prime_g = predict_foward(
            latents_x,
            Q=G.unsqueeze(1).repeat(1, samples_per_action, 1, 1),
        )

        # e) sanity checks
        # x_prime_g == latents_x_prime_l_k
        assert torch.allclose(latents_x_prime_g, latents_x_prime_l_k, atol=1e-5)

        # latents_x_prime_l_k == (L @ K) @ x
        LK = (L @ K).unsqueeze(1).repeat(1, samples_per_action, 1, 1)
        assert torch.allclose(latents_x_prime_l_k, predict_foward(latents_x, Q=LK), atol=1e-5)

        # f) mix latents to get y, y_prime_l, y_prime_k, y_prime_g
        # before we mix, we need to sample content and concatenate it to the latents
        class_idx = dataset.content_embedding.sample_class_idx(num_samples=latents_x.shape[0] * samples_per_action)
        content_embeddings = dataset.content_embedding.embed(class_idx)
        # shape is (num_samples, dim), we need to repeat it
        # to get (num_samples, samples_per_action, dim)
        content_embeddings = content_embeddings.reshape(self.num_samples, samples_per_action, -1)

        latents_x = torch.cat(
            [
                latents_x,
                content_embeddings,
            ],
            dim=-1,
        )
        latents_x_prime_k = torch.cat(
            [
                latents_x_prime_k,
                content_embeddings,
            ],
            dim=-1,
        )
        latents_x_prime_l_k = torch.cat(
            [
                latents_x_prime_l_k,
                content_embeddings,
            ],
            dim=-1,
        )
        latents_x_prime_g = torch.cat(
            [
                latents_x_prime_g,
                content_embeddings,
            ],
            dim=-1,
        )

        # now we mix
        mixing_model = dataset.mixing_model.to(device)
        y = mixing_model(latents_x)
        y_prime_l_k = mixing_model(latents_x_prime_l_k)
        y_prime_k = mixing_model(latents_x_prime_k)
        y_prime_g = mixing_model(latents_x_prime_g)

        return (y, y_prime_l_k, y_prime_k, y_prime_g)

    @jaxtyped(typechecker=typechecked)
    def _compute_embeddings(
        self,
        solver,
        y: Float[Tensor, "batch samples_per_action dim"],
        y_prime_l_k: Float[Tensor, "batch samples_per_action dim"],
        y_prime_k: Float[Tensor, "batch samples_per_action dim"],
        y_prime_g: Float[Tensor, "batch samples_per_action dim"],
    ) -> Tuple[
        Float[Tensor, "batch samples_per_action latent_dim"],
        Float[Tensor, "batch samples_per_action latent_dim"],
        Float[Tensor, "batch samples_per_action latent_dim"],
        Float[Tensor, "batch samples_per_action latent_dim"],
    ]:
        solver.to(y.device)
        emb_x = solver.embeddings(y)
        emb_x_prime_l_k = solver.embeddings(y_prime_l_k)
        emb_x_prime_k = solver.embeddings(y_prime_k)
        emb_x_prime_g = solver.embeddings(y_prime_g)

        return (emb_x, emb_x_prime_l_k, emb_x_prime_k, emb_x_prime_g)

    @jaxtyped(typechecker=typechecked)
    def _estimate_group_matrices(
        self,
        solver,
        emb_x: Float[Tensor, "batch samples_per_action dim"],
        emb_x_prime_l_k: Float[Tensor, "batch samples_per_action dim"],
        emb_x_prime_k: Float[Tensor, "batch samples_per_action dim"],
        emb_x_prime_g: Float[Tensor, "batch samples_per_action dim"],
    ) -> Tuple[
        # target embeddings
        Tuple[
            Float[Tensor, "batch dim"],
            Float[Tensor, "batch dim"],
        ],
        # estimated group matrices
        Tuple[
            Float[Tensor, "batch dim dim"],
            Float[Tensor, "batch dim dim"],
            Float[Tensor, "batch dim dim"],
        ],
    ]:
        def predict_group(x, x_prime):
            positives = dt.SinglePairedGroupData(
                x=x,
                x_prime=x_prime,
            )
            hold_out_target, reference, targets_x_prime_pred, Q = solver.loss.group_predictions(
                positives=positives,
                target_dim_selector=0,
                reference_dim_selector=slice(1, None),
            )
            return hold_out_target.x, targets_x_prime_pred, Q

        emb_x_hold_out, _, G_hat = predict_group(
            x=emb_x,
            x_prime=emb_x_prime_g,
        )
        emb_x_prime_k_hold_out, _, L_hat = predict_group(
            # important here we need to use emb_x_prime_k as x
            x=emb_x_prime_k,
            x_prime=emb_x_prime_l_k,
        )

        emb_x_hold_out2, _, K_hat = predict_group(
            x=emb_x,
            x_prime=emb_x_prime_k,
        )
        assert torch.all(emb_x_hold_out == emb_x_hold_out2)

        return (
            (
                emb_x_hold_out,
                emb_x_prime_k_hold_out,
            ),
            (
                G_hat,
                L_hat,
                K_hat,
            ),
        )


def align_embeddings(
    z_true: Float[Union[Tensor, np.ndarray], "batch dim1"],
    z_pred: Float[Union[Tensor, np.ndarray], "batch dim2"],
    bias: bool = True,
) -> Tuple[
    Float[Union[Tensor, np.ndarray], "batch dim1"],
    float,
]:
    """
    Align embeddings via Linear Regression.
    Maps z_pred to z_true space via Linear Regression.
    """

    # Align embeddings with ground truth
    lr_model = LinearRegression(fit_intercept=bias)
    lr_model.fit(z_pred, z_true)
    aligned_embeddings = lr_model.predict(z_pred)
    r2 = lr_model.score(z_pred, z_true)

    return aligned_embeddings, r2
