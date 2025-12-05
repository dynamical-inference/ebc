from collections import defaultdict
from typing import TYPE_CHECKING, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import torch
from config_dataclass import config_dataclass, config_field
from jaxtyping import Float, Integer, jaxtyped
from jaxtyping._typeguard import typechecked
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from torch import Tensor

from groupcl.metrics.base import GroupMetric
from groupcl.metrics.identifiability import get_subspace_slice, guess_embedding_space
from groupcl.utils.datatypes import GroundTruthData, GroupSolverPrediction

if TYPE_CHECKING:
    from groupcl.datasets.synthetic import DSpritesDataset
    from groupcl.loader import GCLDataLoader
    from groupcl.solver.contrastive_solver import GroupContrastiveLearningSolver


@jaxtyped(typechecker=typechecked)
@config_dataclass
class ContentClassification(GroupMetric):
    """Classifies class_idx from the latent embeddings."""

    group: Literal[
        "x",
        "x_prime",
        "all",
    ] = config_field(default="x")

    pred_subspace: Optional[
        Literal[
            "group",
            "content",
        ]
    ] = config_field(default=None)

    @property
    def name(self) -> str:
        pred_subspace_name = f"_p[{self.pred_subspace}]" if self.pred_subspace is not None else ""
        return f"{self.__class__.__name__}_{self.group}{pred_subspace_name}"

    @jaxtyped(typechecker=typechecked)
    def _get_data(
        self,
        predictions: GroupSolverPrediction,
    ) -> Tuple[
        Float[Tensor, "samples latent_dim"],
        Integer[Tensor, "samples"],
    ]:
        if self.group == "x":
            return predictions.embeddings.x, predictions.embeddings.class_idx
        elif self.group == "x_prime":
            return predictions.embeddings.x_prime, predictions.embeddings.class_idx
        elif self.group == "all":
            return (
                torch.cat([predictions.embeddings.x, predictions.embeddings.y, predictions.embeddings.x_prime, predictions.embeddings.y_prime], dim=0),
                torch.cat(
                    [predictions.embeddings.class_idx, predictions.embeddings.class_idx, predictions.embeddings.class_idx, predictions.embeddings.class_idx],
                    dim=0,
                ),
            )

    def slice_pred_space(self, pred_space, loader, solver):
        if self.pred_subspace is None:
            return pred_space

        emb_dims = guess_embedding_space(solver, loader.dataset)
        pred_subspace_slice = get_subspace_slice(self.pred_subspace, *emb_dims)
        pred_space = pred_space[..., pred_subspace_slice]

        return pred_space

    @jaxtyped(typechecker=typechecked)
    def compute(
        self,
        predictions: GroupSolverPrediction,
        ground_truth: GroundTruthData,
        loader=None,
        solver=None,
        **kwargs,
    ) -> float:
        assert torch.all(ground_truth.index == predictions.embeddings.indices), "Ground truth and predictions indices do not match"

        embeddings, latents_class_idx = self._get_data(
            predictions,
        )

        embeddings = self.slice_pred_space(embeddings, loader, solver)

        return self._compute(
            X=embeddings.detach().cpu(),
            y=latents_class_idx.detach().cpu(),
        )


@jaxtyped(typechecker=typechecked)
@config_dataclass
class KNNClassification(ContentClassification):
    """Classifies class_idx via a KNN classifier over the latent embeddings."""

    k: int = config_field(default=1)

    @property
    def name(self) -> str:
        return f"{super().name}_k{self.k}"

    @jaxtyped(typechecker=typechecked)
    def _compute(
        self,
        X: Float[Tensor, " num_samples latent_dim"],
        y: Integer[Tensor, " num_samples"],
        **kwargs,
    ) -> float:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

        self.knn = KNeighborsClassifier(n_neighbors=self.k)
        self.knn.fit(X_train, y_train)
        y_pred = self.knn.predict(X_test)
        return (y_pred == y_test).float().mean().item()


@jaxtyped(typechecker=typechecked)
@config_dataclass
class LDAClassification(ContentClassification):
    """Classifies class_idx via a KNN classifier over the latent embeddings."""

    @jaxtyped(typechecker=typechecked)
    def _compute(
        self,
        X: Float[Tensor, " num_samples latent_dim"],
        y: Integer[Tensor, " num_samples"],
        **kwargs,
    ) -> float:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

        self.lda = LinearDiscriminantAnalysis()
        self.lda.fit(X_train, y_train)
        y_pred = self.lda.predict(X_test)
        return (y_pred == y_test).float().mean().item()


@jaxtyped(typechecker=typechecked)
@config_dataclass
class LogisticRegressionClassification(ContentClassification):
    """Classifies class_idx via a KNN classifier over the latent embeddings."""

    @jaxtyped(typechecker=typechecked)
    def _compute(
        self,
        X: Float[Tensor, " num_samples latent_dim"],
        y: Integer[Tensor, " num_samples"],
        **kwargs,
    ) -> float:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

        self.classifier = LogisticRegression(
            multi_class="multinomial",
            solver="lbfgs",
            max_iter=1000,
            verbose=1,
        )
        self.classifier.fit(X_train, y_train)
        y_pred = self.classifier.predict(X_test)
        return (y_pred == y_test).float().mean().item()


@jaxtyped(typechecker=typechecked)
@config_dataclass
class NNActionPrediction(GroupMetric):
    """Nearest neighbor action prediction."""

    batch_size: int = config_field(default=10)
    epochs: int = config_field(default=1000)
    seed: int = config_field(default=42)
    samples_per_action: Optional[int] = config_field(default=None)
    action_names: List[List[str]] = config_field(default_factory=lambda: [["orientation"], ["posX"], ["posY"]])
    action_update_step_sizes: int = config_field(default=1)

    @property
    def name(self) -> str:
        num_samples = self.batch_size * self.epochs
        num_samples_str = f"_{num_samples:.1e}"
        seed_str = f"_seed_{self.seed:03d}"
        samples_per_action_str = f"_samples_per_action_{self.samples_per_action:03d}" if self.samples_per_action is not None else ""
        step_str = f"_step_{self.action_update_step_sizes:02d}"
        action_names_str = "_".join(["+".join(action_names) for action_names in self.action_names])
        return f"{super().name}_{action_names_str}{num_samples_str}{seed_str}{step_str}{samples_per_action_str}"

    @jaxtyped(typechecker=typechecked)
    def compute(
        self,
        solver: "GroupContrastiveLearningSolver",
        loader: "GCLDataLoader",
        **kwargs,
    ) -> Dict[
        str,
        Union[
            Integer[np.ndarray, "num_actions num_actions"],
            Float[np.ndarray, "num_actions num_actions"],
        ],
    ]:
        if self.samples_per_action is None:
            samples_per_action = loader.samples_per_action
        else:
            samples_per_action = self.samples_per_action

        (
            similarity_matrices,
            nn_action_predictions,
            true_actions,
        ) = self.nearest_neighbor_prediction(
            dataset=loader.dataset,
            solver=solver,
            samples_per_action=samples_per_action,
        )

        true_actions = true_actions.cpu().numpy()
        pred_actions = nn_action_predictions.cpu().numpy()
        unique_actions = np.unique(
            np.concatenate(
                [
                    true_actions.flatten(),
                    pred_actions.flatten(),
                ]
            )
        )
        cm = confusion_matrix(true_actions.flatten(), pred_actions.flatten(), labels=unique_actions)

        similarity_matrices = np.array(similarity_matrices.mean(dim=0).detach().cpu())
        return dict(
            confusion_matrix=cm,
            similarity_matrices=similarity_matrices,
        )

    def create_latent_updates(
        self,
        dataset: "DSpritesDataset",
    ) -> torch.Tensor:
        # for the inner lists of latent names we build the cartesian product of the axis variations
        # for th outer list we simply concatenate the cartesian products
        latent_updates = []
        for action_combinations in self.action_names:
            latent_names = dataset.latent_names
            latent_names = dataset.latent_names
            latent_sizes = dataset.latent_sizes
            cartesian_product = torch.cartesian_prod(
                *[
                    torch.arange(start=0, end=latent_sizes[latent_names.index(action_name)].item(), step=self.action_update_step_sizes)
                    for action_name in action_combinations
                ]
            )
            # if we only have one action in the action_combinations, we need to unsqueeze the cartesian product
            if len(action_combinations) == 1:
                cartesian_product = cartesian_product.unsqueeze(-1)

            latent_update = torch.zeros((cartesian_product.shape[0], dataset.latent_dim))

            for i, action_name in enumerate(action_combinations):
                axis_id = latent_names.index(action_name)
                latent_update[:, axis_id] = cartesian_product[:, i]
            latent_updates.append(latent_update)

        return torch.cat(latent_updates, dim=0)

    @jaxtyped(typechecker=typechecked)
    def nearest_neighbor_prediction(
        self,
        dataset: "DSpritesDataset",
        solver: "GroupContrastiveLearningSolver",
        samples_per_action: int = 10,
        return_intermediate_results: bool = False,
        **kwargs,
    ) -> Union[
        Tuple[
            Float[torch.Tensor, "num_samples num_actions num_actions"],
            Integer[torch.Tensor, "num_samples num_actions"],
            Integer[torch.Tensor, "num_samples num_actions"],
        ],
        Tuple[
            Float[torch.Tensor, "num_samples num_actions num_actions"],
            Integer[torch.Tensor, "num_samples num_actions"],
            Integer[torch.Tensor, "num_samples num_actions"],
            Dict[str, any],
        ],
    ]:
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        intermediate_results = defaultdict(list)
        nn_action_predictions = []
        true_actions = []
        similarity_matrices = []
        for _ in range(self.epochs):
            # 1) pick reference sample
            reference_x = dataset._generate_starting_points(num_samples=self.batch_size)

            # 2a) pick action

            latent_update = self.create_latent_updates(dataset=dataset)
            latent_update = latent_update.to(reference_x.device)
            latent_update = latent_update.unsqueeze(-2).unsqueeze(0)
            # along the varying axis, set the starting variation to 0 for consistent plots

            # get unique latent names
            unique_action_names = []
            for axis_combinations in self.action_names:
                for axis_name in axis_combinations:
                    if axis_name not in unique_action_names:
                        unique_action_names.append(axis_name)

            for action_name in unique_action_names:
                axis_id = dataset.latent_names.index(action_name)
                reference_x[..., axis_id] = 0

            # 2b) apply to reference to get transformed reference

            reference_x_prime = reference_x.unsqueeze(-2).unsqueeze(-2) + latent_update
            reference_x_prime = torch.remainder(reference_x_prime, dataset.latent_sizes)

            # 3) sample different references for Q fitting
            fitting_x = dataset._generate_starting_points(num_samples=samples_per_action * np.prod(reference_x_prime.shape[:-2]).item())
            fitting_x = fitting_x.view(*reference_x_prime.shape[:-2], samples_per_action, dataset.latent_dim)
            # 4) apply action to fitting samples
            fitting_x_prime = fitting_x + latent_update
            fitting_x_prime = torch.remainder(fitting_x_prime, dataset.latent_sizes)

            # 5) get observed data and compute model embeddings
            reference_x_prime = reference_x_prime.squeeze(-2)
            reference_y = dataset.latent_to_img(reference_x)
            reference_y_prime = dataset.latent_to_img(reference_x_prime)
            fitting_y = dataset.latent_to_img(fitting_x)
            fitting_y_prime = dataset.latent_to_img(fitting_x_prime)

            reference_emb_x = solver.embeddings(reference_y)
            reference_emb_x_prime = solver.embeddings(reference_y_prime)
            fitting_emb_x = solver.embeddings(fitting_y)
            fitting_emb_x_prime = solver.embeddings(fitting_y_prime)

            # 6) compute Q and predict reference_prime
            reference_emb_x_prime_pred, Q = solver.loss.group_model(
                ref=fitting_emb_x.view(-1, *fitting_emb_x.shape[-2:]),
                ref_prime=fitting_emb_x_prime.view(-1, *fitting_emb_x_prime.shape[-2:]),
                target=reference_emb_x.repeat_interleave(latent_update.shape[1], dim=0),
            )
            reference_emb_x_prime_pred = reference_emb_x_prime_pred.view(reference_emb_x_prime.shape)
            Q = Q.view(*reference_emb_x_prime.shape[:2], *Q.shape[1:])

            # 7) Find NN: Compute MSE between each predicted embedding and all embeddings
            mse_matrix = torch.cdist(reference_emb_x_prime_pred, reference_emb_x_prime, p=2).pow(2)
            # Convert to similarity (negative MSE so smaller distances = higher similarity)
            similarity_matrix = -mse_matrix
            nn_idx = torch.argmax(similarity_matrix, dim=-1)

            index_for_gather = nn_idx.unsqueeze(-1)  # Shape: (2, 39, 1)
            # Expand to match the third dimension of reference_x_prime
            index_for_gather = index_for_gather.expand(-1, -1, reference_x_prime.shape[2])  # Shape: (2, 39, 5)
            nn_latent_prediction = torch.gather(reference_x_prime, 1, index_for_gather)
            # the actions are 1-indexed so we need to add 1
            nn_action_prediction = nn_idx

            true_action = (
                torch.arange(
                    start=0,
                    end=latent_update.shape[1],
                )
                .unsqueeze(0)
                .repeat(nn_action_prediction.shape[0], 1)
            )

            nn_action_predictions.append(nn_action_prediction.detach().cpu())
            true_actions.append(true_action.detach().cpu())
            similarity_matrices.append(similarity_matrix.detach().cpu())
            if return_intermediate_results:
                # add embeddings to intermediate results
                intermediate_results["reference_emb_x_prime_pred"].append(reference_emb_x_prime_pred)
                intermediate_results["reference_emb_x_prime"].append(reference_emb_x_prime)
                intermediate_results["reference_emb_x"].append(reference_emb_x)
                intermediate_results["reference_x"].append(reference_x)
                intermediate_results["reference_x_prime"].append(reference_x_prime)
                intermediate_results["Q"].append(Q)
                intermediate_results["nn_latent_predictions"].append(nn_latent_prediction)
                intermediate_results["latent_update"].append(latent_update)

        nn_action_predictions = torch.cat(nn_action_predictions, dim=0)
        true_actions = torch.cat(true_actions, dim=0)
        similarity_matrices = torch.cat(similarity_matrices, dim=0)

        if return_intermediate_results:
            for key, value in intermediate_results.items():
                intermediate_results[key] = torch.cat(value, dim=0).detach().cpu()
            return similarity_matrices.detach().cpu(), nn_action_predictions.detach().cpu(), true_actions.detach().cpu(), dict(intermediate_results)
        else:
            return similarity_matrices.detach().cpu(), nn_action_predictions.detach().cpu(), true_actions.detach().cpu()


from tqdm import tqdm


@jaxtyped(typechecker=typechecked)
@config_dataclass
class NNActionAccuracy(GroupMetric):
    """Nearest neighbor after having applied the action representation and compute the accuracy."""

    k: int = config_field(default=1)
    batch_size: int = config_field(default=10)
    max_samples: int = config_field(default=None)
    seed: int = config_field(default=42)

    @property
    def name(self) -> str:
        return f"{self.__class__.__name__}_top_{self.k}_n_{self.max_samples}"

    def compute(
        self,
        predictions: GroupSolverPrediction,
        **kwargs,
    ) -> float:
        if self.max_samples is not None:
            # downsample the predictions to max_samples
            torch.manual_seed(self.seed)
            np.random.seed(self.seed)
            rand_indices = torch.randperm(predictions.embeddings.x.shape[0])[: self.max_samples]
            predictions = predictions[rand_indices]

        emb_x = predictions.embeddings.x
        emb_x_prime = predictions.embeddings.x_prime
        emb_x_prime_pred = predictions.dynamics.x_prime

        # pool of embeddings to find the nearest neighbors in
        all_embeddings = torch.cat(
            [emb_x_prime, emb_x],
            dim=0,
        )
        correct_neighbor_idx = torch.arange(emb_x.shape[0])

        embedding_topk_neighbors, distances = self._get_nearest_neighbors(emb_x_prime_pred, all_embeddings)

        # compute topk accuracy
        topk_acc = (embedding_topk_neighbors == correct_neighbor_idx.unsqueeze(-1)).any(-1).float().mean().item()
        return topk_acc

    @jaxtyped(typechecker=typechecked)
    def _get_nearest_neighbors(
        self, embeddings: Float[Tensor, "n d"], all_embeddings: Float[Tensor, "m d"], k: Optional[int] = None
    ) -> Tuple[
        Integer[Tensor, "n k"],
        Float[Tensor, "n m"],
    ]:
        """Get the k nearest neighbors of the embeddings in all_embeddings."""
        if k is None:
            k = self.k
        # compute the pairwise distances between the embeddings and all_embeddings
        if isinstance(embeddings, torch.Tensor):
            embeddings = embeddings.to(all_embeddings.device)
        if isinstance(embeddings, np.ndarray):
            embeddings = torch.from_numpy(embeddings)
        if isinstance(all_embeddings, np.ndarray):
            all_embeddings = torch.from_numpy(all_embeddings).to(embeddings.device)

        # compute batched across embeddings
        topk_neighbor_indices = []
        topk_neighbor_distances = []
        distances = []
        for i in tqdm(range(0, embeddings.shape[0], self.batch_size)):
            embeddings_batch_slice = slice(i, i + self.batch_size)
            distance_matrix = torch.cdist(embeddings[embeddings_batch_slice].float(), all_embeddings.float())
            distance_matrix = -distance_matrix
            topk_neighbors = distance_matrix.topk(self.k, dim=-1)
            topk_neighbor_idx = topk_neighbors.indices.detach().cpu()
            topk_neighbor_dist = topk_neighbors.values.detach().cpu()
            distance_matrix = distance_matrix.detach().cpu()

            topk_neighbor_indices.append(topk_neighbor_idx)
            topk_neighbor_distances.append(topk_neighbor_dist)
            distances.append(distance_matrix)

        topk_neighbor_indices = torch.cat(topk_neighbor_indices, dim=0)
        topk_neighbor_distances = torch.cat(topk_neighbor_distances, dim=0)
        distances = torch.cat(distances, dim=0)

        return topk_neighbor_indices, distances
