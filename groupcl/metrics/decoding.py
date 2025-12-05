from typing import Dict, Literal, Optional, Union

import numpy as np
import torch
from config_dataclass import config_dataclass, config_field
from jaxtyping import Float, Integer, jaxtyped
from jaxtyping._typeguard import typechecked
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from torch import Tensor

from groupcl.metrics.base import EmbeddingMetric
from groupcl.metrics.identifiability import get_subspace_slice, guess_embedding_space
from groupcl.utils.datatypes import EmbeddingPrediction


@jaxtyped(typechecker=typechecked)
@config_dataclass
class BehaviorDecoding(EmbeddingMetric):
    """Decodes behavior from the latent embeddings."""

    subspace: Optional[
        Literal[
            "group",
            "content",
        ]
    ] = config_field(default=None)

    @property
    def name(self) -> str:
        subspace_name = f"_from{self.subspace.capitalize()}" if self.subspace is not None else ""
        target_name = f"{self.target_name}Decoding"
        return f"{target_name}{subspace_name}_{self.method_name}"

    @property
    def target_name(self) -> str:
        raise NotImplementedError("BehaviorDecoding does not support target_name, implement it in the subclass")

    @property
    def method_name(self) -> str:
        raise NotImplementedError("BehaviorDecoding does not support method_name, implement it in the subclass")

    def target_variable(
        self,
        dataset,
    ) -> Union[
        Float[Tensor, " samples"],
        Integer[Tensor, " samples"],
    ]:
        raise NotImplementedError("BehaviorDecoding does not support target_variable, implement it in the subclass")

    def slice_pred_space(self, embeddings, dataset, solver):
        if self.subspace is None:
            return embeddings

        emb_dims = guess_embedding_space(solver, dataset)
        subspace_slice = get_subspace_slice(self.subspace, *emb_dims)
        embeddings = embeddings[..., subspace_slice]

        return embeddings

    @jaxtyped(typechecker=typechecked)
    def compute(
        self,
        predictions: EmbeddingPrediction,
        dataset=None,
        solver=None,
        **kwargs,
    ) -> Union[float, Dict[str, float]]:
        embeddings = predictions.x

        embeddings = self.slice_pred_space(embeddings, dataset, solver)
        target_variable = self.target_variable(dataset)

        return self._compute(
            X=embeddings.detach().cpu(),
            y=target_variable.detach().cpu(),
        )


class PositionTargetMixin:
    @property
    def target_name(self) -> str:
        return "Position"

    def target_variable(
        self,
        dataset,
    ) -> Float[Tensor, "samples"]:
        return dataset._data["position"]


class DirectionTargetMixin:
    @property
    def target_name(self) -> str:
        return "Direction"

    def target_variable(
        self,
        dataset,
    ) -> Integer[Tensor, "samples"]:
        direction_right = dataset._data["direction_right"]
        # direction_right is 1.0 for right and 0.0 for left
        # convert to long to make it a classifcation target
        direction = (direction_right > 0.5).long()
        return direction


class TimeIndexTargetMixin:
    @property
    def target_name(self) -> str:
        return "TimeIndex"

    def target_variable(
        self,
        dataset,
    ) -> Float[Tensor, "samples"]:
        return torch.arange(len(dataset)).float()


def accuracy(y_pred: Integer[Tensor, " num_samples"], y_test: Integer[Tensor, " num_samples"]) -> float:
    return (y_pred == y_test).float().mean().item()


@config_dataclass
class MethodMixin:
    def fit_predict(
        self,
        X_train: Float[Tensor, " num_samples latent_dim"],
        y_train: Union[Integer[Tensor, " num_samples"], Float[Tensor, " num_samples"]],
        X_test: Float[Tensor, " num_samples latent_dim"],
    ) -> Union[Integer[Tensor, " num_samples"], Float[Tensor, " num_samples"]]:
        raise NotImplementedError("MethodMixin does not support fit_predict, implement it in the subclass")

    def split(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
        return X_train, X_test, y_train, y_test


@config_dataclass
class ClassificationMixin(MethodMixin):
    def _compute(
        self,
        X: Float[Tensor, " num_samples latent_dim"],
        y: Integer[Tensor, " num_samples"],
        **kwargs,
    ) -> float:
        X_train, X_test, y_train, y_test = self.split(X, y)
        y_pred = self.fit_predict(X_train, y_train, X_test)
        return {
            "accuracy": accuracy(y_pred, y_test),
        }


def mean_absolute_error(
    y_pred: Float[Tensor, " num_samples"],
    y_target: Float[Tensor, " num_samples"],
    unit_factor: float = 1.0,
) -> float:
    # first multiply by unit_factor
    y_pred = y_pred * unit_factor
    y_target = y_target * unit_factor
    return torch.mean(torch.abs(y_pred - y_target)).item()


def median_absolute_error(
    y_pred: Float[Tensor, " num_samples"],
    y_target: Float[Tensor, " num_samples"],
    unit_factor: float = 1.0,
) -> float:
    # first multiply by unit_factor
    y_pred = y_pred * unit_factor
    y_target = y_target * unit_factor
    return torch.median(torch.abs(y_pred - y_target)).item()


@config_dataclass
class RegressionMixin(MethodMixin):
    def _compute(
        self,
        X: Float[Tensor, " num_samples latent_dim"],
        y: Float[Tensor, " num_samples"],
        **kwargs,
    ) -> Dict[str, float]:
        X_train, X_test, y_train, y_test = self.split(X, y)
        y_pred = self.fit_predict(X_train, y_train, X_test)

        # compute r2 and median error
        res = dict(
            r2=r2_score(y_pred, y_test),
            median_err=median_absolute_error(y_pred, y_test),
            mean_err=mean_absolute_error(y_pred, y_test),
        )

        return res


@config_dataclass
class KNNClassificationMixin(ClassificationMixin):
    n_neighbors: int = config_field(default=3)

    @property
    def method_name(self) -> str:
        return f"KNNClassification{self.n_neighbors}"

    def fit_predict(
        self,
        X_train: Float[Tensor, " num_samples latent_dim"],
        y_train: Integer[Tensor, " num_samples"],
        X_test: Float[Tensor, " num_samples latent_dim"],
    ) -> Integer[Tensor, " num_samples"]:
        self.knn = KNeighborsClassifier(n_neighbors=self.n_neighbors)
        self.knn.fit(X_train, y_train)
        y_pred = self.knn.predict(X_test)
        return torch.tensor(y_pred)


@config_dataclass
class LogisticRegressionMixin(ClassificationMixin):
    @property
    def method_name(self) -> str:
        return "LogisticRegression"

    def fit_predict(
        self,
        X_train: Float[Tensor, " num_samples latent_dim"],
        y_train: Integer[Tensor, " num_samples"],
        X_test: Float[Tensor, " num_samples latent_dim"],
    ) -> Integer[Tensor, " num_samples"]:
        self.classifier = LogisticRegression(
            multi_class="multinomial",
            solver="lbfgs",
            max_iter=1000,
            verbose=1,
        )
        self.classifier.fit(X_train, y_train)
        y_pred = self.classifier.predict(X_test)
        return torch.tensor(y_pred)


@config_dataclass
class DecisionTreeClassificationMixin(ClassificationMixin):
    @property
    def method_name(self) -> str:
        return "DecisionTreeClassification"

    def fit_predict(
        self,
        X_train: Float[Tensor, " num_samples latent_dim"],
        y_train: Integer[Tensor, " num_samples"],
        X_test: Float[Tensor, " num_samples latent_dim"],
    ) -> Integer[Tensor, " num_samples"]:
        self.classifier = DecisionTreeClassifier()
        self.classifier.fit(X_train, y_train)
        y_pred = self.classifier.predict(X_test)
        return torch.tensor(y_pred)


@config_dataclass
class KNNRegressionMixin(RegressionMixin):
    n_neighbors: int = config_field(default=3)

    @property
    def method_name(self) -> str:
        return f"KNNRegression{self.n_neighbors}"

    def fit_predict(
        self,
        X_train: Float[Tensor, " num_samples latent_dim"],
        y_train: Float[Tensor, " num_samples"],
        X_test: Float[Tensor, " num_samples latent_dim"],
    ) -> Float[Tensor, " num_samples"]:
        self.knn = KNeighborsRegressor(n_neighbors=self.n_neighbors)
        self.knn.fit(X_train, y_train)
        y_pred = self.knn.predict(X_test)
        return torch.tensor(y_pred)


@config_dataclass
class DecisionTreeRegressionMixin(RegressionMixin):
    @property
    def method_name(self) -> str:
        return "DecisionTreeRegression"

    def fit_predict(
        self,
        X_train: Float[Tensor, " num_samples latent_dim"],
        y_train: Float[Tensor, " num_samples"],
        X_test: Float[Tensor, " num_samples latent_dim"],
    ) -> Float[Tensor, " num_samples"]:
        self.tree = DecisionTreeRegressor()
        self.tree.fit(X_train, y_train)
        y_pred = self.tree.predict(X_test)
        return torch.tensor(y_pred)


@config_dataclass
class LinearRegressionMixin(RegressionMixin):
    @property
    def method_name(self) -> str:
        return "LinearRegression"

    def fit_predict(
        self,
        X_train: Float[Tensor, " num_samples latent_dim"],
        y_train: Float[Tensor, " num_samples"],
        X_test: Float[Tensor, " num_samples latent_dim"],
    ) -> Float[Tensor, " num_samples"]:
        self.lr = LinearRegression()
        self.lr.fit(X_train, y_train)
        y_pred = self.lr.predict(X_test)
        return torch.tensor(y_pred)


@jaxtyped(typechecker=typechecked)
@config_dataclass
class PositionKNNRegression(PositionTargetMixin, KNNRegressionMixin, BehaviorDecoding):
    pass


@jaxtyped(typechecker=typechecked)
@config_dataclass
class PositionDTRegression(PositionTargetMixin, DecisionTreeRegressionMixin, BehaviorDecoding):
    pass


@jaxtyped(typechecker=typechecked)
@config_dataclass
class PositionLinearRegression(PositionTargetMixin, LinearRegressionMixin, BehaviorDecoding):
    pass


@jaxtyped(typechecker=typechecked)
@config_dataclass
class DirectionKNNClassification(DirectionTargetMixin, KNNClassificationMixin, BehaviorDecoding):
    pass


@jaxtyped(typechecker=typechecked)
@config_dataclass
class DirectionLogisticRegression(DirectionTargetMixin, LogisticRegressionMixin, BehaviorDecoding):
    pass


@jaxtyped(typechecker=typechecked)
@config_dataclass
class DirectionDTClassification(DirectionTargetMixin, DecisionTreeClassificationMixin, BehaviorDecoding):
    pass


@jaxtyped(typechecker=typechecked)
@config_dataclass
class TimeIndexKNNRegression(TimeIndexTargetMixin, KNNRegressionMixin, BehaviorDecoding):
    pass


@jaxtyped(typechecker=typechecked)
@config_dataclass
class TimeIndexDTRegression(TimeIndexTargetMixin, DecisionTreeRegressionMixin, BehaviorDecoding):
    pass


@jaxtyped(typechecker=typechecked)
@config_dataclass
class TimeIndexLinearRegression(TimeIndexTargetMixin, LinearRegressionMixin, BehaviorDecoding):
    pass


from groupcl.utils.behavior import discretize_variable, generate_group_factors


def find_group_factor_index(query_factor, unique_factors):
    if isinstance(query_factor, list):
        query_factor = torch.tensor(query_factor, device=unique_factors.device)

    # Compare each row with the query
    matches = (unique_factors == query_factor).all(dim=1)
    indices = torch.where(matches)[0]

    if len(indices) > 0:
        return indices[0].item()
    else:
        return None


@jaxtyped(typechecker=typechecked)
@config_dataclass
class LinearPositionPrediction(BehaviorDecoding):
    prediction_step_size: float = config_field(default=0.05)
    prediction_steps_ahead: int = config_field(default=10)
    groupby: Optional[Literal["direction", "direction_change"]] = config_field(default_factory=None)
    return_rollout: bool = config_field(default=False, skip_default=True)
    max_eval_samples: Optional[int] = config_field(default=None, skip_default=True)
    max_train_samples: Optional[int] = config_field(default=None, skip_default=True)
    seed: int = config_field(default=42, skip_default=True)

    @property
    def name(self) -> str:
        subspace_name = f"_from{self.subspace.capitalize()}" if self.subspace is not None else ""
        target_name = f"{self.target_name}LinearPrediction"

        return f"{target_name}{subspace_name}_{self.method_name}"

    @property
    def method_name(self) -> str:
        groupby_name = f"_groupby{self.groupby.capitalize()}" if self.groupby is not None else ""
        step_size = self.prediction_step_size * 100
        # if int, round to int
        if int(step_size) == step_size:
            step_size = int(step_size)
        return f"LinearRegression_{step_size}cm_{self.prediction_steps_ahead}steps{groupby_name}"

    @property
    def target_name(self) -> str:
        return "LatentPosition"

    def prep_group_factors(self, dataset):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        old_behavior_variable_key = dataset.behavior_variable_key

        behavior_variable_key = [
            "position",
        ]
        bin_sizes = [self.prediction_step_size]
        equivariant_dims = [0]

        if self.groupby == "direction":
            behavior_variable_key += [
                "direction_left",
                "direction_right",
            ]
            bin_sizes += [None, None]
        elif self.groupby == "direction_change":
            behavior_variable_key += [
                "direction_left",
                "direction_right",
            ]
            bin_sizes += [None, None]
            equivariant_dims += [1, 2]

        dataset.behavior_variable_key = behavior_variable_key
        behavior_variables = dataset.behavior_variable.to(device)
        index = dataset.index.to(device)
        dataset.behavior_variable_key = old_behavior_variable_key

        auxilary_variable = discretize_variable(
            behavior_variables,
            bin_size=bin_sizes,
        )
        full_product = torch.cartesian_prod(index, index)

        group_factors, index_pairs = generate_group_factors(
            auxilary_variable,
            full_product,
            equivariant_dims=equivariant_dims,
        )

        (
            unique_group_factors,
            group_factor_ids,
        ) = torch.unique(group_factors, return_inverse=True, dim=0)

        return (
            unique_group_factors,
            index_pairs,
            group_factor_ids,
        )

    @jaxtyped(typechecker=typechecked)
    def compute(
        self,
        predictions: EmbeddingPrediction,
        dataset=None,
        solver=None,
        **kwargs,
    ) -> Union[float, dict]:
        embeddings = predictions.x

        embeddings = self.slice_pred_space(embeddings, dataset, solver)
        (
            unique_group_factors,
            index_pairs,
            group_factor_ids,
        ) = self.prep_group_factors(dataset)

        return self._compute(
            embeddings=embeddings.detach().cpu(),
            index_pairs=index_pairs.detach().cpu(),
            unique_group_factors=unique_group_factors.detach().cpu(),
            group_factor_ids=group_factor_ids.detach().cpu(),
        )

    def _fit_models(
        self,
        embeddings: Float[Tensor, " num_samples latent_dim"],
        index_pairs: Integer[Tensor, " num_pairs 2"],
        unique_group_factors: Integer[Tensor, " num_factors "],
        group_factor_ids: Integer[Tensor, " num_pairs"],
    ) -> list:
        trained_models = []
        train_factors = [group_factor for group_factor in unique_group_factors if abs(group_factor[0]) == 1]

        for train_factor in train_factors:
            group_id = find_group_factor_index(train_factor, unique_group_factors)
            mask = group_factor_ids == group_id
            index_group_pairs = index_pairs[mask]
            emb_group = embeddings[index_group_pairs]

            # Downsample if max_train_samples is specified
            if self.max_train_samples is not None and len(emb_group) > self.max_train_samples:
                generator = torch.Generator()
                generator.manual_seed(self.seed * 2)
                indices = torch.randperm(len(emb_group), generator=generator)[: self.max_train_samples]
                emb_group = emb_group[indices]

            # Get embeddings for the pairs
            emb_group_x = emb_group[:, 0, :]  # First element of each pair
            emb_group_x_prime = emb_group[:, 1, :]  # Second element of each pair

            # Fit linear regression model    # Fit linear regression model
            model = LinearRegression()
            model.fit(emb_group_x, emb_group_x_prime)

            trained_models.append(
                dict(
                    group_factor=train_factor,
                    model=model,
                )
            )

        return trained_models

    def _evaluate_models(
        self,
        trained_models: list,
        embeddings: Float[Tensor, " num_samples latent_dim"],
        index_pairs: Integer[Tensor, " num_pairs 2"],
        unique_group_factors: Integer[Tensor, " num_factors "],
        group_factor_ids: Integer[Tensor, " num_pairs"],
    ) -> list:
        rollout_results = []

        max_position_diff = unique_group_factors[:, 0].max().abs()
        assert max_position_diff >= self.prediction_steps_ahead, (
            f"prediction_steps_ahead can not be larger than {max_position_diff} but is {self.prediction_steps_ahead}"
        )
        for trained_model in trained_models:
            model_group_factor = trained_model["group_factor"]
            model = trained_model["model"]

            for k in range(1, self.prediction_steps_ahead + 1):
                data_group_factor = model_group_factor.clone()
                data_group_factor[0] *= k
                i = find_group_factor_index(data_group_factor, unique_group_factors)
                if i is None:
                    continue

                assert torch.all(data_group_factor == unique_group_factors[i])
                mask = group_factor_ids == i
                index_group_pairs = index_pairs[mask]
                emb_group = embeddings[index_group_pairs]

                # Downsample if max_eval_samples is specified
                if self.max_eval_samples is not None and len(emb_group) > self.max_eval_samples:
                    generator = torch.Generator()
                    generator.manual_seed(self.seed)
                    indices = torch.randperm(len(emb_group), generator=generator)[: self.max_eval_samples]
                    emb_group = emb_group[indices]

                # Get embeddings for the pairs
                emb_group_x = emb_group[:, 0, :].cpu().numpy()  # First element of each pair
                emb_group_x_prime = emb_group[:, 1, :].cpu().numpy()  # Second element of each pair

                last_pred = emb_group_x
                for _ in range(k):
                    last_pred = model.predict(last_pred)

                test_score = r2_score(last_pred, emb_group_x_prime)
                rollout_results.append(
                    dict(
                        model_group_factor=model_group_factor.cpu().tolist(),
                        group_factor=data_group_factor.cpu().tolist(),
                        test_score=float(test_score),
                        num_pairs=len(emb_group),
                        k=k,
                        position_diff=data_group_factor[0].item() * self.prediction_step_size,
                    )
                )

        return rollout_results

    def _process_results(self, rollout_results: list) -> dict:
        # Collect scores for each k value
        scores_by_k = []
        for k in range(1, self.prediction_steps_ahead + 1):
            k_scores = [result["test_score"] for result in rollout_results if result["k"] == k]
            scores_by_k.append(np.mean(k_scores) if k_scores else float("nan"))

        last_scores = scores_by_k[-1]
        all_scores = np.mean(scores_by_k)
        return_dict = dict(
            score_last_step=last_scores,
            score_all_steps=all_scores,
            scores=scores_by_k,
        )

        # Convert any inf or -inf values to nan in the return dict
        for key, value in return_dict.items():
            if isinstance(value, (int, float)):
                if np.isinf(value):
                    return_dict[key] = None
            elif isinstance(value, list):
                return_dict[key] = [None if np.isinf(v) else v for v in value]

        if self.return_rollout:
            return_dict["rollout_results"] = rollout_results
        return return_dict

    def _compute(
        self,
        embeddings: Float[Tensor, " num_samples latent_dim"],
        index_pairs: Integer[Tensor, " num_pairs 2"],
        unique_group_factors: Integer[Tensor, " num_factors "],
        group_factor_ids: Integer[Tensor, " num_pairs"],
    ) -> dict:
        trained_models = self._fit_models(embeddings, index_pairs, unique_group_factors, group_factor_ids)

        rollout_results = self._evaluate_models(trained_models, embeddings, index_pairs, unique_group_factors, group_factor_ids)

        return self._process_results(rollout_results)
