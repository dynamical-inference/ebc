from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Dict

from config_dataclass import Configurable, config_dataclass
from jaxtyping import jaxtyped
from jaxtyping._typeguard import typechecked

from groupcl.utils.datatypes import EmbeddingPrediction, GroundTruthData, GroupSolverPrediction

if TYPE_CHECKING:
    from groupcl.loader.base import BaseDataLoader
    from groupcl.solver.base import BaseSolver


@jaxtyped(typechecker=typechecked)
@config_dataclass
class Metric(Configurable, ABC):
    """Base class for all metrics."""

    @abstractmethod
    def compute(self, *args, **kwargs):
        """Compute the metric."""

    @property
    def name(self) -> str:
        return self.__class__.__name__


@jaxtyped(typechecker=typechecked)
@config_dataclass
class GroupMetric(Metric):
    """Group metric."""

    @jaxtyped(typechecker=typechecked)
    @abstractmethod
    def compute(
        self,
        predictions: GroupSolverPrediction,
        ground_truth: GroundTruthData,
        solver: "BaseSolver",
        loader: "BaseDataLoader",
    ) -> Dict[str, Any]:
        """Compute the metric."""


@jaxtyped(typechecker=typechecked)
@config_dataclass
class EmbeddingMetric(Metric):
    """Embedding metric."""

    @jaxtyped(typechecker=typechecked)
    @abstractmethod
    def compute(
        self,
        predictions: EmbeddingPrediction,
        solver: "BaseSolver",
    ) -> Dict[str, Any]:
        """Compute the metric."""
