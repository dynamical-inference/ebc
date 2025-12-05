import json
from dataclasses import field
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cebra
import numpy as np
import torch
from cebra import CEBRA
from cebra.models import register
from cebra.models.model import _OffsetModel
from config_dataclass import Configurable, config_dataclass, config_field
from dj_ml_core.utils import make_json_serializable, unique_timestamp
from sklearn.utils.validation import check_is_fitted
from torch import nn

from groupcl.datasets import HippcampusRatDataset
from groupcl.datasets.splits import CebraRatSplit
from groupcl.metrics.decoding import BehaviorDecoding
from groupcl.utils import datatypes as dt


@register("ebc_mlp")
class EbCMLP(_OffsetModel):
    def __init__(self, num_neurons, num_units, num_output, num_layers=3, normalize=True):
        super().__init__(
            nn.Flatten(start_dim=1, end_dim=-1),
            *self._make_layers(num_neurons, num_output, num_units, num_layers),
            num_input=num_neurons,
            num_output=num_output,
            normalize=normalize,
        )

    def _make_layers(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int,
        num_layers: int,
    ) -> List[nn.Module]:
        first_layers = [
            nn.Linear(
                input_dim,
                hidden_dim,
            ),
            nn.GELU(),
        ]
        middle_layers = []
        for _ in range(num_layers - 1):
            middle_layers.extend(
                [
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.GELU(),
                ]
            )
        last_layers = [
            nn.Linear(hidden_dim, output_dim),
        ]
        return first_layers + middle_layers + last_layers

    def get_offset(self) -> cebra.data.datatypes.Offset:
        """See :py:meth:`~.Model.get_offset`"""
        return cebra.data.Offset(0, 1)


@register("ebc_mlp_mse")
class EbCMLPMSE(EbCMLP):
    def __init__(self, num_neurons, num_units, num_output, num_layers=3, normalize=False):
        super().__init__(num_neurons, num_units, num_output, num_layers, normalize)


@config_dataclass
class CebraRatExperiment(Configurable):
    dataset: HippcampusRatDataset = config_field(default_factory=HippcampusRatDataset)
    dataset_split: CebraRatSplit = config_field(default_factory=CebraRatSplit)

    conditional: str = config_field(default="time_delta")
    time_offsets: int = config_field(default=10)

    model_architecture: str = config_field(default="offset10-model")
    output_dimension: int = config_field(default=32)
    num_hidden_units: int = config_field(default=32, skip_default=True)

    batch_size: int = config_field(default=512)
    learning_rate: float = config_field(default=3e-4)
    max_iterations: int = config_field(default=10000)
    distance: str = config_field(default="cosine")
    temperature: int = config_field(default=1)

    # CEBRA sklearn overwrites seed, we'll just use this to indicate a random seed
    seed: int = config_field(default_factory=lambda: np.random.randint(0, 2**32))

    # Logging parameters
    base_dir: str = field(default="logs")
    name: str = field(default="cebra_rat")
    time_stamp: str = field(default_factory=unique_timestamp)

    # run time parameters
    device: str = field(default="cuda_if_available")
    verbose: bool = field(default=True)

    def __lazy_post_init__(self):
        super().__lazy_post_init__()
        (
            self.train_data,
            self.val_data,
            self.test_data,
        ) = self.dataset_split.create_split(self.dataset)

        self.model = CEBRA(
            model_architecture=self.model_architecture,
            num_hidden_units=self.num_hidden_units,
            batch_size=self.batch_size,
            learning_rate=self.learning_rate,
            temperature=self.temperature,
            output_dimension=self.output_dimension,
            max_iterations=self.max_iterations,
            distance=self.distance,
            conditional=self.conditional,
            device=self.device,
            verbose=self.verbose,
            time_offsets=self.time_offsets,
        )

    @property
    def log_dir(self):
        return Path(self.base_dir) / self.name / self.time_stamp

    def _save_additional(self, path: Path):
        self.model.save(path / "cebra_model.pt")

    def _load_additional(self, path: Path):
        map_location = torch.device("cpu") if not torch.cuda.is_available() else None
        self.model = cebra.CEBRA.load(
            path / "cebra_model.pt",
            weights_only=False,
            map_location=map_location,
        )

    def run(self):
        try:
            check_is_fitted(self.model)
            print("Model already trained, skipping training")
        except Exception:
            print("Training Model:")
            neural_data = self.train_data.observed_variable
            behavior_variable = self.train_data.behavior_variable
            self.model.fit(neural_data, behavior_variable)

    def _evaluate(self, dataset, metrics: List[BehaviorDecoding]):
        neural_data = dataset.observed_variable.cpu()
        embeddings = self.model.transform(neural_data)
        predictions = dt.EmbeddingPrediction(x=torch.from_numpy(embeddings), indices=dataset.index)
        results = {}
        for metric in metrics:
            m_res = metric.compute(predictions=predictions, dataset=dataset)
            results[metric.name] = m_res
        return results

    def evaluate_train(self, **kwargs) -> Dict[str, Any]:
        return self._evaluate(self.train_data, **kwargs)

    def evaluate_val(self, **kwargs) -> Dict[str, Any]:
        return self._evaluate(self.val_data, **kwargs)

    def evaluate_test(self, **kwargs) -> Dict[str, Any]:
        return self._evaluate(self.test_data, **kwargs)

    @classmethod
    def load(cls, path: Path):
        inst = super().load(path)
        # reconstruct base dir, name and time stamp from path
        inst.base_dir = path.parent.parent
        inst.name = path.parent.name
        inst.time_stamp = path.name
        return inst

    def evaluate(
        self,
        save_results: bool = False,
        **kwargs,
    ) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
        """Evaluate the model on train, val, and test datasets."""
        train_metrics = self.evaluate_train(**kwargs)
        val_metrics = self.evaluate_val(**kwargs)
        test_metrics = self.evaluate_test(**kwargs)
        results = make_json_serializable(
            dict(
                train=dict(results=train_metrics),
                val=dict(results=val_metrics),
                test=dict(results=test_metrics),
            )
        )

        if save_results:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            with open(self.log_dir / "results.json", "w") as f:
                json.dump(results, f, indent=4)
        return results
