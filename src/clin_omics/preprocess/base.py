from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import pandas as pd

from clin_omics.dataset import CanonicalDataset


@dataclass
class BasePreprocessor(ABC):
    source_layer: str | None = None
    target: str = "X"

    def fit(self, dataset: CanonicalDataset) -> "BasePreprocessor":
        frame = self._get_source_frame(dataset)
        self._fit_frame(frame)
        return self

    def transform(self, dataset: CanonicalDataset) -> CanonicalDataset:
        frame = self._get_source_frame(dataset)
        transformed = self._transform_frame(frame)
        return self._build_output_dataset(dataset, transformed)

    def fit_transform(self, dataset: CanonicalDataset) -> CanonicalDataset:
        return self.fit(dataset).transform(dataset)

    def _get_source_frame(self, dataset: CanonicalDataset) -> pd.DataFrame:
        if self.source_layer is None:
            return dataset.X
        try:
            return dataset.layers[self.source_layer]
        except KeyError as exc:
            raise KeyError(f"Layer '{self.source_layer}' not found.") from exc

    def _fit_frame(self, frame: pd.DataFrame) -> None:
        return None

    @abstractmethod
    def _transform_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError

    def _build_output_dataset(
        self, dataset: CanonicalDataset, transformed: pd.DataFrame
    ) -> CanonicalDataset:
        new_layers = {name: layer.copy() for name, layer in dataset.layers.items()}
        new_provenance: dict[str, Any] = dict(dataset.provenance)
        history = list(new_provenance.get("transform_history", []))
        history.append(self._history_entry())
        new_provenance["transform_history"] = history

        if self.target == "X":
            new_X = transformed
        else:
            new_X = dataset.X.copy()
            new_layers[self.target] = transformed

        return CanonicalDataset(
            X=new_X,
            obs=dataset.obs.copy(),
            var=dataset.var.copy(),
            layers=new_layers,
            provenance=new_provenance,
            dataset_id=dataset.dataset_id,
            embeddings={name: frame.copy() for name, frame in dataset.embeddings.items()},
            feature_scores={name: frame.copy() for name, frame in dataset.feature_scores.items()},
            assignments={name: series.copy() for name, series in dataset.assignments.items()},
        )

    def _history_entry(self) -> dict[str, Any]:
        return {
            "transform": self.__class__.__name__,
            "source_layer": self.source_layer,
            "target": self.target,
        }
