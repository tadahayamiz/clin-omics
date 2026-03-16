from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from clin_omics.dataset import CanonicalDataset
from clin_omics.preprocess.base import BasePreprocessor


@dataclass
class VarianceFilter(BasePreprocessor):
    threshold: float = 0.0
    keep_features_: list[str] = field(init=False, default_factory=list)

    def __post_init__(self) -> None:
        self.target = "X"

    def _fit_frame(self, frame: pd.DataFrame) -> None:
        variances = frame.astype(float).var(axis=0, ddof=0)
        self.keep_features_ = variances[variances > self.threshold].index.tolist()

    def _transform_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        return frame.loc[:, self.keep_features_].copy()

    def _build_output_dataset(
        self, dataset: CanonicalDataset, transformed: pd.DataFrame
    ) -> CanonicalDataset:
        keep_columns = transformed.columns.tolist()
        new_var = dataset.var.set_index("feature_id").loc[keep_columns].reset_index()
        new_provenance = dict(dataset.provenance)
        history = list(new_provenance.get("transform_history", []))
        history.append(self._history_entry())
        new_provenance["transform_history"] = history

        return CanonicalDataset(
            X=transformed,
            obs=dataset.obs.copy(),
            var=new_var,
            layers={},
            provenance=new_provenance,
            dataset_id=dataset.dataset_id,
        )
