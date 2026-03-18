from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from clin_omics.dataset import CanonicalDataset
from clin_omics.preprocess.base import BasePreprocessor
from clin_omics.preprocess.scaling import ZScoreScaler


@dataclass
class FilterLowExpression(BasePreprocessor):
    min_count: float = 10.0
    min_samples: int = 1
    keep_features_: list[str] = field(init=False, default_factory=list)

    def __post_init__(self) -> None:
        self.target = "X"

    def _fit_frame(self, frame: pd.DataFrame) -> None:
        numeric = frame.astype(float)
        keep_mask = (numeric >= float(self.min_count)).sum(axis=0) >= int(self.min_samples)
        self.keep_features_ = keep_mask[keep_mask].index.tolist()

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


@dataclass
class CPMNormalizer(BasePreprocessor):
    scale_factor: float = 1_000_000.0

    def _transform_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        numeric = frame.astype(float)
        library_sizes = numeric.sum(axis=1).replace(0.0, np.nan)
        cpm = numeric.div(library_sizes, axis=0) * float(self.scale_factor)
        cpm = cpm.fillna(0.0)
        return pd.DataFrame(cpm.to_numpy(dtype=float), index=frame.index.copy(), columns=frame.columns.copy())


@dataclass
class LogCPMTransform(BasePreprocessor):
    scale_factor: float = 1_000_000.0
    prior_count: float = 1.0

    def _transform_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        numeric = frame.astype(float)
        if (numeric < 0).any().any():
            raise ValueError("LogCPMTransform requires non-negative values.")
        library_sizes = numeric.sum(axis=1).replace(0.0, np.nan)
        cpm = numeric.div(library_sizes, axis=0) * float(self.scale_factor)
        transformed = np.log2(cpm.fillna(0.0) + float(self.prior_count))
        return pd.DataFrame(
            transformed.to_numpy(dtype=float),
            index=frame.index.copy(),
            columns=frame.columns.copy(),
        )


@dataclass
class BulkRNASeqPreprocessor:
    min_count: float = 10.0
    min_samples: int = 1
    cpm_target: str = "cpm"
    log_cpm_target: str = "log_cpm"
    make_zscore: bool = False
    zscore_target: str = "zscore_log_cpm"
    prior_count: float = 1.0
    counts_raw_target: str = "counts_raw"
    counts_filtered_target: str = "counts_filtered"

    def fit(self, dataset: CanonicalDataset) -> "BulkRNASeqPreprocessor":
        self._filter = FilterLowExpression(min_count=self.min_count, min_samples=self.min_samples)
        self._filter.fit(dataset)
        return self

    def transform(self, dataset: CanonicalDataset) -> CanonicalDataset:
        if not hasattr(self, "_filter"):
            self.fit(dataset)

        filtered = self._filter.transform(dataset)
        out = CanonicalDataset(
            X=filtered.X.copy(),
            obs=filtered.obs.copy(),
            var=filtered.var.copy(),
            layers={
                self.counts_raw_target: filtered.X.copy(),
                self.counts_filtered_target: filtered.X.copy(),
            },
            provenance=dict(filtered.provenance),
            dataset_id=filtered.dataset_id,
            embeddings={name: frame.copy() for name, frame in filtered.embeddings.items()},
            feature_scores={name: frame.copy() for name, frame in filtered.feature_scores.items()},
            assignments={name: series.copy() for name, series in filtered.assignments.items()},
        )
        out = CPMNormalizer(target=self.cpm_target).fit_transform(out)
        out = LogCPMTransform(target=self.log_cpm_target, prior_count=self.prior_count).fit_transform(out)
        if self.make_zscore:
            out = ZScoreScaler(source_layer=self.log_cpm_target, target=self.zscore_target).fit_transform(out)
        return out

    def fit_transform(self, dataset: CanonicalDataset) -> CanonicalDataset:
        return self.fit(dataset).transform(dataset)
