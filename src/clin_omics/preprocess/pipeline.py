from __future__ import annotations

from dataclasses import dataclass, field

from clin_omics.dataset import CanonicalDataset
from clin_omics.preprocess.base import BasePreprocessor


@dataclass
class PreprocessPipeline:
    steps: list[tuple[str, BasePreprocessor]] = field(default_factory=list)

    def fit(self, dataset: CanonicalDataset) -> "PreprocessPipeline":
        current = dataset
        for _, step in self.steps:
            step.fit(current)
            current = step.transform(current)
        return self

    def transform(self, dataset: CanonicalDataset) -> CanonicalDataset:
        current = dataset
        for _, step in self.steps:
            current = step.transform(current)
        return current

    def fit_transform(self, dataset: CanonicalDataset) -> CanonicalDataset:
        current = dataset
        for _, step in self.steps:
            current = step.fit_transform(current)
        return current
