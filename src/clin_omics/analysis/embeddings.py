from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.decomposition import FactorAnalysis, PCA

from clin_omics.dataset import CanonicalDataset


@dataclass
class PCAEmbedding:
    n_components: int = 2
    source_layer: str | None = None
    key: str = "pca"

    def fit_transform(self, dataset: CanonicalDataset) -> CanonicalDataset:
        frame = dataset.layers[self.source_layer] if self.source_layer else dataset.X
        model = PCA(n_components=self.n_components)
        scores = model.fit_transform(frame.to_numpy())
        components = [f"PC{i+1}" for i in range(scores.shape[1])]
        embedding = pd.DataFrame(scores, index=frame.index, columns=components)
        loadings = pd.DataFrame(
            model.components_.T,
            index=frame.columns,
            columns=components,
        )
        new_embeddings = {name: value.copy() for name, value in dataset.embeddings.items()}
        new_feature_scores = {
            name: value.copy() for name, value in dataset.feature_scores.items()
        }
        new_embeddings[self.key] = embedding
        new_feature_scores[f"{self.key}_loadings"] = loadings
        return CanonicalDataset(
            X=dataset.X.copy(),
            obs=dataset.obs.copy(),
            var=dataset.var.copy(),
            layers={name: layer.copy() for name, layer in dataset.layers.items()},
            provenance=dict(dataset.provenance),
            dataset_id=dataset.dataset_id,
            embeddings=new_embeddings,
            feature_scores=new_feature_scores,
            assignments={name: value.copy() for name, value in dataset.assignments.items()},
        )


@dataclass
class FactorAnalysisEmbedding:
    n_components: int = 2
    source_layer: str | None = None
    key: str = "factor_analysis"

    def fit_transform(self, dataset: CanonicalDataset) -> CanonicalDataset:
        frame = dataset.layers[self.source_layer] if self.source_layer else dataset.X
        model = FactorAnalysis(n_components=self.n_components)
        scores = model.fit_transform(frame.to_numpy())
        components = [f"Factor{i+1}" for i in range(scores.shape[1])]
        embedding = pd.DataFrame(scores, index=frame.index, columns=components)
        loadings = pd.DataFrame(model.components_.T, index=frame.columns, columns=components)
        new_embeddings = {name: value.copy() for name, value in dataset.embeddings.items()}
        new_feature_scores = {
            name: value.copy() for name, value in dataset.feature_scores.items()
        }
        new_embeddings[self.key] = embedding
        new_feature_scores[f"{self.key}_loadings"] = loadings
        return CanonicalDataset(
            X=dataset.X.copy(),
            obs=dataset.obs.copy(),
            var=dataset.var.copy(),
            layers={name: layer.copy() for name, layer in dataset.layers.items()},
            provenance=dict(dataset.provenance),
            dataset_id=dataset.dataset_id,
            embeddings=new_embeddings,
            feature_scores=new_feature_scores,
            assignments={name: value.copy() for name, value in dataset.assignments.items()},
        )
