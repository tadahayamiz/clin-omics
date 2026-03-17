from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.decomposition import FactorAnalysis, PCA

try:
    from umap import UMAP
except ImportError:  # pragma: no cover
    UMAP = None

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


@dataclass
class UMAPEmbedding:
    n_components: int = 2
    source_layer: str | None = None
    key: str = "umap"
    n_neighbors: int = 15
    min_dist: float = 0.1
    random_state: int | None = 0

    def fit_transform(self, dataset: CanonicalDataset) -> CanonicalDataset:
        if UMAP is None:
            raise ImportError("umap-learn is required to use UMAPEmbedding")

        if self.source_layer is None:
            frame = dataset.X
        elif self.source_layer in dataset.layers:
            frame = dataset.layers[self.source_layer]
        elif self.source_layer in dataset.embeddings:
            frame = dataset.embeddings[self.source_layer]
        else:
            raise KeyError(f"Source layer or embedding '{self.source_layer}' not found.")
        model = UMAP(
            n_components=self.n_components,
            n_neighbors=self.n_neighbors,
            min_dist=self.min_dist,
            random_state=self.random_state,
        )
        scores = model.fit_transform(frame.to_numpy())
        components = [f"UMAP{i+1}" for i in range(scores.shape[1])]
        embedding = pd.DataFrame(scores, index=frame.index, columns=components)
        new_embeddings = {name: value.copy() for name, value in dataset.embeddings.items()}
        new_embeddings[self.key] = embedding
        return CanonicalDataset(
            X=dataset.X.copy(),
            obs=dataset.obs.copy(),
            var=dataset.var.copy(),
            layers={name: layer.copy() for name, layer in dataset.layers.items()},
            provenance=dict(dataset.provenance),
            dataset_id=dataset.dataset_id,
            embeddings=new_embeddings,
            feature_scores={
                name: value.copy() for name, value in dataset.feature_scores.items()
            },
            assignments={name: value.copy() for name, value in dataset.assignments.items()},
        )
