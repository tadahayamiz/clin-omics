from __future__ import annotations

from dataclasses import dataclass
import importlib

import networkx as nx
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.neighbors import NearestNeighbors

from clin_omics.dataset import CanonicalDataset


@dataclass
class KMeansClustering:
    n_clusters: int = 2
    embedding_key: str | None = None
    source_layer: str | None = None
    key: str = "cluster_kmeans"
    random_state: int = 0

    def fit_predict(self, dataset: CanonicalDataset) -> CanonicalDataset:
        frame = _get_source_frame(dataset, self.embedding_key, self.source_layer)
        model = KMeans(n_clusters=self.n_clusters, n_init=10, random_state=self.random_state)
        labels = pd.Series(model.fit_predict(frame.to_numpy()), index=frame.index, name=self.key)
        new_assignments = {name: value.copy() for name, value in dataset.assignments.items()}
        new_assignments[self.key] = labels
        return CanonicalDataset(
            X=dataset.X.copy(),
            obs=dataset.obs.copy(),
            var=dataset.var.copy(),
            layers={name: layer.copy() for name, layer in dataset.layers.items()},
            provenance=dict(dataset.provenance),
            dataset_id=dataset.dataset_id,
            embeddings={name: value.copy() for name, value in dataset.embeddings.items()},
            feature_scores={name: value.copy() for name, value in dataset.feature_scores.items()},
            assignments=new_assignments,
        )


@dataclass
class HierarchicalClustering:
    n_clusters: int = 2
    embedding_key: str | None = None
    source_layer: str | None = None
    key: str = "cluster_hierarchical"
    linkage: str = "ward"

    def fit_predict(self, dataset: CanonicalDataset) -> CanonicalDataset:
        frame = _get_source_frame(dataset, self.embedding_key, self.source_layer)
        model = AgglomerativeClustering(n_clusters=self.n_clusters, linkage=self.linkage)
        labels = pd.Series(model.fit_predict(frame.to_numpy()), index=frame.index, name=self.key)
        new_assignments = {name: value.copy() for name, value in dataset.assignments.items()}
        new_assignments[self.key] = labels
        return CanonicalDataset(
            X=dataset.X.copy(),
            obs=dataset.obs.copy(),
            var=dataset.var.copy(),
            layers={name: layer.copy() for name, layer in dataset.layers.items()},
            provenance=dict(dataset.provenance),
            dataset_id=dataset.dataset_id,
            embeddings={name: value.copy() for name, value in dataset.embeddings.items()},
            feature_scores={name: value.copy() for name, value in dataset.feature_scores.items()},
            assignments=new_assignments,
        )


def _get_source_frame(
    dataset: CanonicalDataset, embedding_key: str | None, source_layer: str | None
) -> pd.DataFrame:
    if embedding_key is not None:
        try:
            return dataset.embeddings[embedding_key]
        except KeyError as exc:
            raise KeyError(f"Embedding '{embedding_key}' not found.") from exc
    if source_layer is not None:
        try:
            return dataset.layers[source_layer]
        except KeyError as exc:
            raise KeyError(f"Layer '{source_layer}' not found.") from exc
    return dataset.X


@dataclass
class KNNLeidenClustering:
    n_neighbors: int = 15
    resolution: float = 1.0
    embedding_key: str | None = None
    source_layer: str | None = None
    key: str = "cluster_knn_leiden"
    random_state: int = 0

    def fit_predict(self, dataset: CanonicalDataset) -> CanonicalDataset:
        frame = _get_source_frame(dataset, self.embedding_key, self.source_layer)
        labels = _run_knn_leiden(
            frame=frame,
            n_neighbors=self.n_neighbors,
            resolution=self.resolution,
            random_state=self.random_state,
            key=self.key,
        )
        new_assignments = {name: value.copy() for name, value in dataset.assignments.items()}
        new_assignments[self.key] = labels
        return CanonicalDataset(
            X=dataset.X.copy(),
            obs=dataset.obs.copy(),
            var=dataset.var.copy(),
            layers={name: layer.copy() for name, layer in dataset.layers.items()},
            provenance=dict(dataset.provenance),
            dataset_id=dataset.dataset_id,
            embeddings={name: value.copy() for name, value in dataset.embeddings.items()},
            feature_scores={name: value.copy() for name, value in dataset.feature_scores.items()},
            assignments=new_assignments,
        )


def _run_knn_leiden(
    frame: pd.DataFrame,
    n_neighbors: int,
    resolution: float,
    random_state: int,
    key: str,
) -> pd.Series:
    try:
        igraph = importlib.import_module("igraph")
        leidenalg = importlib.import_module("leidenalg")
    except ModuleNotFoundError as exc:  # pragma: no cover - exercised in CLI tests via monkeypatch
        raise ImportError(
            "KNN->Leiden clustering requires optional dependencies 'igraph' and 'leidenalg'."
        ) from exc

    graph = _build_knn_graph(frame, n_neighbors)
    edges = list(graph.edges())
    ig_graph = igraph.Graph(n=frame.shape[0], edges=edges, directed=False)
    weights = [float(graph[u][v]["weight"]) for u, v in edges]
    partition = leidenalg.find_partition(
        ig_graph,
        leidenalg.RBConfigurationVertexPartition,
        weights=weights,
        resolution_parameter=resolution,
        seed=random_state,
    )
    return pd.Series(partition.membership, index=frame.index, name=key)


def _build_knn_graph(frame: pd.DataFrame, n_neighbors: int) -> nx.Graph:
    if frame.shape[0] < 2:
        raise ValueError("KNN graph requires at least 2 samples.")
    effective_neighbors = min(max(1, n_neighbors), frame.shape[0] - 1)
    nn = NearestNeighbors(n_neighbors=effective_neighbors + 1)
    nn.fit(frame.to_numpy())
    distances, indices = nn.kneighbors(frame.to_numpy())

    graph = nx.Graph()
    graph.add_nodes_from(range(frame.shape[0]))
    for source_idx, (row_distances, row_indices) in enumerate(zip(distances, indices, strict=False)):
        for dist, target_idx in zip(row_distances[1:], row_indices[1:], strict=False):
            weight = 1.0 / (1.0 + float(dist))
            if graph.has_edge(source_idx, int(target_idx)):
                if weight > graph[source_idx][int(target_idx)]["weight"]:
                    graph[source_idx][int(target_idx)]["weight"] = weight
            else:
                graph.add_edge(source_idx, int(target_idx), weight=weight)
    return graph
