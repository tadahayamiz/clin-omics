from __future__ import annotations

from clin_omics.analysis import HierarchicalClustering, KMeansClustering, PCAEmbedding
from clin_omics.dataset import CanonicalDataset


def run_unsupervised_workflow(
    dataset: CanonicalDataset,
    *,
    n_components: int = 2,
    n_clusters: int = 2,
    embedding_key: str = "pca",
    cluster_key: str = "cluster_kmeans",
    source_layer: str | None = None,
) -> CanonicalDataset:
    dataset_with_embedding = PCAEmbedding(
        n_components=n_components,
        source_layer=source_layer,
        key=embedding_key,
    ).fit_transform(dataset)
    return KMeansClustering(
        n_clusters=n_clusters,
        embedding_key=embedding_key,
        key=cluster_key,
    ).fit_predict(dataset_with_embedding)


__all__ = ["run_unsupervised_workflow", "HierarchicalClustering", "KMeansClustering"]
