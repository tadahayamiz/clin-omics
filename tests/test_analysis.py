import pandas as pd

from clin_omics.analysis import (
    FactorAnalysisEmbedding,
    HierarchicalClustering,
    KMeansClustering,
    PCAEmbedding,
    UMAPEmbedding,
    summarize_dataset_qc,
)
from clin_omics.dataset import CanonicalDataset
from clin_omics.workflows import run_unsupervised_workflow


def make_dataset() -> CanonicalDataset:
    X = pd.DataFrame(
        {
            "f1": [0.0, 0.1, 5.0, 5.2],
            "f2": [0.0, 0.2, 4.8, 5.1],
            "f3": [1.0, 1.1, 0.9, 1.0],
        },
        index=["s1", "s2", "s3", "s4"],
    )
    obs = pd.DataFrame({"sample_id": ["s1", "s2", "s3", "s4"]})
    var = pd.DataFrame({"feature_id": ["f1", "f2", "f3"]})
    return CanonicalDataset(X=X, obs=obs, var=var)


def test_pca_embedding_adds_embedding_and_loadings() -> None:
    dataset = make_dataset()
    result = PCAEmbedding(n_components=2, key="pca").fit_transform(dataset)

    assert "pca" in result.embeddings
    assert "pca_loadings" in result.feature_scores
    assert result.embeddings["pca"].shape == (4, 2)
    assert result.feature_scores["pca_loadings"].shape == (3, 2)


def test_factor_analysis_adds_embedding_and_loadings() -> None:
    dataset = make_dataset()
    result = FactorAnalysisEmbedding(n_components=2).fit_transform(dataset)

    assert "factor_analysis" in result.embeddings
    assert "factor_analysis_loadings" in result.feature_scores


def test_kmeans_clustering_uses_embedding() -> None:
    dataset = PCAEmbedding(n_components=2, key="pca").fit_transform(make_dataset())
    result = KMeansClustering(n_clusters=2, embedding_key="pca").fit_predict(dataset)

    assert "cluster_kmeans" in result.assignments
    assert result.assignments["cluster_kmeans"].index.tolist() == ["s1", "s2", "s3", "s4"]


def test_hierarchical_clustering_uses_X_by_default() -> None:
    dataset = make_dataset()
    result = HierarchicalClustering(n_clusters=2).fit_predict(dataset)

    assert "cluster_hierarchical" in result.assignments
    assert len(result.assignments["cluster_hierarchical"].unique()) == 2


def test_qc_summary_reports_counts() -> None:
    dataset = PCAEmbedding(n_components=2).fit_transform(make_dataset())
    dataset = KMeansClustering(n_clusters=2).fit_predict(dataset)
    summary = summarize_dataset_qc(dataset)

    assert summary.loc[0, "n_samples"] == 4
    assert summary.loc[0, "n_embeddings"] == 1
    assert summary.loc[0, "n_assignments"] == 1


def test_unsupervised_workflow_runs_end_to_end() -> None:
    dataset = make_dataset()
    result = run_unsupervised_workflow(dataset, n_components=2, n_clusters=2)

    assert "pca" in result.embeddings
    assert "cluster_kmeans" in result.assignments


class _DummyUMAP:
    def __init__(self, n_components: int, n_neighbors: int, min_dist: float, random_state: int | None):
        self.n_components = n_components
        self.n_neighbors = n_neighbors
        self.min_dist = min_dist
        self.random_state = random_state

    def fit_transform(self, matrix):
        return matrix[:, : self.n_components]


def test_umap_embedding_adds_embedding_from_X(monkeypatch) -> None:
    monkeypatch.setattr("clin_omics.analysis.embeddings.UMAP", _DummyUMAP)

    dataset = make_dataset()
    result = UMAPEmbedding(n_components=2, random_state=0, n_neighbors=2).fit_transform(dataset)

    assert "umap" in result.embeddings
    assert result.embeddings["umap"].shape == (4, 2)
    assert result.embeddings["umap"].index.tolist() == ["s1", "s2", "s3", "s4"]
    assert result.embeddings["umap"].columns.tolist() == ["UMAP1", "UMAP2"]


def test_umap_embedding_uses_source_layer(monkeypatch) -> None:
    monkeypatch.setattr("clin_omics.analysis.embeddings.UMAP", _DummyUMAP)

    dataset = PCAEmbedding(n_components=2, key="pca").fit_transform(make_dataset())
    result = UMAPEmbedding(
        n_components=2,
        source_layer="pca",
        key="umap_from_pca",
        random_state=0,
        n_neighbors=2,
    ).fit_transform(dataset)

    assert "umap_from_pca" in result.embeddings
    assert result.embeddings["umap_from_pca"].shape == (4, 2)
    assert result.embeddings["umap_from_pca"].index.tolist() == dataset.embeddings["pca"].index.tolist()
