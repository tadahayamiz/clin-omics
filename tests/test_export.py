import sys
import types

import pandas as pd
import pytest

from clin_omics.analysis import KMeansClustering, PCAEmbedding
from clin_omics.dataset import CanonicalDataset
from clin_omics.export import (
    export_assignments_table,
    export_embedding_table,
    export_feature_scores_table,
    export_obs_table,
    export_var_table,
    to_anndata,
)


def make_dataset() -> CanonicalDataset:
    X = pd.DataFrame(
        {
            "f1": [0.0, 0.1, 5.0, 5.2],
            "f2": [0.0, 0.2, 4.8, 5.1],
            "f3": [1.0, 1.1, 0.9, 1.0],
        },
        index=["s1", "s2", "s3", "s4"],
    )
    obs = pd.DataFrame({"sample_id": ["s1", "s2", "s3", "s4"], "group": ["a", "a", "b", "b"]})
    var = pd.DataFrame({"feature_id": ["f1", "f2", "f3"]})
    layers = {"normalized": X / 10.0}
    dataset = CanonicalDataset(X=X, obs=obs, var=var, layers=layers)
    dataset = PCAEmbedding(n_components=2, key="pca").fit_transform(dataset)
    dataset = KMeansClustering(n_clusters=2, embedding_key="pca").fit_predict(dataset)
    return dataset


def test_table_exports_return_expected_shapes() -> None:
    dataset = make_dataset()

    obs = export_obs_table(dataset)
    var = export_var_table(dataset)
    embedding = export_embedding_table(dataset, "pca")
    loadings = export_feature_scores_table(dataset, "pca_loadings")
    assignment = export_assignments_table(dataset, "cluster_kmeans")

    assert obs.shape == (4, 2)
    assert var.shape == (3, 1)
    assert embedding.columns.tolist() == ["sample_id", "PC1", "PC2"]
    assert loadings.columns.tolist() == ["feature_id", "PC1", "PC2"]
    assert assignment.columns.tolist() == ["sample_id", "cluster_kmeans"]


def test_to_anndata_raises_without_optional_dependency(monkeypatch: pytest.MonkeyPatch) -> None:
    dataset = make_dataset()
    monkeypatch.setitem(sys.modules, "anndata", None)
    with pytest.raises(ImportError, match="anndata is not installed"):
        to_anndata(dataset)


def test_to_anndata_exports_expected_fields(monkeypatch: pytest.MonkeyPatch) -> None:
    dataset = make_dataset()

    class FakeAnnData:
        def __init__(self, X, obs, var):
            self.X = X
            self.obs = obs
            self.var = var
            self.layers = {}
            self.obsm = {}
            self.varm = {}
            self.uns = {}

    fake_module = types.SimpleNamespace(AnnData=FakeAnnData)
    monkeypatch.setitem(sys.modules, "anndata", fake_module)

    adata = to_anndata(dataset)

    assert list(adata.obs.index) == ["s1", "s2", "s3", "s4"]
    assert list(adata.var.index) == ["f1", "f2", "f3"]
    assert "normalized" in adata.layers
    assert "pca" in adata.obsm
    assert "pca_loadings" in adata.varm
    assert "cluster_kmeans" in adata.obs.columns
    assert adata.uns["dataset_id"] == dataset.dataset_id
