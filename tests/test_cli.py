from __future__ import annotations

import json

import pandas as pd
import pytest

from clin_omics.cli import main
from clin_omics.dataset import CanonicalDataset


def _make_dataset() -> CanonicalDataset:
    obs = pd.DataFrame({"sample_id": ["s1", "s2"], "group": ["A", "B"]})
    var = pd.DataFrame({"feature_id": ["f1", "f2"], "feature_name": ["g1", "g2"]})
    X = pd.DataFrame([[1.0, 2.0], [3.0, 4.0]], index=["s1", "s2"], columns=["f1", "f2"])
    layer = pd.DataFrame([[0.0, 1.0], [1.0, 2.0]], index=X.index, columns=X.columns)
    emb = pd.DataFrame([[0.1, 0.2], [0.3, 0.4]], index=X.index, columns=["PC1", "PC2"])
    feature_scores = pd.DataFrame([[0.9, 0.1], [0.2, 0.8]], index=["f1", "f2"], columns=["PC1", "PC2"])
    assign = pd.Series([0, 1], index=X.index, name="kmeans")
    return CanonicalDataset(
        X=X,
        obs=obs,
        var=var,
        layers={"scaled": layer},
        embeddings={"X_pca": emb},
        feature_scores={"X_pca_loadings": feature_scores},
        assignments={"cluster_kmeans": assign},
    )


def test_cli_version(capsys) -> None:
    exit_code = main(["version"])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert captured.out.strip()


def test_cli_inspect(tmp_path, capsys) -> None:
    dataset = _make_dataset()
    path = tmp_path / "toy.h5"
    dataset.save_h5(path)

    exit_code = main(["inspect", str(path)])
    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert exit_code == 0
    assert payload["shape"] == [2, 2]
    assert payload["n_samples"] == 2
    assert payload["n_features"] == 2
    assert payload["layers"] == ["scaled"]
    assert payload["layer_shapes"] == {"scaled": [2, 2]}
    assert payload["embeddings"] == ["X_pca"]
    assert payload["embedding_shapes"] == {"X_pca": [2, 2]}
    assert payload["feature_scores"] == ["X_pca_loadings"]
    assert payload["feature_score_shapes"] == {"X_pca_loadings": [2, 2]}
    assert payload["assignments"] == ["cluster_kmeans"]
    assert payload["assignment_lengths"] == {"cluster_kmeans": 2}


def test_cli_validate_success(tmp_path, capsys) -> None:
    dataset = _make_dataset()
    path = tmp_path / "toy.h5"
    dataset.save_h5(path)

    exit_code = main(["validate", str(path)])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert captured.out.strip() == "VALID"


def test_cli_validate_failure(tmp_path, capsys) -> None:
    bad_path = tmp_path / "bad.h5"
    bad_path.write_text("not an hdf5 file")

    exit_code = main(["validate", str(bad_path)])
    captured = capsys.readouterr()

    assert exit_code == 1
    assert "INVALID:" in captured.err


def test_cli_build_dataset(tmp_path, capsys) -> None:
    X = pd.DataFrame(
        {
            "sample_id": ["s1", "s2"],
            "f1": [1.0, 3.0],
            "f2": [2.0, 4.0],
        }
    )
    obs = pd.DataFrame({"sample_id": ["s1", "s2"], "group": ["A", "B"]})
    var = pd.DataFrame({"feature_id": ["f1", "f2"], "feature_name": ["g1", "g2"]})

    x_path = tmp_path / "X.csv"
    obs_path = tmp_path / "obs.csv"
    var_path = tmp_path / "var.csv"
    out_path = tmp_path / "dataset.h5"

    X.to_csv(x_path, index=False)
    obs.to_csv(obs_path, index=False)
    var.to_csv(var_path, index=False)

    exit_code = main([
        "build-dataset",
        "--x",
        str(x_path),
        "--obs",
        str(obs_path),
        "--var",
        str(var_path),
        "--out",
        str(out_path),
    ])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert out_path.exists()
    assert str(out_path) in captured.out

    dataset = CanonicalDataset.load_h5(out_path)
    assert dataset.X.shape == (2, 2)
    assert dataset.obs["sample_id"].tolist() == ["s1", "s2"]
    assert dataset.var["feature_id"].tolist() == ["f1", "f2"]


def test_cli_build_dataset_failure(tmp_path, capsys) -> None:
    X = pd.DataFrame({"wrong_id": ["s1", "s1"], "f1": [1.0, 2.0]})
    obs = pd.DataFrame({"sample_id": ["s1"]})
    var = pd.DataFrame({"feature_id": ["f1"]})

    x_path = tmp_path / "X.csv"
    obs_path = tmp_path / "obs.csv"
    var_path = tmp_path / "var.csv"
    out_path = tmp_path / "dataset.h5"

    X.to_csv(x_path, index=False)
    obs.to_csv(obs_path, index=False)
    var.to_csv(var_path, index=False)

    exit_code = main([
        "build-dataset",
        "--x",
        str(x_path),
        "--obs",
        str(obs_path),
        "--var",
        str(var_path),
        "--out",
        str(out_path),
    ])
    captured = capsys.readouterr()

    assert exit_code == 1
    assert "BUILD_FAILED:" in captured.err


def test_cli_pca(tmp_path, capsys) -> None:
    dataset = _make_dataset()
    in_path = tmp_path / "toy.h5"
    out_path = tmp_path / "toy_pca.h5"
    dataset.save_h5(in_path)

    exit_code = main([
        "pca",
        "--in",
        str(in_path),
        "--out",
        str(out_path),
        "--n-components",
        "2",
        "--key",
        "X_pca_cli",
    ])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert out_path.exists()
    assert str(out_path) in captured.out

    updated = CanonicalDataset.load_h5(out_path)
    assert "X_pca_cli" in updated.embeddings
    assert updated.embeddings["X_pca_cli"].shape == (2, 2)


def test_cli_factor_analysis(tmp_path, capsys) -> None:
    dataset = _make_dataset()
    in_path = tmp_path / "toy.h5"
    out_path = tmp_path / "toy_factor_analysis.h5"
    dataset.save_h5(in_path)

    exit_code = main([
        "factor-analysis",
        "--in",
        str(in_path),
        "--out",
        str(out_path),
        "--n-components",
        "2",
        "--key",
        "factor_analysis_cli",
    ])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert out_path.exists()
    assert str(out_path) in captured.out

    updated = CanonicalDataset.load_h5(out_path)
    assert "factor_analysis_cli" in updated.embeddings
    assert updated.embeddings["factor_analysis_cli"].shape == (2, 2)
    assert "factor_analysis_cli_loadings" in updated.feature_scores
    assert updated.feature_scores["factor_analysis_cli_loadings"].shape == (2, 2)


def test_cli_umap(tmp_path, capsys, monkeypatch) -> None:
    dataset = _make_dataset()
    in_path = tmp_path / "toy.h5"
    out_path = tmp_path / "toy_umap.h5"
    dataset.save_h5(in_path)

    class FakeUMAP:
        def __init__(self, n_components, n_neighbors, min_dist, random_state):
            self.n_components = n_components
            self.n_neighbors = n_neighbors
            self.min_dist = min_dist
            self.random_state = random_state

        def fit_transform(self, X):
            import numpy as np
            arr = np.asarray(X, dtype=float)
            return arr[:, : self.n_components] + 1.0

    monkeypatch.setattr("clin_omics.analysis.embeddings.UMAP", FakeUMAP)

    exit_code = main([
        "umap",
        "--in",
        str(in_path),
        "--out",
        str(out_path),
        "--embedding-key",
        "X_pca",
        "--n-components",
        "2",
        "--key",
        "umap_cli",
        "--random-state",
        "0",
    ])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert out_path.exists()
    assert str(out_path) in captured.out

    updated = CanonicalDataset.load_h5(out_path)
    assert "umap_cli" in updated.embeddings
    assert updated.embeddings["umap_cli"].shape == (2, 2)
    assert updated.embeddings["umap_cli"].columns.tolist() == ["UMAP1", "UMAP2"]


def test_cli_cluster_kmeans(tmp_path, capsys) -> None:
    dataset = _make_dataset()
    in_path = tmp_path / "toy.h5"
    out_path = tmp_path / "toy_kmeans.h5"
    dataset.save_h5(in_path)

    exit_code = main([
        "cluster-kmeans",
        "--in",
        str(in_path),
        "--out",
        str(out_path),
        "--n-clusters",
        "2",
        "--embedding-key",
        "X_pca",
        "--key",
        "cluster_kmeans_cli",
    ])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert out_path.exists()
    assert str(out_path) in captured.out

    updated = CanonicalDataset.load_h5(out_path)
    assert "cluster_kmeans_cli" in updated.assignments
    assert len(updated.assignments["cluster_kmeans_cli"]) == 2


def test_cli_cluster_knn_leiden_with_fake_modules(tmp_path, capsys, monkeypatch) -> None:
    dataset = _make_dataset()
    in_path = tmp_path / "toy.h5"
    out_path = tmp_path / "toy_knn_leiden.h5"
    dataset.save_h5(in_path)

    class FakeGraph:
        def __init__(self, n, edges, directed=False):
            self.n = n
            self.edges = edges
            self.directed = directed

    class FakePartition:
        def __init__(self, membership):
            self.membership = membership

    class FakeLeidenAlg:
        RBConfigurationVertexPartition = object()

        @staticmethod
        def find_partition(graph, partition_type, weights, resolution_parameter, seed):
            return FakePartition([i % 2 for i in range(graph.n)])

    class FakeIGraphModule:
        Graph = FakeGraph

    import importlib
    real_import_module = importlib.import_module

    def fake_import_module(name, package=None):
        if name == "igraph":
            return FakeIGraphModule
        if name == "leidenalg":
            return FakeLeidenAlg
        return real_import_module(name, package)

    monkeypatch.setattr(importlib, "import_module", fake_import_module)

    exit_code = main([
        "cluster-knn-leiden",
        "--in",
        str(in_path),
        "--out",
        str(out_path),
        "--neighbors",
        "1",
        "--resolution",
        "1.0",
        "--embedding-key",
        "X_pca",
        "--key",
        "cluster_knn_leiden_cli",
    ])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert out_path.exists()
    assert str(out_path) in captured.out

    updated = CanonicalDataset.load_h5(out_path)
    assert "cluster_knn_leiden_cli" in updated.assignments
    assert updated.assignments["cluster_knn_leiden_cli"].tolist() == [0, 1]



def test_cli_build_dataset_parquet(tmp_path, capsys) -> None:
    pytest.importorskip("pyarrow")

    X = pd.DataFrame(
        {
            "sample_id": ["s1", "s2"],
            "f1": [1.0, 3.0],
            "f2": [2.0, 4.0],
        }
    )
    obs = pd.DataFrame({"sample_id": ["s1", "s2"], "group": ["A", "B"]})
    var = pd.DataFrame({"feature_id": ["f1", "f2"], "feature_name": ["g1", "g2"]})

    x_path = tmp_path / "X.parquet"
    obs_path = tmp_path / "obs.parquet"
    var_path = tmp_path / "var.parquet"
    out_path = tmp_path / "dataset_parquet.h5"

    X.to_parquet(x_path, index=False)
    obs.to_parquet(obs_path, index=False)
    var.to_parquet(var_path, index=False)

    exit_code = main([
        "build-dataset",
        "--x",
        str(x_path),
        "--obs",
        str(obs_path),
        "--var",
        str(var_path),
        "--out",
        str(out_path),
    ])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert out_path.exists()
    assert str(out_path) in captured.out


def test_cli_plot_embedding(tmp_path, capsys) -> None:
    dataset = _make_dataset()
    in_path = tmp_path / "toy.h5"
    out_prefix = tmp_path / "fig_embedding"
    dataset.save_h5(in_path)

    exit_code = main([
        "plot-embedding",
        "--in",
        str(in_path),
        "--embedding-key",
        "X_pca",
        "--color",
        "group",
        "--out-prefix",
        str(out_prefix),
    ])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert str(out_prefix) in captured.out
    assert out_prefix.with_suffix(".png").exists()
    assert out_prefix.with_suffix(".svg").exists()


def test_cli_plot_embedding_assignment_color(tmp_path, capsys) -> None:
    dataset = _make_dataset()
    in_path = tmp_path / "toy_assign.h5"
    out_prefix = tmp_path / "fig_embedding_assign"
    dataset.save_h5(in_path)

    exit_code = main([
        "plot-embedding",
        "--in",
        str(in_path),
        "--embedding-key",
        "X_pca",
        "--color",
        "cluster_kmeans",
        "--out-prefix",
        str(out_prefix),
    ])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert str(out_prefix) in captured.out
    assert out_prefix.with_suffix(".png").exists()
    assert out_prefix.with_suffix(".svg").exists()


def test_cli_plot_embedding_failure_unknown_color_key(tmp_path, capsys) -> None:
    dataset = _make_dataset()
    in_path = tmp_path / "toy_missing_color.h5"
    out_prefix = tmp_path / "fig_embedding_missing"
    dataset.save_h5(in_path)

    exit_code = main([
        "plot-embedding",
        "--in",
        str(in_path),
        "--embedding-key",
        "X_pca",
        "--color",
        "missing_key",
        "--out-prefix",
        str(out_prefix),
    ])
    captured = capsys.readouterr()

    assert exit_code == 1
    assert "PLOT_EMBEDDING_FAILED:" in captured.err
    assert "obs column or assignment key" in captured.err


def test_cli_plot_roc(tmp_path, capsys) -> None:
    table = pd.DataFrame({"y_true": [0, 0, 1, 1], "y_score": [0.1, 0.3, 0.7, 0.9]})
    table_path = tmp_path / "roc.csv"
    out_prefix = tmp_path / "fig_roc"
    table.to_csv(table_path, index=False)

    exit_code = main([
        "plot-roc",
        "--input",
        str(table_path),
        "--y-true-col",
        "y_true",
        "--y-score-col",
        "y_score",
        "--out-prefix",
        str(out_prefix),
    ])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert str(out_prefix) in captured.out
    assert out_prefix.with_suffix(".png").exists()
    assert out_prefix.with_suffix(".svg").exists()


def test_cli_plot_confusion_matrix_failure_missing_column(tmp_path, capsys) -> None:
    table = pd.DataFrame({"y_true": [0, 1], "wrong": [0, 1]})
    table_path = tmp_path / "cm.csv"
    out_prefix = tmp_path / "fig_cm"
    table.to_csv(table_path, index=False)

    exit_code = main([
        "plot-confusion-matrix",
        "--input",
        str(table_path),
        "--y-true-col",
        "y_true",
        "--y-pred-col",
        "y_pred",
        "--out-prefix",
        str(out_prefix),
    ])
    captured = capsys.readouterr()

    assert exit_code == 1
    assert "PLOT_CONFUSION_MATRIX_FAILED:" in captured.err


def test_cli_plot_regression_residuals(tmp_path, capsys) -> None:
    table = pd.DataFrame({"y_true": [1.0, 2.0, 3.0], "y_pred": [0.9, 2.1, 2.8]})
    table_path = tmp_path / "resid.csv"
    out_prefix = tmp_path / "fig_resid"
    table.to_csv(table_path, index=False)

    exit_code = main([
        "plot-regression-residuals",
        "--input",
        str(table_path),
        "--y-true-col",
        "y_true",
        "--y-pred-col",
        "y_pred",
        "--out-prefix",
        str(out_prefix),
        "--fontsize",
        "15",
        "--no-svg",
    ])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert str(out_prefix) in captured.out
    assert out_prefix.with_suffix(".png").exists()
    assert not out_prefix.with_suffix(".svg").exists()


def test_cli_export_assignments_csv(tmp_path, capsys) -> None:
    dataset = _make_dataset()
    in_path = tmp_path / "toy.h5"
    out_path = tmp_path / "assignments.csv"
    dataset.save_h5(in_path)

    exit_code = main([
        "export-assignments",
        "--in",
        str(in_path),
        "--key",
        "cluster_kmeans",
        "--out",
        str(out_path),
    ])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert str(out_path) in captured.out
    exported = pd.read_csv(out_path)
    assert exported.columns.tolist() == ["sample_id", "cluster_kmeans"]
    assert exported["sample_id"].tolist() == ["s1", "s2"]
    assert exported["cluster_kmeans"].tolist() == [0, 1]


def test_cli_export_assignments_tsv(tmp_path, capsys) -> None:
    dataset = _make_dataset()
    in_path = tmp_path / "toy.h5"
    out_path = tmp_path / "assignments.tsv"
    dataset.save_h5(in_path)

    exit_code = main([
        "export-assignments",
        "--in",
        str(in_path),
        "--key",
        "cluster_kmeans",
        "--out",
        str(out_path),
    ])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert str(out_path) in captured.out
    exported = pd.read_csv(out_path, sep="\t")
    assert exported.columns.tolist() == ["sample_id", "cluster_kmeans"]
    assert exported["cluster_kmeans"].tolist() == [0, 1]


def test_cli_export_assignments_failure_bad_suffix(tmp_path, capsys) -> None:
    dataset = _make_dataset()
    in_path = tmp_path / "toy.h5"
    out_path = tmp_path / "assignments.json"
    dataset.save_h5(in_path)

    exit_code = main([
        "export-assignments",
        "--in",
        str(in_path),
        "--key",
        "cluster_kmeans",
        "--out",
        str(out_path),
    ])
    captured = capsys.readouterr()

    assert exit_code == 1
    assert "EXPORT_ASSIGNMENTS_FAILED:" in captured.err
    assert not out_path.exists()


def test_cli_plot_embedding_alpha_size_no_legend(tmp_path, capsys) -> None:
    dataset = _make_dataset()
    in_path = tmp_path / "toy_plot_custom.h5"
    out_prefix = tmp_path / "fig_embedding_custom"
    dataset.save_h5(in_path)

    exit_code = main([
        "plot-embedding",
        "--in",
        str(in_path),
        "--embedding-key",
        "X_pca",
        "--color",
        "group",
        "--alpha",
        "0.35",
        "--size",
        "18",
        "--no-legend",
        "--out-prefix",
        str(out_prefix),
    ])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert str(out_prefix) in captured.out
    assert out_prefix.with_suffix(".png").exists()
    assert out_prefix.with_suffix(".svg").exists()


def test_cli_umap_rejects_embedding_key_and_source_layer(tmp_path, capsys) -> None:
    dataset = _make_dataset()
    in_path = tmp_path / "toy.h5"
    out_path = tmp_path / "toy_umap_fail.h5"
    dataset.save_h5(in_path)

    exit_code = main([
        "umap",
        "--in",
        str(in_path),
        "--out",
        str(out_path),
        "--embedding-key",
        "X_pca",
        "--source-layer",
        "scaled",
    ])
    captured = capsys.readouterr()

    assert exit_code == 2
    assert "not allowed with argument" in captured.err


def test_cli_cluster_kmeans_rejects_embedding_key_and_source_layer(tmp_path, capsys) -> None:
    dataset = _make_dataset()
    in_path = tmp_path / "toy.h5"
    out_path = tmp_path / "toy_kmeans_fail.h5"
    dataset.save_h5(in_path)

    exit_code = main([
        "cluster-kmeans",
        "--in",
        str(in_path),
        "--out",
        str(out_path),
        "--n-clusters",
        "2",
        "--embedding-key",
        "X_pca",
        "--source-layer",
        "scaled",
    ])
    captured = capsys.readouterr()

    assert exit_code == 2
    assert "not allowed with argument" in captured.err


def test_cli_cluster_knn_leiden_rejects_embedding_key_and_source_layer(tmp_path, capsys) -> None:
    dataset = _make_dataset()
    in_path = tmp_path / "toy.h5"
    out_path = tmp_path / "toy_knn_leiden_fail.h5"
    dataset.save_h5(in_path)

    exit_code = main([
        "cluster-knn-leiden",
        "--in",
        str(in_path),
        "--out",
        str(out_path),
        "--embedding-key",
        "X_pca",
        "--source-layer",
        "scaled",
    ])
    captured = capsys.readouterr()

    assert exit_code == 2
    assert "not allowed with argument" in captured.err


def test_cli_build_dataset_accepts_index_style_x(tmp_path, capsys) -> None:
    X = pd.DataFrame(
        {
            "f1": [1.0, 3.0],
            "f2": [2.0, 4.0],
        },
        index=["s1", "s2"],
    )
    obs = pd.DataFrame({"sample_id": ["s1", "s2"], "group": ["A", "B"]})
    var = pd.DataFrame({"feature_id": ["f1", "f2"], "feature_name": ["g1", "g2"]})

    x_path = tmp_path / "X.csv"
    obs_path = tmp_path / "obs.csv"
    var_path = tmp_path / "var.csv"
    out_path = tmp_path / "dataset.h5"

    X.to_csv(x_path, index=True)
    obs.to_csv(obs_path, index=False)
    var.to_csv(var_path, index=False)

    exit_code = main([
        "build-dataset",
        "--x",
        str(x_path),
        "--obs",
        str(obs_path),
        "--var",
        str(var_path),
        "--out",
        str(out_path),
    ])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert out_path.exists()
    assert str(out_path) in captured.out

    dataset = CanonicalDataset.load_h5(out_path)
    assert dataset.X.index.tolist() == ["s1", "s2"]
    assert dataset.X.columns.tolist() == ["f1", "f2"]
