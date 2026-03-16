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
    assign = pd.Series([0, 1], index=X.index, name="kmeans")
    return CanonicalDataset(
        X=X,
        obs=obs,
        var=var,
        layers={"scaled": layer},
        embeddings={"X_pca": emb},
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
    assert payload["layers"] == ["scaled"]
    assert payload["embeddings"] == ["X_pca"]
    assert payload["assignments"] == ["cluster_kmeans"]


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
    X = pd.DataFrame({"wrong_id": ["s1"], "f1": [1.0]})
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
