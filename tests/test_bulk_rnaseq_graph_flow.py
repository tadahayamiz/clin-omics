from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from clin_omics.dataset import CanonicalDataset


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


def _write_toy_tables(tmp_path: Path) -> tuple[Path, Path, Path]:
    X = pd.DataFrame(
        {
            "sample_id": ["s1", "s2", "s3", "s4"],
            "g1": [100, 110, 5, 6],
            "g2": [80, 90, 4, 5],
            "g3": [3, 4, 120, 130],
            "g4": [2, 3, 100, 110],
        }
    )
    obs = pd.DataFrame(
        {
            "sample_id": ["s1", "s2", "s3", "s4"],
            "group": ["A", "A", "B", "B"],
        }
    )
    var = pd.DataFrame(
        {
            "feature_id": ["g1", "g2", "g3", "g4"],
            "feature_name": ["Gene1", "Gene2", "Gene3", "Gene4"],
        }
    )
    x_path = tmp_path / "X.csv"
    obs_path = tmp_path / "obs.csv"
    var_path = tmp_path / "var.csv"
    X.to_csv(x_path, index=False)
    obs.to_csv(obs_path, index=False)
    var.to_csv(var_path, index=False)
    return x_path, obs_path, var_path


def test_bulk_rnaseq_graph_flow_runs_with_fake_leiden_modules(tmp_path, monkeypatch) -> None:
    from clin_omics.workflows.bulk_rnaseq_graph import main

    x_path, obs_path, var_path = _write_toy_tables(tmp_path)
    outdir = tmp_path / "graph_out"

    real_import_module = importlib.import_module

    def fake_import_module(name, package=None):
        if name == "igraph":
            return FakeIGraphModule
        if name == "leidenalg":
            return FakeLeidenAlg
        return real_import_module(name, package)

    monkeypatch.setattr(importlib, "import_module", fake_import_module)

    exit_code = main([
        "--x", str(x_path),
        "--obs", str(obs_path),
        "--var", str(var_path),
        "--outdir", str(outdir),
        "--min-count", "1",
        "--min-samples", "1",
        "--n-pca-components", "2",
        "--leiden-neighbors", "2",
        "--leiden-resolution", "1.0",
    ])

    assert exit_code == 0
    assert (outdir / "bulk_rnaseq_graph_input_dataset.h5").exists()
    assert (outdir / "bulk_rnaseq_graph_processed_dataset.h5").exists()
    assert (outdir / "bulk_rnaseq_graph_summary.json").exists()
    assert (outdir / "cluster_leiden_graph.csv").exists()
    assert (outdir / "pca_graph.png").exists()
    assert (outdir / "pca_graph.svg").exists()

    summary = json.loads((outdir / "bulk_rnaseq_graph_summary.json").read_text(encoding="utf-8"))
    assert summary["run_umap"] is False
    assert summary["plot_color"] == "cluster_leiden_graph"
    assert "pca_graph" in summary["embeddings"]
    assert "cluster_leiden_graph" in summary["assignments"]

    processed = CanonicalDataset.load_h5(outdir / "bulk_rnaseq_graph_processed_dataset.h5")
    assert "cluster_leiden_graph" in processed.assignments
    assert processed.assignments["cluster_leiden_graph"].tolist() == [0, 1, 0, 1]


def test_bulk_rnaseq_graph_flow_rejects_missing_leiden_dependencies(tmp_path, monkeypatch) -> None:
    from clin_omics.workflows.bulk_rnaseq_graph import main

    x_path, obs_path, var_path = _write_toy_tables(tmp_path)
    outdir = tmp_path / "graph_out_missing"

    real_import_module = importlib.import_module

    def fake_import_module(name, package=None):
        if name in {"igraph", "leidenalg"}:
            raise ModuleNotFoundError(name)
        return real_import_module(name, package)

    monkeypatch.setattr(importlib, "import_module", fake_import_module)

    try:
        main([
            "--x", str(x_path),
            "--obs", str(obs_path),
            "--var", str(var_path),
            "--outdir", str(outdir),
            "--min-count", "1",
            "--min-samples", "1",
            "--n-pca-components", "2",
            "--leiden-neighbors", "2",
            "--leiden-resolution", "1.0",
        ])
    except ImportError as exc:
        assert "igraph" in str(exc)
        assert "leidenalg" in str(exc)
    else:
        raise AssertionError("Expected ImportError when optional Leiden dependencies are unavailable.")
