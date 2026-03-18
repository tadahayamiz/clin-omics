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


def _make_preprocessed_dataset() -> CanonicalDataset:
    obs = pd.DataFrame({"sample_id": ["s1", "s2", "s3", "s4"], "group": ["A", "A", "B", "B"]})
    var = pd.DataFrame({"feature_id": ["g1", "g2", "g3", "g4"], "feature_name": ["Gene1", "Gene2", "Gene3", "Gene4"]})
    X = pd.DataFrame(
        [[100.0, 80.0, 3.0, 2.0], [110.0, 90.0, 4.0, 3.0], [5.0, 4.0, 120.0, 100.0], [6.0, 5.0, 130.0, 110.0]],
        index=["s1", "s2", "s3", "s4"],
        columns=["g1", "g2", "g3", "g4"],
    )
    log_cpm = pd.DataFrame(
        [[8.0, 7.8, 1.0, 0.8], [8.2, 8.0, 1.2, 1.0], [1.0, 0.9, 8.1, 7.9], [1.1, 1.0, 8.3, 8.0]],
        index=X.index,
        columns=X.columns,
    )
    return CanonicalDataset(X=X, obs=obs, var=var, layers={"log_cpm": log_cpm})


def test_bulk_rnaseq_graph_from_h5_flow_runs_with_fake_leiden_modules(tmp_path, monkeypatch) -> None:
    from clin_omics.workflows.bulk_rnaseq_graph_from_h5 import main

    dataset = _make_preprocessed_dataset()
    in_path = tmp_path / "preprocessed.h5"
    outdir = tmp_path / "graph_from_h5_out"
    dataset.save_h5(in_path)

    real_import_module = importlib.import_module

    def fake_import_module(name, package=None):
        if name == "igraph":
            return FakeIGraphModule
        if name == "leidenalg":
            return FakeLeidenAlg
        return real_import_module(name, package)

    monkeypatch.setattr(importlib, "import_module", fake_import_module)

    exit_code = main([
        "--dataset-h5", str(in_path),
        "--outdir", str(outdir),
        "--source-layer", "log_cpm",
        "--n-pca-components", "2",
        "--leiden-neighbors", "2",
        "--leiden-resolution", "1.0",
    ])

    assert exit_code == 0
    assert (outdir / "bulk_rnaseq_graph_from_h5_output_dataset.h5").exists()
    assert (outdir / "bulk_rnaseq_graph_from_h5_summary.json").exists()
    assert (outdir / "cluster_leiden_graph.csv").exists()
    assert (outdir / "pca_graph.png").exists()
    assert (outdir / "pca_graph.svg").exists()

    summary = json.loads((outdir / "bulk_rnaseq_graph_from_h5_summary.json").read_text(encoding="utf-8"))
    assert summary["input_dataset"] == str(in_path)
    assert summary["source_layer"] == "log_cpm"
    assert summary["plot_color"] == "cluster_leiden_graph"
    assert "pca_graph" in summary["embeddings"]
    assert "cluster_leiden_graph" in summary["assignments"]

    processed = CanonicalDataset.load_h5(outdir / "bulk_rnaseq_graph_from_h5_output_dataset.h5")
    assert "cluster_leiden_graph" in processed.assignments
    assert processed.assignments["cluster_leiden_graph"].tolist() == [0, 1, 0, 1]
