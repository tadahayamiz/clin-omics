from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path

import pandas as pd

from clin_omics.analysis import PCAEmbedding
from clin_omics.dataset import CanonicalDataset
from clin_omics.workflows.plot_embedding_from_h5 import run_plot_embedding_from_h5


def _make_dataset() -> CanonicalDataset:
    X = pd.DataFrame(
        [[1.0, 2.0, 3.0], [2.0, 1.0, 0.0], [3.0, 4.0, 5.0], [4.0, 3.0, 2.0]],
        index=["s1", "s2", "s3", "s4"],
        columns=["f1", "f2", "f3"],
    )
    obs = pd.DataFrame(
        {
            "sample_id": ["s1", "s2", "s3", "s4"],
            "group": ["A", "A", "B", "B"],
            "score": [0.1, 0.2, 0.8, 0.9],
        }
    )
    var = pd.DataFrame({"feature_id": ["f1", "f2", "f3"]})
    ds = CanonicalDataset(X=X, obs=obs, var=var)
    ds = PCAEmbedding(n_components=2, source_layer=None, key="pca_graph").fit_transform(ds)
    ds.embeddings["umap_graph"] = pd.DataFrame(
        {
            "UMAP1": [0.0, 1.0, 0.1, 1.1],
            "UMAP2": [0.0, 0.0, 1.0, 1.0],
        },
        index=ds.X.index,
    )
    ds.assignments["cluster_leiden_graph"] = pd.Series([0, 0, 1, 1], index=ds.X.index, name="cluster_leiden_graph")
    return ds


def test_plot_embedding_from_h5_uses_existing_embedding_and_default_cluster_color(tmp_path: Path) -> None:
    ds = _make_dataset()
    dataset_h5 = tmp_path / "dataset.h5"
    ds.save_h5(dataset_h5)
    outdir = tmp_path / "plots"

    summary = run_plot_embedding_from_h5(
        Namespace(
            dataset_h5=dataset_h5,
            outdir=outdir,
            embedding_key="umap_graph",
            plot_color=None,
            title=None,
            out_prefix_name=None,
            fontsize=14.0,
            dpi=150,
        )
    )

    assert summary["plot_color"] == "cluster_leiden_graph"
    assert (outdir / "umap_graph_plot.png").exists()
    assert (outdir / "umap_graph_plot.svg").exists()
    payload = json.loads((outdir / "plot_embedding_from_h5_summary.json").read_text(encoding="utf-8"))
    assert payload["embedding_key"] == "umap_graph"


def test_plot_embedding_from_h5_accepts_obs_color_override(tmp_path: Path) -> None:
    ds = _make_dataset()
    dataset_h5 = tmp_path / "dataset.h5"
    ds.save_h5(dataset_h5)
    outdir = tmp_path / "plots_score"

    summary = run_plot_embedding_from_h5(
        Namespace(
            dataset_h5=dataset_h5,
            outdir=outdir,
            embedding_key="pca_graph",
            plot_color="score",
            title="PCA score",
            out_prefix_name="pca_score",
            fontsize=14.0,
            dpi=150,
        )
    )

    assert summary["plot_color"] == "score"
    assert (outdir / "pca_score.png").exists()
    assert (outdir / "pca_score.svg").exists()
