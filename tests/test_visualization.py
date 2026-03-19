from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import pandas as pd

from clin_omics.analysis import PCAEmbedding
from clin_omics.dataset import CanonicalDataset
from clin_omics.visualization import PlotConfig, plot_embedding, resolve_plot_config


def make_dataset() -> CanonicalDataset:
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
    return PCAEmbedding(n_components=2, key="pca").fit_transform(ds)


def test_resolve_plot_config_from_dict() -> None:
    cfg = resolve_plot_config({"fontsize": 18, "save_svg": False})
    assert cfg.fontsize == 18
    assert cfg.save_svg is False
    assert cfg.save_png is True


def test_plot_embedding_saves_png_and_svg(tmp_path: Path) -> None:
    ds = make_dataset()
    out_prefix = tmp_path / "plots" / "pca_plot"
    fig, ax = plot_embedding(
        ds,
        embedding_key="pca",
        color="group",
        config=PlotConfig(save_png=True, save_svg=True),
        out_prefix=out_prefix,
    )
    assert fig is not None
    assert ax.get_xlabel() == "PC1"
    assert out_prefix.with_suffix(".png").exists()
    assert out_prefix.with_suffix(".svg").exists()


def test_plot_embedding_numeric_color(tmp_path: Path) -> None:
    ds = make_dataset()
    out_prefix = tmp_path / "plots" / "pca_numeric"
    fig, ax = plot_embedding(ds, embedding_key="pca", color="score", out_prefix=out_prefix)
    assert fig is not None
    assert ax.get_ylabel() == "PC2"
    assert out_prefix.with_suffix(".png").exists()
    assert out_prefix.with_suffix(".svg").exists()


def test_plot_embedding_assignment_color(tmp_path: Path) -> None:
    ds = make_dataset()
    ds.assignments["cluster_kmeans"] = pd.Series([0, 0, 1, 1], index=ds.X.index, name="cluster_kmeans")
    out_prefix = tmp_path / "plots" / "pca_assignment"
    fig, ax = plot_embedding(ds, embedding_key="pca", color="cluster_kmeans", out_prefix=out_prefix)
    assert fig is not None
    assert ax.get_xlabel() == "PC1"
    assert out_prefix.with_suffix(".png").exists()
    assert out_prefix.with_suffix(".svg").exists()




def test_plot_embedding_assignment_color_uses_categorical_legend(tmp_path: Path) -> None:
    ds = make_dataset()
    ds.assignments["cluster_kmeans"] = pd.Series([0, 0, 1, 1], index=ds.X.index, name="cluster_kmeans")
    out_prefix = tmp_path / "plots" / "pca_assignment_legend"
    fig, ax = plot_embedding(ds, embedding_key="pca", color="cluster_kmeans", out_prefix=out_prefix)
    assert fig is not None
    legend = ax.get_legend()
    assert legend is not None
    labels = [text.get_text() for text in legend.get_texts()]
    assert labels == ["0", "1"]


def test_plot_embedding_numeric_color_with_missing_shows_na_legend(tmp_path: Path) -> None:
    ds = make_dataset()
    ds.obs.loc[1, "score"] = pd.NA
    out_prefix = tmp_path / "plots" / "pca_numeric_missing"
    fig, ax = plot_embedding(ds, embedding_key="pca", color="score", out_prefix=out_prefix)
    assert fig is not None
    legend = ax.get_legend()
    assert legend is not None
    labels = [text.get_text() for text in legend.get_texts()]
    assert labels == ["NA"]


def test_plot_embedding_unknown_color_key_raises() -> None:
    ds = make_dataset()
    try:
        plot_embedding(ds, embedding_key="pca", color="missing_key")
    except Exception as exc:
        assert "Unknown color key" in str(exc)
        assert "obs column or assignment key" in str(exc)
    else:
        raise AssertionError("Expected plot_embedding to fail for an unknown color key")


def test_plot_embedding_honors_alpha_size_and_no_legend() -> None:
    ds = make_dataset()
    fig, ax = plot_embedding(ds, embedding_key="pca", color="group", alpha=0.25, size=12.0, show_legend=False)
    collections = ax.collections
    assert collections, "Expected scatter collections to be present"
    assert collections[0].get_alpha() == 0.25
    assert collections[0].get_sizes().tolist() == [12.0]
    assert ax.get_legend() is None
