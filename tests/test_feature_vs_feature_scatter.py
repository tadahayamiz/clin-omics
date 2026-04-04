from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import pandas as pd
import pytest

from clin_omics.analysis import prepare_two_feature_scatter_data
from clin_omics.dataset import CanonicalDataset
from clin_omics.visualization import plot_feature_vs_feature_scatter
from clin_omics.workflows.plot_feature_vs_feature_from_h5 import run_plot_feature_vs_feature_from_h5
from clin_omics.visualization.style import DEFAULT_CONTROL_COLOR, DEFAULT_TREATMENT_COLOR


def make_dataset() -> CanonicalDataset:
    X = pd.DataFrame(
        [[10.0, 1.0], [12.0, 2.0], [9.0, 3.0], [20.0, 4.0], [22.0, None]],
        index=["s1", "s2", "s3", "s4", "s5"],
        columns=["gene_a", "gene_b"],
    )
    obs = pd.DataFrame(
        {
            "sample_id": ["s1", "s2", "s3", "s4", "s5"],
            "group": ["control", "control", "treated", "treated", pd.NA],
            "continuous_score": [1.0, 2.0, 3.5, 4.2, 5.1],
        }
    )
    var = pd.DataFrame(
        {
            "feature_id": ["gene_a", "gene_b"],
            "gene_symbol": ["GATA1", "MYC"],
        }
    )
    layers = {"log": pd.DataFrame(X.fillna(0.0) + 1.0, index=X.index, columns=X.columns)}
    return CanonicalDataset(X=X, obs=obs, var=var, layers=layers)


def test_prepare_two_feature_scatter_data_aligns_by_sample_id_and_filters_missing() -> None:
    ds = make_dataset()
    ds.obs = ds.obs.iloc[[2, 0, 4, 1, 3]].reset_index(drop=True)
    scatter_data = prepare_two_feature_scatter_data(
        ds,
        x_feature="gene_a",
        y_feature="gene_b",
        label_field="group",
        label_order=["control", "treated"],
    )
    assert scatter_data.n_total == 5
    assert scatter_data.n_used == 4
    assert scatter_data.n_missing_x == 0
    assert scatter_data.n_missing_y == 1
    assert scatter_data.n_missing_label == 1
    assert scatter_data.labels == ("control", "treated")
    assert scatter_data.data["sample_id"].tolist() == ["s1", "s2", "s3", "s4"]


def test_prepare_two_feature_scatter_data_supports_var_lookup_columns() -> None:
    ds = make_dataset()
    scatter_data = prepare_two_feature_scatter_data(
        ds,
        x_feature="GATA1",
        y_feature="MYC",
        x_feature_lookup_col="gene_symbol",
        y_feature_lookup_col="gene_symbol",
        label_field="group",
    )
    assert scatter_data.x_feature == "gene_a"
    assert scatter_data.y_feature == "gene_b"
    assert scatter_data.x_feature_query == "GATA1"
    assert scatter_data.y_feature_query == "MYC"


def test_prepare_two_feature_scatter_data_rejects_continuous_label_field() -> None:
    ds = make_dataset()
    with pytest.raises(Exception, match="categorical-like"):
        prepare_two_feature_scatter_data(
            ds,
            x_feature="gene_a",
            y_feature="gene_b",
            label_field="continuous_score",
        )


def test_plot_feature_vs_feature_scatter_applies_style_and_saves_outputs(tmp_path: Path) -> None:
    ds = make_dataset()
    scatter_data = prepare_two_feature_scatter_data(
        ds,
        x_feature="gene_a",
        y_feature="gene_b",
        label_field="group",
        label_order=["control", "treated"],
    )
    out_prefix = tmp_path / "feature_feature_scatter"
    fig, ax, summary = plot_feature_vs_feature_scatter(
        scatter_data,
        control_label="control",
        color_overrides={"treated": "#123456"},
        config={
            "marker_size": 35.0,
            "marker": "^",
            "marker_edge_width": 0.5,
            "alpha": 0.4,
            "xscale": "linear",
            "yscale": "linear",
            "show_top_spine": False,
            "show_right_spine": False,
        },
        out_prefix=out_prefix,
    )
    assert fig is not None
    assert ax.spines["top"].get_visible() is False
    assert ax.spines["right"].get_visible() is False
    assert ax.get_xscale() == "linear"
    assert ax.get_yscale() == "linear"
    assert summary["color_map"]["control"] == DEFAULT_CONTROL_COLOR
    assert summary["color_map"]["treated"] == "#123456"
    assert out_prefix.with_suffix(".png").exists()
    assert out_prefix.with_suffix(".svg").exists()


def test_plot_feature_vs_feature_scatter_without_label_field_uses_single_color(tmp_path: Path) -> None:
    ds = make_dataset()
    scatter_data = prepare_two_feature_scatter_data(ds, x_feature="gene_a", y_feature="gene_b")
    out_prefix = tmp_path / "feature_feature_scatter_single"
    _, _, summary = plot_feature_vs_feature_scatter(scatter_data, out_prefix=out_prefix, show_legend=False)
    assert summary["color_map"] is None
    assert summary["show_legend"] is False
    assert out_prefix.with_suffix(".png").exists()


def test_plot_feature_vs_feature_from_h5_workflow_runs(tmp_path: Path) -> None:
    ds = make_dataset()
    dataset_h5 = tmp_path / "dataset.h5"
    outdir = tmp_path / "plots"
    ds.save_h5(dataset_h5)

    summary = run_plot_feature_vs_feature_from_h5(
        type(
            "Args",
            (),
            {
                "dataset_h5": dataset_h5,
                "x_feature": "GATA1",
                "y_feature": "MYC",
                "x_feature_lookup_col": "gene_symbol",
                "y_feature_lookup_col": "gene_symbol",
                "label_field": "group",
                "label_order": ["control", "treated"],
                "outdir": outdir,
                "layer": None,
                "title": "A vs B",
                "xlabel": None,
                "ylabel": None,
                "out_prefix_name": "a_vs_b",
                "control_label": "control",
                "label_color": ["treated=#3366CC"],
                "show_legend": True,
                "fontsize": 13.0,
                "dpi": 120,
                "width": 5.5,
                "height": 4.5,
                "save_png": True,
                "save_svg": False,
                "marker_size": 24.0,
                "marker": "s",
                "marker_edge_width": 0.0,
                "alpha": 0.6,
                "xscale": "linear",
                "yscale": "linear",
                "show_top_spine": False,
                "show_right_spine": False,
            },
        )()
    )
    assert summary["x_feature"] == "gene_a"
    assert summary["y_feature"] == "gene_b"
    assert summary["color_map"]["control"] == DEFAULT_CONTROL_COLOR
    assert summary["color_map"]["treated"] == "#3366CC"
    assert (outdir / "a_vs_b.png").exists()
    assert (outdir / "a_vs_b_summary.json").exists()
