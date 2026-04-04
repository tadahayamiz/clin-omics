from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import pandas as pd
import pytest

from clin_omics.analysis import prepare_feature_vs_obs_comparison
from clin_omics.dataset import CanonicalDataset
from clin_omics.visualization import plot_feature_vs_obs
from clin_omics.workflows.plot_feature_vs_obs_from_h5 import run_plot_feature_vs_obs_from_h5
from clin_omics.visualization.style import DEFAULT_CONTROL_COLOR, DEFAULT_TREATMENT_COLOR, resolve_group_colors



def make_dataset() -> CanonicalDataset:
    X = pd.DataFrame(
        [[10.0, 1.0], [12.0, 2.0], [9.0, 3.0], [20.0, 4.0], [22.0, 5.0]],
        index=["s1", "s2", "s3", "s4", "s5"],
        columns=["gene_a", "gene_b"],
    )
    obs = pd.DataFrame(
        {
            "sample_id": ["s1", "s2", "s3", "s4", "s5"],
            "group": ["control", "control", "treated", "treated", pd.NA],
            "binary_code": [0, 0, 1, 1, 1],
            "continuous_score": [1.0, 2.0, 3.5, 4.2, 5.1],
        }
    )
    var = pd.DataFrame({
        "feature_id": ["gene_a", "gene_b"],
        "gene_symbol": ["GATA1", "MYC"],
        "protein_name": ["Protein A", "Protein B"],
    })
    layers = {"log": pd.DataFrame(X + 1.0, index=X.index, columns=X.columns)}
    return CanonicalDataset(X=X, obs=obs, var=var, layers=layers)



def test_prepare_feature_vs_obs_comparison_filters_missing_and_honors_group_order() -> None:
    ds = make_dataset()
    comparison = prepare_feature_vs_obs_comparison(
        ds,
        feature="gene_a",
        obs_field="group",
        group_order=["control", "treated"],
    )
    assert comparison.n_total == 5
    assert comparison.n_used == 4
    assert comparison.n_missing_group == 1
    assert comparison.n_missing_feature == 0
    assert comparison.groups == ("control", "treated")
    assert comparison.data["sample_id"].tolist() == ["s1", "s2", "s3", "s4"]



def test_prepare_feature_vs_obs_comparison_supports_layer_selection() -> None:
    ds = make_dataset()
    comparison = prepare_feature_vs_obs_comparison(ds, feature="gene_a", obs_field="group", layer="log")
    values = comparison.data["value"].tolist()
    assert values == [11.0, 13.0, 10.0, 21.0]





def test_prepare_feature_vs_obs_comparison_supports_var_lookup_column() -> None:
    ds = make_dataset()
    comparison = prepare_feature_vs_obs_comparison(
        ds,
        feature="GATA1",
        feature_lookup_col="gene_symbol",
        obs_field="group",
    )
    assert comparison.feature == "gene_a"
    assert comparison.feature_query == "GATA1"
    assert comparison.feature_lookup_col == "gene_symbol"
    assert comparison.data["value"].tolist() == [10.0, 12.0, 9.0, 20.0]


def test_prepare_feature_vs_obs_comparison_rejects_unknown_var_lookup_column() -> None:
    ds = make_dataset()
    with pytest.raises(Exception, match="Unknown var lookup column"):
        prepare_feature_vs_obs_comparison(
            ds,
            feature="GATA1",
            feature_lookup_col="missing_col",
            obs_field="group",
        )


def test_prepare_feature_vs_obs_comparison_rejects_non_unique_var_lookup_match() -> None:
    ds = make_dataset()
    ds.var.loc[1, "gene_symbol"] = "GATA1"
    with pytest.raises(Exception, match="matched multiple feature_id values"):
        prepare_feature_vs_obs_comparison(
            ds,
            feature="GATA1",
            feature_lookup_col="gene_symbol",
            obs_field="group",
        )


def test_plot_feature_vs_obs_from_h5_workflow_runs_with_var_lookup_column(tmp_path: Path) -> None:
    ds = make_dataset()
    dataset_h5 = tmp_path / "dataset_lookup.h5"
    outdir = tmp_path / "plots_lookup"
    ds.save_h5(dataset_h5)

    summary = run_plot_feature_vs_obs_from_h5(
        type("Args", (), {
            "dataset_h5": dataset_h5,
            "feature": "GATA1",
            "feature_lookup_col": "gene_symbol",
            "obs_field": "group",
            "outdir": outdir,
            "layer": None,
            "title": None,
            "xlabel": None,
            "ylabel": None,
            "out_prefix_name": "gene_symbol_lookup",
            "control_group": "control",
            "group_order": ["control", "treated"],
            "group_color": None,
            "show_box": True,
            "annotate_mann_whitney": False,
            "fontsize": None,
            "dpi": None,
            "width": None,
            "height": None,
            "save_png": True,
            "save_svg": False,
            "marker_size": None,
            "line_width": None,
            "alpha": None,
            "jitter": None,
            "yscale": None,
            "show_top_spine": False,
            "show_right_spine": False,
        })()
    )

    assert summary["feature"] == "gene_a"
    assert summary["feature_query"] == "GATA1"
    assert summary["feature_lookup_col"] == "gene_symbol"
    assert (outdir / "gene_symbol_lookup.png").exists()


def test_prepare_feature_vs_obs_comparison_rejects_continuous_numeric_obs_field() -> None:
    ds = make_dataset()
    try:
        prepare_feature_vs_obs_comparison(ds, feature="gene_a", obs_field="continuous_score")
    except Exception as exc:
        assert "categorical-like" in str(exc)
    else:
        raise AssertionError("Expected continuous numeric obs field to be rejected")



def test_resolve_group_colors_uses_control_gray_and_single_treatment_blue() -> None:
    color_map = resolve_group_colors(["control", "treated"], control_group="control")
    assert color_map["control"] == DEFAULT_CONTROL_COLOR
    assert color_map["treated"] == DEFAULT_TREATMENT_COLOR



def test_plot_feature_vs_obs_applies_spine_and_color_overrides_and_saves_outputs(tmp_path: Path) -> None:
    ds = make_dataset()
    comparison = prepare_feature_vs_obs_comparison(ds, feature="gene_a", obs_field="group")
    out_prefix = tmp_path / "feature_vs_obs"
    fig, ax, summary = plot_feature_vs_obs(
        comparison,
        control_group="control",
        color_overrides={"treated": "#123456"},
        config={
            "show_top_spine": False,
            "show_right_spine": False,
            "marker_size": 25.0,
            "alpha": 0.5,
            "jitter": 0.05,
            "yscale": "linear",
        },
        out_prefix=out_prefix,
    )
    assert fig is not None
    assert ax.spines["top"].get_visible() is False
    assert ax.spines["right"].get_visible() is False
    assert summary["color_map"]["control"] == DEFAULT_CONTROL_COLOR
    assert summary["color_map"]["treated"] == "#123456"
    assert out_prefix.with_suffix(".png").exists()
    assert out_prefix.with_suffix(".svg").exists()


def test_plot_feature_vs_obs_from_h5_workflow_runs(tmp_path: Path) -> None:
    ds = make_dataset()
    dataset_h5 = tmp_path / "dataset.h5"
    outdir = tmp_path / "plots"
    ds.save_h5(dataset_h5)

    summary = run_plot_feature_vs_obs_from_h5(
        type("Args", (), {
            "dataset_h5": dataset_h5,
            "feature": "gene_a",
            "obs_field": "group",
            "outdir": outdir,
            "layer": None,
            "title": "Gene A by group",
            "xlabel": None,
            "ylabel": None,
            "out_prefix_name": "gene_a_group",
            "control_group": "control",
            "group_order": ["control", "treated"],
            "group_color": ["treated=#3366CC"],
            "show_box": True,
            "fontsize": 13.0,
            "dpi": 120,
            "width": 5.5,
            "height": 4.5,
            "save_png": True,
            "save_svg": True,
            "marker_size": 19.0,
            "line_width": 1.2,
            "alpha": 0.7,
            "jitter": 0.04,
            "yscale": "linear",
            "show_top_spine": False,
            "show_right_spine": False,
        })()
    )

    assert summary["feature"] == "gene_a"
    assert summary["obs_field"] == "group"
    assert summary["color_map"]["control"] == DEFAULT_CONTROL_COLOR
    assert summary["color_map"]["treated"] == "#3366CC"
    assert (outdir / "gene_a_group.png").exists()
    assert (outdir / "gene_a_group.svg").exists()
    assert (outdir / "gene_a_group_summary.json").exists()


def test_plot_feature_vs_obs_with_mann_whitney_annotation(tmp_path: Path) -> None:
    ds = make_dataset()
    comparison = prepare_feature_vs_obs_comparison(ds, feature="gene_a", obs_field="group")
    out_prefix = tmp_path / "feature_vs_obs_mw"
    fig, ax, summary = plot_feature_vs_obs(
        comparison,
        control_group="control",
        annotate_mann_whitney=True,
        out_prefix=out_prefix,
    )
    assert fig is not None
    assert ax is not None
    assert summary["stat_annotation"] is not None
    assert summary["stat_annotation"]["test"] == "mann_whitney"
    assert "Mann-Whitney" in summary["stat_annotation"]["label"]
    assert out_prefix.with_suffix(".png").exists()


def test_plot_feature_vs_obs_mann_whitney_requires_two_groups() -> None:
    ds = make_dataset()
    ds.obs["group3"] = ["control", "control", "mid", "treated", "treated"]
    comparison = prepare_feature_vs_obs_comparison(ds, feature="gene_a", obs_field="group3")
    with pytest.raises(Exception, match="exactly two groups"):
        plot_feature_vs_obs(comparison, annotate_mann_whitney=True)


def test_plot_feature_vs_obs_from_h5_workflow_runs_with_mann_whitney(tmp_path: Path) -> None:
    ds = make_dataset()
    dataset_h5 = tmp_path / "dataset.h5"
    outdir = tmp_path / "plots_mw"
    ds.save_h5(dataset_h5)

    summary = run_plot_feature_vs_obs_from_h5(
        type("Args", (), {
            "dataset_h5": dataset_h5,
            "feature": "gene_a",
            "obs_field": "group",
            "outdir": outdir,
            "layer": None,
            "title": None,
            "xlabel": None,
            "ylabel": None,
            "out_prefix_name": "gene_a_group_mw",
            "control_group": "control",
            "group_order": ["control", "treated"],
            "group_color": None,
            "show_box": True,
            "annotate_mann_whitney": True,
            "fontsize": None,
            "dpi": None,
            "width": None,
            "height": None,
            "save_png": True,
            "save_svg": False,
            "marker_size": None,
            "line_width": None,
            "alpha": None,
            "jitter": None,
            "yscale": None,
            "show_top_spine": False,
            "show_right_spine": False,
        })()
    )

    assert summary["stat_annotation"] is not None
    assert summary["stat_annotation"]["test"] == "mann_whitney"
    assert (outdir / "gene_a_group_mw_summary.json").exists()



def test_prepare_feature_vs_obs_comparison_aligns_by_sample_id_not_obs_row_order() -> None:
    ds = make_dataset()
    ds.obs = ds.obs.iloc[[2, 0, 4, 1, 3]].reset_index(drop=True)
    comparison = prepare_feature_vs_obs_comparison(ds, feature="gene_a", obs_field="group")
    assert comparison.data["sample_id"].tolist() == ["s1", "s2", "s3", "s4"]
    assert comparison.data["group"].astype(str).tolist() == ["control", "control", "treated", "treated"]
    assert comparison.data["value"].tolist() == [10.0, 12.0, 9.0, 20.0]



def test_prepare_feature_vs_obs_comparison_counts_missing_feature_values() -> None:
    ds = make_dataset()
    ds.X.loc["s4", "gene_a"] = pd.NA
    comparison = prepare_feature_vs_obs_comparison(ds, feature="gene_a", obs_field="group")
    assert comparison.n_total == 5
    assert comparison.n_used == 3
    assert comparison.n_missing_group == 1
    assert comparison.n_missing_feature == 1
    assert comparison.data["sample_id"].tolist() == ["s1", "s2", "s3"]



def test_prepare_feature_vs_obs_comparison_rejects_group_order_with_group_absent_after_filtering() -> None:
    ds = make_dataset()
    ds.obs["filtered_group"] = ["control", "control", pd.NA, pd.NA, pd.NA]
    with pytest.raises(Exception, match="group_order contains groups absent after filtering"):
        prepare_feature_vs_obs_comparison(
            ds,
            feature="gene_a",
            obs_field="filtered_group",
            group_order=["control", "treated"],
        )



def test_prepare_feature_vs_obs_comparison_accepts_binary_numeric_obs_field() -> None:
    ds = make_dataset()
    comparison = prepare_feature_vs_obs_comparison(ds, feature="gene_a", obs_field="binary_code")
    assert comparison.groups == ("0", "1")
    assert comparison.n_used == 5
    assert comparison.data["group"].astype(str).tolist() == ["0", "0", "1", "1", "1"]



def test_prepare_feature_vs_obs_comparison_rejects_duplicate_feature_index() -> None:
    ds = make_dataset()
    ds.X.index = ["s1", "s1", "s3", "s4", "s5"]
    with pytest.raises(Exception, match="index must be unique"):
        prepare_feature_vs_obs_comparison(ds, feature="gene_a", obs_field="group")
