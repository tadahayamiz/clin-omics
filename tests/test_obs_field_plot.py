from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import pandas as pd

from clin_omics.dataset import CanonicalDataset
from clin_omics.visualization import plot_obs_field, summarize_obs_field

REPO_ROOT = Path(__file__).resolve().parents[1]


def _subprocess_env() -> dict[str, str]:
    env = dict(os.environ)
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{REPO_ROOT / 'src'}{os.pathsep}{existing}" if existing else str(REPO_ROOT / 'src')
    return env


def make_dataset() -> CanonicalDataset:
    X = pd.DataFrame(
        [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0], [9.0, 10.0]],
        index=["s1", "s2", "s3", "s4", "s5"],
        columns=["f1", "f2"],
    )
    obs = pd.DataFrame(
        {
            "sample_id": ["s1", "s2", "s3", "s4", "s5"],
            "hepatic_encephalopathy": ["none", "none", "controlled", pd.NA, "severe"],
            "age_at_sampling_days": [10, 20, 30, pd.NA, 55],
            "cluster_like": [0, 1, 1, 2, 2],
        }
    )
    var = pd.DataFrame({"feature_id": ["f1", "f2"]})
    return CanonicalDataset(X=X, obs=obs, var=var)


def test_summarize_obs_field_detects_bar_for_categorical() -> None:
    ds = make_dataset()
    summary = summarize_obs_field(ds, field="hepatic_encephalopathy")
    assert summary["plot_type"] == "bar"
    assert summary["n_total"] == 5
    assert summary["n_used"] == 4
    assert summary["n_missing"] == 1


def test_summarize_obs_field_detects_hist_for_continuous_numeric() -> None:
    ds = make_dataset()
    summary = summarize_obs_field(ds, field="age_at_sampling_days")
    assert summary["plot_type"] == "hist"
    assert summary["n_total"] == 5
    assert summary["n_used"] == 4


def test_summarize_obs_field_detects_bar_for_discrete_numeric() -> None:
    ds = make_dataset()
    summary = summarize_obs_field(ds, field="cluster_like")
    assert summary["plot_type"] == "bar"


def test_plot_obs_field_saves_outputs_and_title_contains_counts(tmp_path: Path) -> None:
    ds = make_dataset()
    fig, ax, summary = plot_obs_field(ds, field="hepatic_encephalopathy", out_prefix=tmp_path / "obs_he")
    assert fig is not None
    assert summary["plot_type"] == "bar"
    assert "n=4/5" in ax.get_title()
    assert (tmp_path / "obs_he.png").exists()
    assert (tmp_path / "obs_he.svg").exists()


def test_plot_obs_field_from_h5_workflow_runs(tmp_path: Path) -> None:
    ds = make_dataset()
    in_path = tmp_path / "dataset.h5"
    outdir = tmp_path / "obs_plot_out"
    ds.save_h5(in_path)

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "clin_omics.workflows.plot_obs_field_from_h5",
            "--dataset-h5",
            str(in_path),
            "--field",
            "age_at_sampling_days",
            "--outdir",
            str(outdir),
        ],
        cwd=REPO_ROOT,
        env=_subprocess_env(),
        capture_output=True,
        text=True,
        check=True,
    )

    summary = json.loads(result.stdout)
    assert summary["field"] == "age_at_sampling_days"
    assert summary["plot_type"] == "hist"
    assert (outdir / "obs_age_at_sampling_days.png").exists()
    assert (outdir / "obs_age_at_sampling_days.svg").exists()
    assert (outdir / "obs_age_at_sampling_days_summary.json").exists()
