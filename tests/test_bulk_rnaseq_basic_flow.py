from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]


def _subprocess_env() -> dict[str, str]:
    env = dict(os.environ)
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{REPO_ROOT / 'src'}{os.pathsep}{existing}" if existing else str(REPO_ROOT / 'src')
    return env


def test_bulk_rnaseq_basic_flow_runs(tmp_path: Path) -> None:
    x = pd.DataFrame(
        {
            "sample_id": ["S1", "S2", "S3", "S4"],
            "G1": [100, 120, 90, 110],
            "G2": [5, 0, 3, 1],
            "G3": [50, 52, 48, 51],
        }
    )
    obs = pd.DataFrame(
        {
            "sample_id": ["S1", "S2", "S3", "S4"],
            "group": ["A", "A", "B", "B"],
        }
    )
    var = pd.DataFrame(
        {
            "feature_id": ["G1", "G2", "G3"],
            "feature_name": ["Gene1", "Gene2", "Gene3"],
        }
    )

    x_path = tmp_path / "assay.csv"
    obs_path = tmp_path / "obs.csv"
    var_path = tmp_path / "var.csv"
    outdir = tmp_path / "basic_out"
    x.to_csv(x_path, index=False)
    obs.to_csv(obs_path, index=False)
    var.to_csv(var_path, index=False)

    result = subprocess.run(
        [
            sys.executable,
            '-m',
            'clin_omics.workflows.bulk_rnaseq_basic',
            '--x',
            str(x_path),
            '--obs',
            str(obs_path),
            '--var',
            str(var_path),
            '--outdir',
            str(outdir),
            '--min-count',
            '10',
            '--min-samples',
            '2',
            '--make-zscore',
        ],
        cwd=REPO_ROOT,
        env=_subprocess_env(),
        capture_output=True,
        text=True,
        check=True,
    )

    summary = json.loads(result.stdout)
    assert summary["n_samples"] == 4
    assert summary["n_features"] == 2
    assert "log_cpm" in summary["layers"]
    assert "pca_basic" in summary["embeddings"]
    assert "cluster_kmeans_basic" in summary["assignments"]

    assert (outdir / "bulk_rnaseq_basic_input_dataset.h5").exists()
    assert (outdir / "bulk_rnaseq_basic_processed_dataset.h5").exists()
    assert (outdir / "bulk_rnaseq_basic_summary.json").exists()
    assert (outdir / "cluster_kmeans_basic.csv").exists()
    assert (outdir / "pca_basic.png").exists()
    assert (outdir / "pca_basic.svg").exists()


def test_bulk_rnaseq_basic_flow_can_skip_kmeans(tmp_path: Path) -> None:
    x = pd.DataFrame({"sample_id": ["S1", "S2", "S3"], "G1": [10, 20, 30], "G2": [40, 50, 60]})
    obs = pd.DataFrame({"sample_id": ["S1", "S2", "S3"], "group": ["A", "A", "B"]})
    var = pd.DataFrame({"feature_id": ["G1", "G2"]})

    x_path = tmp_path / "assay.csv"
    obs_path = tmp_path / "obs.csv"
    var_path = tmp_path / "var.csv"
    outdir = tmp_path / "basic_out"
    x.to_csv(x_path, index=False)
    obs.to_csv(obs_path, index=False)
    var.to_csv(var_path, index=False)

    result = subprocess.run(
        [
            sys.executable,
            '-m',
            'clin_omics.workflows.bulk_rnaseq_basic',
            '--x', str(x_path),
            '--obs', str(obs_path),
            '--var', str(var_path),
            '--outdir', str(outdir),
            '--skip-kmeans',
        ],
        cwd=REPO_ROOT,
        env=_subprocess_env(),
        capture_output=True,
        text=True,
        check=True,
    )

    summary = json.loads(result.stdout)
    assert summary["run_kmeans"] is False
    assert summary["assignments"] == []
    assert (outdir / "cluster_kmeans_basic.csv").exists() is False
    assert (outdir / "pca_basic.png").exists()


def test_bulk_rnaseq_basic_flow_runs_with_index_style_x(tmp_path: Path) -> None:
    x = pd.DataFrame(
        {
            "G1": [100, 120, 90, 110],
            "G2": [5, 0, 3, 1],
            "G3": [50, 52, 48, 51],
        },
        index=["S1", "S2", "S3", "S4"],
    )
    obs = pd.DataFrame(
        {
            "sample_id": ["S1", "S2", "S3", "S4"],
            "group": ["A", "A", "B", "B"],
        }
    )
    var = pd.DataFrame(
        {
            "feature_id": ["G1", "G2", "G3"],
            "feature_name": ["Gene1", "Gene2", "Gene3"],
        }
    )

    x_path = tmp_path / "assay.csv"
    obs_path = tmp_path / "obs.csv"
    var_path = tmp_path / "var.csv"
    outdir = tmp_path / "basic_out"
    x.to_csv(x_path, index=True)
    obs.to_csv(obs_path, index=False)
    var.to_csv(var_path, index=False)

    result = subprocess.run(
        [
            sys.executable,
            '-m',
            'clin_omics.workflows.bulk_rnaseq_basic',
            '--x',
            str(x_path),
            '--obs',
            str(obs_path),
            '--var',
            str(var_path),
            '--outdir',
            str(outdir),
            '--min-count',
            '10',
            '--min-samples',
            '2',
        ],
        cwd=REPO_ROOT,
        env=_subprocess_env(),
        capture_output=True,
        text=True,
        check=True,
    )

    summary = json.loads(result.stdout)
    assert summary["n_samples"] == 4
    assert summary["n_features"] == 2
    assert "cluster_kmeans_basic" in summary["assignments"]
    assert (outdir / "cluster_kmeans_basic.csv").exists()
