from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_bulk_rnaseq_smoke_script_runs(tmp_path: Path) -> None:
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
    outdir = tmp_path / "smoke_out"
    x.to_csv(x_path, index=False)
    obs.to_csv(obs_path, index=False)
    var.to_csv(var_path, index=False)

    result = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "smoke_bulk_rnaseq.py"),
            "--x",
            str(x_path),
            "--obs",
            str(obs_path),
            "--var",
            str(var_path),
            "--outdir",
            str(outdir),
            "--min-count",
            "10",
            "--min-samples",
            "2",
            "--make-zscore",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )

    summary = json.loads(result.stdout)
    assert summary["n_samples"] == 4
    assert summary["n_features"] == 2
    assert "log_cpm" in summary["layers"]
    assert "pca_smoke" in summary["embeddings"]
    assert "cluster_kmeans_smoke" in summary["assignments"]

    assert (outdir / "smoke_input_dataset.h5").exists()
    assert (outdir / "smoke_processed_dataset.h5").exists()
    assert (outdir / "smoke_summary.json").exists()
    assert (outdir / "cluster_kmeans_smoke.csv").exists()
    assert (outdir / "pca_smoke.png").exists()
    assert (outdir / "pca_smoke.svg").exists()
