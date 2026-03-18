from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd

from clin_omics.dataset import CanonicalDataset

REPO_ROOT = Path(__file__).resolve().parents[1]


def _subprocess_env() -> dict[str, str]:
    env = dict(os.environ)
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{REPO_ROOT / 'src'}{os.pathsep}{existing}" if existing else str(REPO_ROOT / 'src')
    return env


def _make_preprocessed_dataset() -> CanonicalDataset:
    obs = pd.DataFrame({"sample_id": ["S1", "S2", "S3", "S4"], "group": ["A", "A", "B", "B"]})
    var = pd.DataFrame({"feature_id": ["G1", "G2", "G3"], "feature_name": ["Gene1", "Gene2", "Gene3"]})
    X = pd.DataFrame(
        [[100.0, 5.0, 50.0], [120.0, 0.0, 52.0], [90.0, 3.0, 48.0], [110.0, 1.0, 51.0]],
        index=["S1", "S2", "S3", "S4"],
        columns=["G1", "G2", "G3"],
    )
    log_cpm = pd.DataFrame(
        [[8.0, 1.0, 6.0], [8.3, 0.5, 6.1], [7.8, 0.8, 5.9], [8.1, 0.6, 6.0]],
        index=X.index,
        columns=X.columns,
    )
    return CanonicalDataset(X=X, obs=obs, var=var, layers={"log_cpm": log_cpm})


def test_bulk_rnaseq_basic_from_h5_flow_runs(tmp_path: Path) -> None:
    dataset = _make_preprocessed_dataset()
    in_path = tmp_path / "preprocessed.h5"
    outdir = tmp_path / "basic_from_h5_out"
    dataset.save_h5(in_path)

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "clin_omics.workflows.bulk_rnaseq_basic_from_h5",
            "--dataset-h5",
            str(in_path),
            "--outdir",
            str(outdir),
            "--source-layer",
            "log_cpm",
            "--n-pca-components",
            "2",
            "--kmeans-clusters",
            "2",
        ],
        cwd=REPO_ROOT,
        env=_subprocess_env(),
        capture_output=True,
        text=True,
        check=True,
    )

    summary = json.loads(result.stdout)
    assert summary["input_dataset"] == str(in_path)
    assert summary["output_dataset"].endswith("bulk_rnaseq_basic_from_h5_output_dataset.h5")
    assert summary["source_layer"] == "log_cpm"
    assert summary["n_samples"] == 4
    assert summary["n_features"] == 3
    assert "pca_basic" in summary["embeddings"]
    assert "cluster_kmeans_basic" in summary["assignments"]

    assert (outdir / "bulk_rnaseq_basic_from_h5_output_dataset.h5").exists()
    assert (outdir / "bulk_rnaseq_basic_from_h5_summary.json").exists()
    assert (outdir / "cluster_kmeans_basic.csv").exists()
    assert (outdir / "pca_basic.png").exists()
    assert (outdir / "pca_basic.svg").exists()
