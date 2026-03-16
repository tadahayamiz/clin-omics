from __future__ import annotations

from pathlib import Path

import pandas as pd

from clin_omics.dataset import CanonicalDataset


def read_dataset_h5(path: str | Path) -> CanonicalDataset:
    return CanonicalDataset.load_h5(path)


def read_table(path: str | Path) -> pd.DataFrame:
    input_path = Path(path)
    suffix = input_path.suffix.lower()

    if suffix in {".csv"}:
        return pd.read_csv(input_path)
    if suffix in {".tsv", ".txt"}:
        return pd.read_csv(input_path, sep="\t")
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(input_path)

    raise ValueError(
        f"Unsupported table format for {input_path}. Expected .csv, .tsv, .txt, or .parquet."
    )
