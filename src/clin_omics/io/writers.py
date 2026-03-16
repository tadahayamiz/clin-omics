from __future__ import annotations

from pathlib import Path

from clin_omics.dataset import CanonicalDataset


def write_dataset_h5(dataset: CanonicalDataset, path: str | Path) -> Path:
    return dataset.save_h5(path)
