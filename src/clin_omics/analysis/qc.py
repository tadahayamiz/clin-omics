from __future__ import annotations

import pandas as pd

from clin_omics.dataset import CanonicalDataset


def summarize_dataset_qc(dataset: CanonicalDataset) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "n_samples": [dataset.X.shape[0]],
            "n_features": [dataset.X.shape[1]],
            "n_layers": [len(dataset.layers)],
            "n_embeddings": [len(dataset.embeddings)],
            "n_assignments": [len(dataset.assignments)],
        }
    )
