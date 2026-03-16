from __future__ import annotations

import pandas as pd

from clin_omics.dataset import CanonicalDataset


def export_obs_table(dataset: CanonicalDataset) -> pd.DataFrame:
    return dataset.obs.copy()


def export_var_table(dataset: CanonicalDataset) -> pd.DataFrame:
    return dataset.var.copy()


def export_embedding_table(dataset: CanonicalDataset, key: str) -> pd.DataFrame:
    frame = dataset.embeddings[key].copy()
    return pd.DataFrame({"sample_id": dataset.obs["sample_id"].tolist()}).join(frame)


def export_feature_scores_table(dataset: CanonicalDataset, key: str) -> pd.DataFrame:
    frame = dataset.feature_scores[key].copy()
    return pd.DataFrame({"feature_id": dataset.var["feature_id"].tolist()}).join(frame)


def export_assignments_table(dataset: CanonicalDataset, key: str) -> pd.DataFrame:
    series = dataset.assignments[key].reindex(dataset.obs["sample_id"]).copy()
    return pd.DataFrame({"sample_id": dataset.obs["sample_id"].tolist(), key: series.to_numpy()})
