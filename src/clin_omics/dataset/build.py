from __future__ import annotations

from typing import Any

import pandas as pd

from clin_omics.curation import (
    build_obs_from_clinical,
    build_var_from_matrix,
    curate_clinical_table,
    curate_omics_long_to_matrix,
    curate_omics_matrix,
)
from clin_omics.dataset.model import CanonicalDataset


def build_dataset(
    *,
    omics: pd.DataFrame,
    clinical: pd.DataFrame | None = None,
    feature_meta: pd.DataFrame | None = None,
    omics_format: str = "matrix",
    sample_id_col: str = "sample_id",
    feature_id_col: str = "feature_id",
    value_col: str = "value",
    duplicate_policy: str = "error",
    clinical_sample_id_col: str = "sample_id",
    clinical_rename_map: dict[str, str] | None = None,
    clinical_category_maps: dict[str, dict[Any, Any]] | None = None,
    clinical_dtype_map: dict[str, str] | None = None,
    layers: dict[str, pd.DataFrame] | None = None,
    provenance: dict[str, Any] | None = None,
) -> CanonicalDataset:
    if omics_format == "long":
        X = curate_omics_long_to_matrix(
            omics,
            sample_id_col=sample_id_col,
            feature_id_col=feature_id_col,
            value_col=value_col,
            duplicate_policy=duplicate_policy,
        )
    elif omics_format == "matrix":
        X = curate_omics_matrix(omics, sample_id_col=sample_id_col)
    else:
        raise ValueError("omics_format must be either 'long' or 'matrix'.")

    if clinical is None:
        obs = pd.DataFrame({"sample_id": X.index.tolist()})
    else:
        curated_clinical = curate_clinical_table(
            clinical,
            sample_id_col=clinical_sample_id_col,
            rename_map=clinical_rename_map,
            category_maps=clinical_category_maps,
            dtype_map=clinical_dtype_map,
        )
        obs = build_obs_from_clinical(X, curated_clinical)

    if feature_meta is None:
        var = build_var_from_matrix(X)
    else:
        feature_meta_work = feature_meta.copy()
        if feature_id_col != "feature_id":
            feature_meta_work = feature_meta_work.rename(columns={feature_id_col: "feature_id"})
        var = (
            feature_meta_work.set_index("feature_id")
            .reindex(X.columns)
            .rename_axis("feature_id")
            .reset_index()
        )

    provenance_payload = {} if provenance is None else dict(provenance)
    provenance_payload.setdefault("build", {})
    provenance_payload["build"].update(
        {
            "omics_format": omics_format,
            "n_obs": int(X.shape[0]),
            "n_var": int(X.shape[1]),
        }
    )

    return CanonicalDataset(
        X=X,
        obs=obs,
        var=var,
        layers={} if layers is None else layers,
        provenance=provenance_payload,
    )
