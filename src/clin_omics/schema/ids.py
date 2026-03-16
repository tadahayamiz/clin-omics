from __future__ import annotations

import pandas as pd

from clin_omics.exceptions import SchemaValidationError


def _validate_ids(values: pd.Series, *, label: str) -> pd.Index:
    if values.isna().any():
        raise SchemaValidationError(f"{label} contains missing values.")

    if (values.astype(str).str.len() == 0).any():
        raise SchemaValidationError(f"{label} contains empty strings.")

    duplicated = values[values.duplicated()].tolist()
    if duplicated:
        unique_duplicated = list(dict.fromkeys(duplicated))
        raise SchemaValidationError(f"{label} contains duplicates: {unique_duplicated}")

    return pd.Index(values.tolist(), name=label)


def validate_sample_ids(values: pd.Series) -> pd.Index:
    return _validate_ids(values, label="sample_id")



def validate_feature_ids(values: pd.Series) -> pd.Index:
    return _validate_ids(values, label="feature_id")
