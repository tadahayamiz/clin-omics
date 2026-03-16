from __future__ import annotations

import pandas as pd

from clin_omics.constants import REQUIRED_VAR_ID_COLUMN
from clin_omics.exceptions import SchemaValidationError
from clin_omics.schema import validate_feature_ids, validate_sample_ids


_ALLOWED_DUPLICATE_POLICIES = {"error", "mean", "sum", "first"}


def curate_omics_long_to_matrix(
    table: pd.DataFrame,
    *,
    sample_id_col: str = "sample_id",
    feature_id_col: str = "feature_id",
    value_col: str = "value",
    duplicate_policy: str = "error",
) -> pd.DataFrame:
    """Convert a long-format omics table to a samples x features matrix."""
    required = [sample_id_col, feature_id_col, value_col]
    missing = [col for col in required if col not in table.columns]
    if missing:
        raise SchemaValidationError(
            f"Long omics table is missing required columns: {missing}"
        )

    if duplicate_policy not in _ALLOWED_DUPLICATE_POLICIES:
        raise SchemaValidationError(
            f"duplicate_policy must be one of {sorted(_ALLOWED_DUPLICATE_POLICIES)}."
        )

    work = table[[sample_id_col, feature_id_col, value_col]].copy()
    validate_sample_ids(work[sample_id_col]) if not work.empty and not work[sample_id_col].duplicated().any() else None
    validate_feature_ids(work[feature_id_col]) if not work.empty and not work[feature_id_col].duplicated().any() else None

    work[value_col] = pd.to_numeric(work[value_col], errors="raise")

    duplicated_mask = work.duplicated(subset=[sample_id_col, feature_id_col], keep=False)
    if duplicated_mask.any() and duplicate_policy == "error":
        duplicated_rows = work.loc[duplicated_mask, [sample_id_col, feature_id_col]].drop_duplicates()
        examples = duplicated_rows.head(10).to_dict(orient="records")
        raise SchemaValidationError(
            "Long omics table contains duplicate sample-feature pairs. "
            f"Examples: {examples}"
        )

    if duplicate_policy == "error":
        collapsed = work
    elif duplicate_policy == "mean":
        collapsed = (
            work.groupby([sample_id_col, feature_id_col], as_index=False)[value_col].mean()
        )
    elif duplicate_policy == "sum":
        collapsed = (
            work.groupby([sample_id_col, feature_id_col], as_index=False)[value_col].sum()
        )
    else:  # first
        collapsed = (
            work.groupby([sample_id_col, feature_id_col], as_index=False)[value_col].first()
        )

    sample_order = pd.Index(pd.unique(collapsed[sample_id_col]), name="sample_id")
    feature_order = pd.Index(pd.unique(collapsed[feature_id_col]), name=REQUIRED_VAR_ID_COLUMN)

    matrix = collapsed.pivot(
        index=sample_id_col,
        columns=feature_id_col,
        values=value_col,
    )
    matrix = matrix.reindex(index=sample_order, columns=feature_order)
    matrix.index.name = None
    matrix.columns.name = None
    return matrix



def curate_omics_matrix(
    matrix: pd.DataFrame,
    *,
    sample_id_col: str | None = None,
    feature_id_col: str | None = None,
) -> pd.DataFrame:
    """Normalize a matrix-format omics table to a samples x features DataFrame."""
    work = matrix.copy()

    if sample_id_col is not None:
        if sample_id_col not in work.columns:
            raise SchemaValidationError(
                f"Matrix table does not contain sample id column '{sample_id_col}'."
            )
        work = work.set_index(sample_id_col, drop=True)

    if feature_id_col is not None:
        if feature_id_col not in work.index.names:
            raise SchemaValidationError(
                "feature_id_col is not supported for row-wise feature matrices; "
                "provide a samples x features table instead."
            )

    if work.index.hasnans:
        raise SchemaValidationError("Matrix index contains missing sample ids.")
    if work.index.duplicated().any():
        duplicated = work.index[work.index.duplicated()].tolist()
        raise SchemaValidationError(f"Matrix index contains duplicate sample ids: {duplicated}")
    if pd.Index(work.columns).hasnans:
        raise SchemaValidationError("Matrix columns contain missing feature ids.")
    if pd.Index(work.columns).duplicated().any():
        duplicated = pd.Index(work.columns)[pd.Index(work.columns).duplicated()].tolist()
        raise SchemaValidationError(f"Matrix columns contain duplicate feature ids: {duplicated}")

    validate_sample_ids(pd.Series(work.index.astype(str), name="sample_id"))
    validate_feature_ids(pd.Series(pd.Index(work.columns).astype(str), name=REQUIRED_VAR_ID_COLUMN))

    for col in work.columns:
        work[col] = pd.to_numeric(work[col], errors="raise")

    work.index = pd.Index(work.index.astype(str))
    work.columns = pd.Index(pd.Index(work.columns).astype(str))
    work.index.name = None
    work.columns.name = None
    return work
