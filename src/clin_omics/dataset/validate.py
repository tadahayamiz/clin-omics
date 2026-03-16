from __future__ import annotations

import pandas as pd
from pandas.api.types import is_numeric_dtype

from clin_omics.constants import REQUIRED_OBS_ID_COLUMN, REQUIRED_VAR_ID_COLUMN
from clin_omics.exceptions import SchemaValidationError
from clin_omics.schema import validate_layer_shapes, validate_obs_table, validate_var_table


def _ensure_numeric_frame(frame: pd.DataFrame, label: str) -> None:
    non_numeric = [col for col in frame.columns if not is_numeric_dtype(frame[col])]
    if non_numeric:
        raise SchemaValidationError(f"{label} must contain only numeric columns: {non_numeric}")


def _validate_embedding_frames(
    embeddings: dict[str, pd.DataFrame], expected_index: list[str]
) -> None:
    for name, frame in embeddings.items():
        if not isinstance(frame, pd.DataFrame):
            raise SchemaValidationError(f"Embedding '{name}' must be a pandas DataFrame.")
        _ensure_numeric_frame(frame, f"Embedding '{name}'")
        if frame.index.tolist() != expected_index:
            raise SchemaValidationError(
                f"Embedding '{name}' index must exactly match obs['sample_id'] in order."
            )


def _validate_feature_score_frames(
    feature_scores: dict[str, pd.DataFrame], expected_index: list[str]
) -> None:
    for name, frame in feature_scores.items():
        if not isinstance(frame, pd.DataFrame):
            raise SchemaValidationError(
                f"Feature score '{name}' must be a pandas DataFrame."
            )
        _ensure_numeric_frame(frame, f"Feature score '{name}'")
        if frame.index.tolist() != expected_index:
            raise SchemaValidationError(
                f"Feature score '{name}' index must exactly match var['feature_id'] in order."
            )


def _validate_assignments(
    assignments: dict[str, pd.Series], expected_index: list[str]
) -> None:
    for name, series in assignments.items():
        if not isinstance(series, pd.Series):
            raise SchemaValidationError(f"Assignment '{name}' must be a pandas Series.")
        if series.index.tolist() != expected_index:
            raise SchemaValidationError(
                f"Assignment '{name}' index must exactly match obs['sample_id'] in order."
            )


def validate_dataset_components(
    X: pd.DataFrame,
    obs: pd.DataFrame,
    var: pd.DataFrame,
    layers: dict[str, pd.DataFrame] | None = None,
    embeddings: dict[str, pd.DataFrame] | None = None,
    feature_scores: dict[str, pd.DataFrame] | None = None,
    assignments: dict[str, pd.Series] | None = None,
) -> None:
    validated_obs = validate_obs_table(obs)
    validated_var = validate_var_table(var)

    if not isinstance(X, pd.DataFrame):
        raise SchemaValidationError("X must be a pandas DataFrame.")

    _ensure_numeric_frame(X, "X")

    expected_shape = (len(validated_obs), len(validated_var))
    if X.shape != expected_shape:
        raise SchemaValidationError(
            f"X has shape {X.shape}, expected {expected_shape}."
        )

    expected_index = validated_obs[REQUIRED_OBS_ID_COLUMN].tolist()
    expected_columns = validated_var[REQUIRED_VAR_ID_COLUMN].tolist()

    if X.index.tolist() != expected_index:
        raise SchemaValidationError(
            "X index must exactly match obs['sample_id'] in order."
        )

    if X.columns.tolist() != expected_columns:
        raise SchemaValidationError(
            "X columns must exactly match var['feature_id'] in order."
        )

    validate_layer_shapes(layers or {}, n_obs=expected_shape[0], n_var=expected_shape[1])

    for name, layer in (layers or {}).items():
        _ensure_numeric_frame(layer, f"Layer '{name}'")
        if layer.index.tolist() != expected_index:
            raise SchemaValidationError(
                f"Layer '{name}' index must exactly match obs['sample_id'] in order."
            )
        if layer.columns.tolist() != expected_columns:
            raise SchemaValidationError(
                f"Layer '{name}' columns must exactly match var['feature_id'] in order."
            )

    _validate_embedding_frames(embeddings or {}, expected_index)
    _validate_feature_score_frames(feature_scores or {}, expected_columns)
    _validate_assignments(assignments or {}, expected_index)
