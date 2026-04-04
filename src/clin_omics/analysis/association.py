from __future__ import annotations

from dataclasses import dataclass
from math import erfc, sqrt
from typing import Any

import pandas as pd

from clin_omics.constants import REQUIRED_OBS_ID_COLUMN
from clin_omics.dataset import CanonicalDataset
from clin_omics.exceptions import ClinOmicsError


@dataclass(frozen=True)
class MannWhitneyResult:
    group_a: str
    group_b: str
    n_a: int
    n_b: int
    u_statistic: float
    p_value: float
    method: str = "normal_approx_tie_corrected"
    alternative: str = "two-sided"

    def to_summary(self) -> dict[str, Any]:
        return {
            "test": "mann_whitney",
            "group_a": self.group_a,
            "group_b": self.group_b,
            "n_a": self.n_a,
            "n_b": self.n_b,
            "u_statistic": self.u_statistic,
            "p_value": self.p_value,
            "method": self.method,
            "alternative": self.alternative,
        }


@dataclass(frozen=True)
class FeatureObsComparison:
    data: pd.DataFrame
    feature: str
    feature_query: str
    feature_lookup_col: str | None
    obs_field: str
    layer: str | None
    n_total: int
    n_used: int
    n_missing_group: int
    n_missing_feature: int
    groups: tuple[str, ...]

    def to_summary(self) -> dict[str, Any]:
        return {
            "feature": self.feature,
            "feature_query": self.feature_query,
            "feature_lookup_col": self.feature_lookup_col,
            "obs_field": self.obs_field,
            "layer": self.layer,
            "n_total": self.n_total,
            "n_used": self.n_used,
            "n_missing_group": self.n_missing_group,
            "n_missing_feature": self.n_missing_feature,
            "groups": list(self.groups),
        }


@dataclass(frozen=True)
class TwoFeatureScatterData:
    data: pd.DataFrame
    x_feature: str
    x_feature_query: str
    x_feature_lookup_col: str | None
    y_feature: str
    y_feature_query: str
    y_feature_lookup_col: str | None
    label_field: str | None
    layer: str | None
    n_total: int
    n_used: int
    n_missing_x: int
    n_missing_y: int
    n_missing_label: int
    labels: tuple[str, ...]

    def to_summary(self) -> dict[str, Any]:
        return {
            "x_feature": self.x_feature,
            "x_feature_query": self.x_feature_query,
            "x_feature_lookup_col": self.x_feature_lookup_col,
            "y_feature": self.y_feature,
            "y_feature_query": self.y_feature_query,
            "y_feature_lookup_col": self.y_feature_lookup_col,
            "label_field": self.label_field,
            "layer": self.layer,
            "n_total": self.n_total,
            "n_used": self.n_used,
            "n_missing_x": self.n_missing_x,
            "n_missing_y": self.n_missing_y,
            "n_missing_label": self.n_missing_label,
            "labels": list(self.labels),
        }


def _resolve_feature_frame(dataset: CanonicalDataset, *, layer: str | None) -> pd.DataFrame:
    if layer is None:
        return dataset.X
    if layer not in dataset.layers:
        raise ClinOmicsError(f"Unknown layer: {layer}")
    return dataset.layers[layer]


def _resolve_obs_series(dataset: CanonicalDataset, *, obs_field: str) -> pd.Series:
    if obs_field not in dataset.obs.columns:
        raise ClinOmicsError(f"Unknown obs field: {obs_field}")
    series = dataset.obs[obs_field].copy()
    series.index = dataset.obs[REQUIRED_OBS_ID_COLUMN].astype(str).tolist()
    return series


def resolve_feature_id(
    dataset: CanonicalDataset,
    *,
    feature: str,
    feature_lookup_col: str | None = None,
) -> str:
    feature_query = str(feature)
    if feature_lookup_col is None:
        if feature_query not in dataset.X.columns:
            raise ClinOmicsError(f"Unknown feature: {feature_query}")
        return feature_query

    if feature_lookup_col not in dataset.var.columns:
        raise ClinOmicsError(f"Unknown var lookup column: {feature_lookup_col}")

    lookup_series = dataset.var[feature_lookup_col].astype("string")
    matches = dataset.var.loc[lookup_series == feature_query, "feature_id"].astype(str).tolist()
    if not matches:
        raise ClinOmicsError(
            f"Feature query '{feature_query}' was not found in var column '{feature_lookup_col}'"
        )
    unique_matches = list(dict.fromkeys(matches))
    if len(unique_matches) != 1:
        raise ClinOmicsError(
            f"Feature query '{feature_query}' matched multiple feature_id values in var column '{feature_lookup_col}': {unique_matches}"
        )
    return unique_matches[0]


def _is_supported_group_series(values: pd.Series) -> bool:
    non_missing = values.dropna()
    if non_missing.empty:
        return True
    if pd.api.types.is_bool_dtype(non_missing):
        return True
    numeric = pd.to_numeric(non_missing, errors="coerce")
    if numeric.notna().sum() != non_missing.shape[0]:
        return True
    unique = int(numeric.nunique())
    n_used = int(non_missing.shape[0])
    if unique <= 2:
        return True
    # Treat mostly-unique numeric series as continuous, while allowing repeated
    # numeric labels such as binary / low-cardinality coded groups.
    return unique < n_used


def _resolve_label_order(
    values: pd.Series,
    *,
    requested_order: list[str] | tuple[str, ...] | None,
) -> tuple[str, ...]:
    present = tuple(pd.unique(values.astype(str)).tolist())
    if not present:
        return tuple()
    if requested_order is None:
        return tuple(sorted(present))
    requested = [str(v) for v in requested_order]
    missing = [label for label in requested if label not in present]
    if missing:
        raise ClinOmicsError(f"group_order contains groups absent after filtering: {missing}")
    extras = [label for label in present if label not in requested]
    return tuple(requested + extras)


def prepare_feature_vs_obs_comparison(
    dataset: CanonicalDataset,
    *,
    feature: str,
    obs_field: str,
    layer: str | None = None,
    group_order: list[str] | tuple[str, ...] | None = None,
    feature_lookup_col: str | None = None,
) -> FeatureObsComparison:
    frame = _resolve_feature_frame(dataset, layer=layer)
    resolved_feature = resolve_feature_id(
        dataset,
        feature=feature,
        feature_lookup_col=feature_lookup_col,
    )

    groups = _resolve_obs_series(dataset, obs_field=obs_field)
    if not _is_supported_group_series(groups):
        raise ClinOmicsError(
            f"Obs field must be categorical-like for feature-vs-obs plotting: {obs_field}"
        )

    values = pd.to_numeric(frame[resolved_feature], errors="coerce")
    combined = pd.DataFrame({"group": groups, "value": values}, index=frame.index.astype(str))
    if combined.index.has_duplicates:
        raise ClinOmicsError("Feature matrix index must be unique for feature-vs-obs plotting")

    n_total = int(combined.shape[0])
    n_missing_group = int(combined["group"].isna().sum())
    n_missing_feature = int(combined["value"].isna().sum())

    plot_data = combined.dropna(subset=["group", "value"]).copy()
    plot_data["group"] = plot_data["group"].astype(str)
    ordered = _resolve_label_order(plot_data["group"], requested_order=group_order)

    if len(ordered) < 2:
        raise ClinOmicsError("At least two groups are required after filtering missing values")

    plot_data["group"] = pd.Categorical(plot_data["group"], categories=list(ordered), ordered=True)
    plot_data = plot_data.sort_values(["group", "value"], kind="stable").reset_index(names="sample_id")

    return FeatureObsComparison(
        data=plot_data,
        feature=resolved_feature,
        feature_query=str(feature),
        feature_lookup_col=feature_lookup_col,
        obs_field=obs_field,
        layer=layer,
        n_total=n_total,
        n_used=int(plot_data.shape[0]),
        n_missing_group=n_missing_group,
        n_missing_feature=n_missing_feature,
        groups=ordered,
    )


def prepare_two_feature_scatter_data(
    dataset: CanonicalDataset,
    *,
    x_feature: str,
    y_feature: str,
    layer: str | None = None,
    label_field: str | None = None,
    label_order: list[str] | tuple[str, ...] | None = None,
    x_feature_lookup_col: str | None = None,
    y_feature_lookup_col: str | None = None,
) -> TwoFeatureScatterData:
    frame = _resolve_feature_frame(dataset, layer=layer)
    if frame.index.has_duplicates:
        raise ClinOmicsError("Feature matrix index must be unique for feature-feature scatter plotting")

    resolved_x_feature = resolve_feature_id(
        dataset,
        feature=x_feature,
        feature_lookup_col=x_feature_lookup_col,
    )
    resolved_y_feature = resolve_feature_id(
        dataset,
        feature=y_feature,
        feature_lookup_col=y_feature_lookup_col,
    )

    x_values = pd.to_numeric(frame[resolved_x_feature], errors="coerce")
    y_values = pd.to_numeric(frame[resolved_y_feature], errors="coerce")
    combined = pd.DataFrame(
        {
            "x_value": x_values,
            "y_value": y_values,
        },
        index=frame.index.astype(str),
    )
    n_total = int(combined.shape[0])
    n_missing_x = int(combined["x_value"].isna().sum())
    n_missing_y = int(combined["y_value"].isna().sum())

    labels: tuple[str, ...] = tuple()
    n_missing_label = 0
    if label_field is not None:
        label_series = _resolve_obs_series(dataset, obs_field=label_field)
        if not _is_supported_group_series(label_series):
            raise ClinOmicsError(
                f"Label field must be categorical-like for feature-feature scatter plotting: {label_field}"
            )
        combined["label"] = label_series
        n_missing_label = int(combined["label"].isna().sum())
        plot_data = combined.dropna(subset=["x_value", "y_value", "label"]).copy()
        plot_data["label"] = plot_data["label"].astype(str)
        labels = _resolve_label_order(plot_data["label"], requested_order=label_order)
        plot_data["label"] = pd.Categorical(plot_data["label"], categories=list(labels), ordered=True)
        plot_data = plot_data.sort_values(["label", "x_value", "y_value"], kind="stable")
    else:
        plot_data = combined.dropna(subset=["x_value", "y_value"]).copy()

    plot_data = plot_data.reset_index(names="sample_id")
    return TwoFeatureScatterData(
        data=plot_data,
        x_feature=resolved_x_feature,
        x_feature_query=str(x_feature),
        x_feature_lookup_col=x_feature_lookup_col,
        y_feature=resolved_y_feature,
        y_feature_query=str(y_feature),
        y_feature_lookup_col=y_feature_lookup_col,
        label_field=label_field,
        layer=layer,
        n_total=n_total,
        n_used=int(plot_data.shape[0]),
        n_missing_x=n_missing_x,
        n_missing_y=n_missing_y,
        n_missing_label=n_missing_label,
        labels=labels,
    )


def _format_p_value(p_value: float) -> str:
    if p_value < 1e-4:
        return f"{p_value:.2e}"
    return f"{p_value:.4f}"


def mann_whitney_two_group(comparison: FeatureObsComparison) -> MannWhitneyResult:
    groups = list(comparison.groups)
    if len(groups) != 2:
        raise ClinOmicsError(
            "Mann-Whitney annotation requires exactly two groups after filtering"
        )

    group_a, group_b = groups
    values_a = comparison.data.loc[comparison.data["group"] == group_a, "value"].to_numpy(dtype=float)
    values_b = comparison.data.loc[comparison.data["group"] == group_b, "value"].to_numpy(dtype=float)
    n_a = int(values_a.shape[0])
    n_b = int(values_b.shape[0])
    if n_a == 0 or n_b == 0:
        raise ClinOmicsError("Mann-Whitney annotation requires both groups to be non-empty")

    all_values = pd.Series(list(values_a) + list(values_b), dtype=float)
    ranks = all_values.rank(method="average").to_numpy(dtype=float)
    rank_sum_a = float(ranks[:n_a].sum())
    u_a = rank_sum_a - (n_a * (n_a + 1) / 2.0)

    n_total = n_a + n_b
    mu_u = n_a * n_b / 2.0

    tie_counts = all_values.value_counts(dropna=False).to_numpy(dtype=float)
    tie_term = float(((tie_counts ** 3) - tie_counts).sum())
    if n_total <= 1:
        variance_u = 0.0
    else:
        variance_u = (n_a * n_b / 12.0) * (
            (n_total + 1.0) - tie_term / (n_total * (n_total - 1.0))
        )

    if variance_u <= 0.0:
        p_value = 1.0
    else:
        sigma_u = sqrt(variance_u)
        z = (abs(u_a - mu_u) - 0.5) / sigma_u
        p_value = float(erfc(z / sqrt(2.0)))
        p_value = min(max(p_value, 0.0), 1.0)

    return MannWhitneyResult(
        group_a=group_a,
        group_b=group_b,
        n_a=n_a,
        n_b=n_b,
        u_statistic=float(u_a),
        p_value=p_value,
    )


def format_mann_whitney_label(result: MannWhitneyResult) -> str:
    return f"p = {_format_p_value(result.p_value)}\n(Mann-Whitney)"


__all__ = [
    "FeatureObsComparison",
    "MannWhitneyResult",
    "TwoFeatureScatterData",
    "format_mann_whitney_label",
    "resolve_feature_id",
    "mann_whitney_two_group",
    "prepare_feature_vs_obs_comparison",
    "prepare_two_feature_scatter_data",
]
