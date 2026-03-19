from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from clin_omics.dataset import CanonicalDataset
from clin_omics.exceptions import ClinOmicsError
from clin_omics.visualization.save import save_figure
from clin_omics.visualization.style import PlotConfig, resolve_plot_config


def _resolve_obs_values(dataset: CanonicalDataset, field: str) -> pd.Series:
    if field not in dataset.obs.columns:
        raise ClinOmicsError(f"Unknown obs field: {field}")
    return dataset.obs[field]


def _coerce_numeric(values: pd.Series) -> pd.Series:
    return pd.to_numeric(values, errors="coerce")


def _is_discrete_numeric(values: pd.Series, *, max_unique: int) -> bool:
    numeric = _coerce_numeric(values).dropna()
    if numeric.empty:
        return False
    arr = numeric.to_numpy(dtype=float)
    if not np.isfinite(arr).all():
        return False
    if not np.allclose(arr, np.round(arr)):
        return False
    n_unique = int(numeric.nunique())
    n_used = int(numeric.shape[0])
    return n_unique <= int(max_unique) and n_unique <= max(3, int(np.sqrt(max(n_used, 1))))


def summarize_obs_field(
    dataset: CanonicalDataset,
    *,
    field: str,
    max_categories: int = 12,
) -> dict[str, Any]:
    values = _resolve_obs_values(dataset, field)
    numeric = _coerce_numeric(values)
    n_total = int(values.shape[0])
    n_used = int(values.notna().sum())
    n_missing = int(values.isna().sum())
    n_unique = int(values.dropna().nunique())
    is_numeric_like = bool(numeric.notna().sum() == values.notna().sum() and n_used > 0)

    if pd.api.types.is_bool_dtype(values) or _is_discrete_numeric(values, max_unique=max_categories):
        plot_type = "bar"
    elif is_numeric_like:
        plot_type = "hist"
    else:
        plot_type = "bar"

    return {
        "field": field,
        "plot_type": plot_type,
        "n_total": n_total,
        "n_used": n_used,
        "n_missing": n_missing,
        "n_unique": n_unique,
    }



def plot_obs_field(
    dataset: CanonicalDataset,
    *,
    field: str,
    title: str | None = None,
    bins: int = 30,
    max_categories: int = 12,
    rotate_xticks: float = 45.0,
    config: PlotConfig | dict | None = None,
    out_prefix: str | Path | None = None,
):
    resolved = resolve_plot_config(config)
    values = _resolve_obs_values(dataset, field)
    summary = summarize_obs_field(dataset, field=field, max_categories=max_categories)

    fig, ax = plt.subplots(figsize=resolved.figsize)

    if summary["plot_type"] == "hist":
        numeric = _coerce_numeric(values).dropna()
        ax.hist(numeric.to_numpy(), bins=int(bins))
        ax.set_xlabel(field, fontsize=resolved.label_fontsize)
        ax.set_ylabel("Count", fontsize=resolved.label_fontsize)
    else:
        categories = pd.Series(values, dtype="object").where(values.notna(), "NA")
        counts = categories.value_counts(dropna=False)
        if len(counts) > max_categories:
            top = counts.iloc[: max_categories - 1]
            other_count = int(counts.iloc[max_categories - 1 :].sum())
            counts = pd.concat([top, pd.Series({"Other": other_count})])
        ax.bar(range(len(counts)), counts.to_numpy())
        ax.set_xticks(range(len(counts)))
        ax.set_xticklabels([str(v) for v in counts.index.tolist()], rotation=rotate_xticks, ha="right")
        ax.set_xlabel(field, fontsize=resolved.label_fontsize)
        ax.set_ylabel("Count", fontsize=resolved.label_fontsize)

    display_title = title or f"{field} (n={summary['n_used']}/{summary['n_total']}, missing={summary['n_missing']})"
    ax.set_title(display_title, fontsize=resolved.title_fontsize)
    ax.tick_params(axis="both", labelsize=resolved.tick_fontsize)

    save_figure(fig, out_prefix, config=resolved)
    return fig, ax, summary
