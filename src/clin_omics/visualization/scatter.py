from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas.api.types import CategoricalDtype

from clin_omics.dataset import CanonicalDataset
from clin_omics.exceptions import ClinOmicsError
from clin_omics.visualization.save import save_figure
from clin_omics.visualization.style import PlotConfig, resolve_plot_config


def _resolve_embedding(dataset: CanonicalDataset, embedding_key: str) -> pd.DataFrame:
    if embedding_key not in dataset.embeddings:
        raise ClinOmicsError(f"Unknown embedding key: {embedding_key}")
    embedding = dataset.embeddings[embedding_key]
    if embedding.shape[1] < 2:
        raise ClinOmicsError("Embedding must have at least two columns for scatter plotting.")
    return embedding


def _resolve_color_values(dataset: CanonicalDataset, embedding: pd.DataFrame, color: str) -> pd.Series:
    if color in dataset.obs.columns:
        return dataset.obs.set_index("sample_id").loc[embedding.index, color]
    if color in dataset.assignments:
        return dataset.assignments[color].reindex(embedding.index)
    raise ClinOmicsError(f"Unknown color key: {color}. Expected an obs column or assignment key.")


def _is_discrete_numeric(values: pd.Series, *, max_unique: int = 20) -> bool:
    non_null = values.dropna()
    if non_null.empty or not pd.api.types.is_numeric_dtype(values):
        return False
    arr = non_null.to_numpy(dtype=float)
    if not np.isfinite(arr).all():
        return False
    if not np.allclose(arr, np.round(arr)):
        return False
    return int(non_null.nunique()) <= int(max_unique)


def _should_use_categorical_color(dataset: CanonicalDataset, color: str, color_values: pd.Series) -> bool:
    if color in dataset.assignments:
        return True
    if pd.api.types.is_bool_dtype(color_values) or isinstance(color_values.dtype, CategoricalDtype):
        return True
    if not pd.api.types.is_numeric_dtype(color_values):
        return True
    return _is_discrete_numeric(color_values)


def plot_embedding(
    dataset: CanonicalDataset,
    *,
    embedding_key: str,
    color: str | None = None,
    title: str | None = None,
    alpha: float = 1.0,
    size: float = 40.0,
    show_legend: bool = True,
    config: PlotConfig | dict | None = None,
    out_prefix: str | Path | None = None,
):
    resolved = resolve_plot_config(config)
    embedding = _resolve_embedding(dataset, embedding_key)

    fig, ax = plt.subplots(figsize=resolved.figsize)
    x = embedding.iloc[:, 0]
    y = embedding.iloc[:, 1]

    scatter_kwargs = {"alpha": float(alpha), "s": float(size)}
    if color is not None:
        color_values = _resolve_color_values(dataset, embedding, color)
        if _should_use_categorical_color(dataset, color, color_values):
            categories = pd.Series(color_values, dtype="object")
            categories = categories.where(categories.notna(), "NA")
            for category in categories.unique().tolist():
                mask = categories == category
                ax.scatter(x[mask], y[mask], label=str(category), **scatter_kwargs)
            if show_legend:
                ax.legend(fontsize=resolved.legend_fontsize)
        else:
            valid_mask = color_values.notna().to_numpy()
            scatter = ax.scatter(x[valid_mask], y[valid_mask], c=color_values.loc[valid_mask].to_numpy(), **scatter_kwargs)
            fig.colorbar(scatter, ax=ax)
            if (~valid_mask).any():
                ax.scatter(x[~valid_mask], y[~valid_mask], c="lightgray", label="NA", **scatter_kwargs)
                if show_legend:
                    ax.legend(fontsize=resolved.legend_fontsize)
    else:
        ax.scatter(x, y, **scatter_kwargs)

    ax.set_xlabel(str(embedding.columns[0]), fontsize=resolved.label_fontsize)
    ax.set_ylabel(str(embedding.columns[1]), fontsize=resolved.label_fontsize)
    ax.set_title(title or embedding_key, fontsize=resolved.title_fontsize)
    ax.tick_params(axis='both', labelsize=resolved.tick_fontsize)

    save_figure(fig, out_prefix, config=resolved)
    return fig, ax
