from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

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
        if pd.api.types.is_numeric_dtype(color_values):
            scatter = ax.scatter(x, y, c=color_values.to_numpy(), **scatter_kwargs)
            fig.colorbar(scatter, ax=ax)
        else:
            categories = pd.Series(color_values, dtype="object").fillna("NA")
            for category in categories.unique().tolist():
                mask = categories == category
                ax.scatter(x[mask], y[mask], label=str(category), **scatter_kwargs)
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
