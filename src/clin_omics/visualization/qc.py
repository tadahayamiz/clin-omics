from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from clin_omics.visualization.save import save_figure
from clin_omics.visualization.style import PlotConfig, resolve_plot_config


def plot_expression_qc_metric(
    sample_qc: pd.DataFrame,
    *,
    metric: str,
    title: str | None = None,
    bins: int = 30,
    config: PlotConfig | dict | None = None,
    out_prefix: str | Path | None = None,
):
    if metric not in sample_qc.columns:
        raise ValueError(f"Unknown sample QC metric: {metric}")
    values = pd.to_numeric(sample_qc[metric], errors="coerce").dropna()
    resolved = resolve_plot_config(config)

    fig, ax = plt.subplots(figsize=resolved.figsize)
    finite_values = values.to_numpy(dtype=float)
    finite_values = finite_values[np.isfinite(finite_values)]
    if finite_values.size == 0:
        hist_bins = 1
    elif np.isclose(float(finite_values.min()), float(finite_values.max())):
        hist_bins = 1
    else:
        hist_bins = int(bins)
    ax.hist(finite_values, bins=hist_bins)
    ax.set_xlabel(metric, fontsize=resolved.label_fontsize)
    ax.set_ylabel("Sample count", fontsize=resolved.label_fontsize)
    display_title = title or f"{metric} (n={values.shape[0]}/{sample_qc.shape[0]})"
    ax.set_title(display_title, fontsize=resolved.title_fontsize)
    ax.tick_params(axis="both", labelsize=resolved.tick_fontsize)
    save_figure(fig, out_prefix, config=resolved)
    return fig, ax


def plot_expression_qc_scatter(
    sample_qc: pd.DataFrame,
    *,
    x: str,
    y: str,
    title: str | None = None,
    alpha: float = 0.85,
    size: float = 36.0,
    config: PlotConfig | dict | None = None,
    out_prefix: str | Path | None = None,
):
    missing = [metric for metric in (x, y) if metric not in sample_qc.columns]
    if missing:
        raise ValueError(f"Unknown sample QC metric(s): {missing}")

    plot_data = sample_qc[[x, y]].apply(pd.to_numeric, errors="coerce").dropna()
    resolved = resolve_plot_config(config)

    fig, ax = plt.subplots(figsize=resolved.figsize)
    ax.scatter(plot_data[x].to_numpy(dtype=float), plot_data[y].to_numpy(dtype=float), alpha=alpha, s=size)
    ax.set_xlabel(x, fontsize=resolved.label_fontsize)
    ax.set_ylabel(y, fontsize=resolved.label_fontsize)
    display_title = title or f"{y} vs {x} (n={plot_data.shape[0]}/{sample_qc.shape[0]})"
    ax.set_title(display_title, fontsize=resolved.title_fontsize)
    ax.tick_params(axis="both", labelsize=resolved.tick_fontsize)
    save_figure(fig, out_prefix, config=resolved)
    return fig, ax
