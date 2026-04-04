from __future__ import annotations

from pathlib import Path
from typing import Mapping

import matplotlib.pyplot as plt
import numpy as np

from clin_omics.analysis.association import (
    FeatureObsComparison,
    TwoFeatureScatterData,
    format_mann_whitney_label,
    mann_whitney_two_group,
)
from clin_omics.visualization.save import save_figure
from clin_omics.visualization.style import PlotConfig, resolve_group_colors, resolve_plot_config


def plot_feature_vs_obs(
    comparison: FeatureObsComparison,
    *,
    title: str | None = None,
    ylabel: str | None = None,
    xlabel: str | None = None,
    show_box: bool = True,
    control_group: str | None = None,
    color_overrides: Mapping[str, str] | None = None,
    config: PlotConfig | dict | None = None,
    out_prefix: str | Path | None = None,
    annotate_mann_whitney: bool = False,
):
    resolved = resolve_plot_config(config)
    groups = list(comparison.groups)
    color_map = resolve_group_colors(groups, control_group=control_group, color_overrides=color_overrides)

    fig, ax = plt.subplots(figsize=resolved.figsize)
    rng = np.random.default_rng(0)

    if show_box:
        box_data = [
            comparison.data.loc[comparison.data["group"] == group, "value"].to_numpy(dtype=float)
            for group in groups
        ]
        bp = ax.boxplot(
            box_data,
            positions=np.arange(1, len(groups) + 1),
            widths=0.45,
            patch_artist=True,
            showfliers=False,
            medianprops={"linewidth": resolved.line_width},
            boxprops={"linewidth": resolved.line_width},
            whiskerprops={"linewidth": resolved.line_width},
            capprops={"linewidth": resolved.line_width},
        )
        for patch, group in zip(bp["boxes"], groups, strict=True):
            patch.set_facecolor(color_map[group])
            patch.set_alpha(0.18)
            patch.set_edgecolor(color_map[group])
        for median, group in zip(bp["medians"], groups, strict=True):
            median.set_color(color_map[group])

    for idx, group in enumerate(groups, start=1):
        values = comparison.data.loc[comparison.data["group"] == group, "value"].to_numpy(dtype=float)
        x = idx + rng.uniform(-resolved.jitter, resolved.jitter, size=values.shape[0])
        ax.scatter(
            x,
            values,
            s=resolved.marker_size,
            alpha=resolved.alpha,
            color=color_map[group],
            linewidths=resolved.marker_edge_width,
            marker=resolved.marker,
        )

    ax.set_xticks(np.arange(1, len(groups) + 1))
    ax.set_xticklabels(groups)
    ax.set_xlabel(xlabel or comparison.obs_field, fontsize=resolved.label_fontsize)
    ax.set_ylabel(ylabel or comparison.feature, fontsize=resolved.label_fontsize)
    display_title = title or (
        f"{comparison.feature} vs {comparison.obs_field} "
        f"(n={comparison.n_used}/{comparison.n_total})"
    )
    ax.set_title(display_title, fontsize=resolved.title_fontsize)
    ax.tick_params(axis="both", labelsize=resolved.tick_fontsize)
    ax.set_yscale(resolved.yscale)
    ax.spines["top"].set_visible(resolved.show_top_spine)
    ax.spines["right"].set_visible(resolved.show_right_spine)

    stat_annotation = None
    if annotate_mann_whitney:
        result = mann_whitney_two_group(comparison)
        label = format_mann_whitney_label(result)
        current_ymin, current_ymax = ax.get_ylim()
        if resolved.yscale == "log" and current_ymin > 0 and current_ymax > 0:
            bracket_y = current_ymax * 1.08
            text_y = current_ymax * 1.14
            ax.set_ylim(current_ymin, current_ymax * 1.24)
        else:
            span = current_ymax - current_ymin
            if span <= 0:
                span = max(abs(current_ymax), 1.0)
            bracket_y = current_ymax + 0.08 * span
            text_y = current_ymax + 0.16 * span
            ax.set_ylim(current_ymin, current_ymax + 0.28 * span)
        ax.plot(
            [1.0, 1.0, 2.0, 2.0],
            [bracket_y, bracket_y + 0.02 * (ax.get_ylim()[1] - ax.get_ylim()[0]), bracket_y + 0.02 * (ax.get_ylim()[1] - ax.get_ylim()[0]), bracket_y],
            color="black",
            linewidth=resolved.line_width,
        )
        ax.text(1.5, text_y, label, ha="center", va="bottom", fontsize=resolved.tick_fontsize)
        stat_annotation = result.to_summary() | {"label": label}

    save_figure(fig, out_prefix, config=resolved)
    summary = comparison.to_summary() | {
        "color_map": color_map,
        "show_box": show_box,
        "stat_annotation": stat_annotation,
    }
    return fig, ax, summary


def plot_feature_vs_feature_scatter(
    scatter_data: TwoFeatureScatterData,
    *,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    control_label: str | None = None,
    color_overrides: Mapping[str, str] | None = None,
    config: PlotConfig | dict | None = None,
    out_prefix: str | Path | None = None,
    show_legend: bool = True,
):
    resolved = resolve_plot_config(config)
    fig, ax = plt.subplots(figsize=resolved.figsize)

    data = scatter_data.data.copy()
    color_map: dict[str, str] | None = None
    if scatter_data.label_field is not None and scatter_data.labels:
        labels = list(scatter_data.labels)
        color_map = resolve_group_colors(
            labels,
            control_group=control_label,
            color_overrides=color_overrides,
        )
        for label in labels:
            label_mask = data["label"].astype(str) == str(label)
            subset = data.loc[label_mask]
            ax.scatter(
                subset["x_value"].to_numpy(dtype=float),
                subset["y_value"].to_numpy(dtype=float),
                s=resolved.marker_size,
                alpha=resolved.alpha,
                c=color_map[label],
                marker=resolved.marker,
                linewidths=resolved.marker_edge_width,
                label=label,
            )
        if show_legend:
            ax.legend(fontsize=resolved.legend_fontsize, frameon=False)
    else:
        ax.scatter(
            data["x_value"].to_numpy(dtype=float),
            data["y_value"].to_numpy(dtype=float),
            s=resolved.marker_size,
            alpha=resolved.alpha,
            c=DEFAULT_SINGLE_COLOR,
            marker=resolved.marker,
            linewidths=resolved.marker_edge_width,
        )

    ax.set_xlabel(xlabel or scatter_data.x_feature, fontsize=resolved.label_fontsize)
    ax.set_ylabel(ylabel or scatter_data.y_feature, fontsize=resolved.label_fontsize)
    ax.set_title(
        title or f"{scatter_data.x_feature} vs {scatter_data.y_feature} (n={scatter_data.n_used}/{scatter_data.n_total})",
        fontsize=resolved.title_fontsize,
    )
    ax.tick_params(axis="both", labelsize=resolved.tick_fontsize)
    ax.set_xscale(resolved.xscale)
    ax.set_yscale(resolved.yscale)
    ax.spines["top"].set_visible(resolved.show_top_spine)
    ax.spines["right"].set_visible(resolved.show_right_spine)

    save_figure(fig, out_prefix, config=resolved)
    summary = scatter_data.to_summary() | {
        "color_map": color_map,
        "show_legend": bool(show_legend and scatter_data.label_field is not None and scatter_data.labels),
    }
    return fig, ax, summary


DEFAULT_SINGLE_COLOR = "#4477AA"

__all__ = ["plot_feature_vs_obs", "plot_feature_vs_feature_scatter"]
