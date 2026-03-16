from __future__ import annotations

from pathlib import Path

from matplotlib.figure import Figure

from clin_omics.visualization.style import PlotConfig, normalize_out_prefix, resolve_plot_config


def save_figure(
    fig: Figure,
    out_prefix: str | Path | None,
    *,
    config: PlotConfig | dict | None = None,
) -> list[Path]:
    resolved = resolve_plot_config(config)
    prefix = normalize_out_prefix(out_prefix)
    if prefix is None:
        return []
    prefix.parent.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []
    if resolved.save_png:
        png_path = prefix.with_suffix('.png')
        fig.savefig(png_path, dpi=resolved.dpi, bbox_inches=resolved.bbox_inches)
        saved.append(png_path)
    if resolved.save_svg:
        svg_path = prefix.with_suffix('.svg')
        fig.savefig(svg_path, bbox_inches=resolved.bbox_inches)
        saved.append(svg_path)
    return saved
