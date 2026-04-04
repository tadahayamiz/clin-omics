from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping

TOL_BRIGHT = (
    "#4477AA",
    "#EE6677",
    "#228833",
    "#CCBB44",
    "#66CCEE",
    "#AA3377",
    "#BBBBBB",
)
DEFAULT_CONTROL_COLOR = "#7F7F7F"
DEFAULT_TREATMENT_COLOR = "#4477AA"


@dataclass(frozen=True)
class PlotConfig:
    figsize: tuple[float, float] = (6.0, 5.0)
    fontsize: float = 14.0
    title_fontsize: float = 16.0
    label_fontsize: float = 14.0
    tick_fontsize: float = 12.0
    legend_fontsize: float = 12.0
    dpi: int = 300
    save_png: bool = True
    save_svg: bool = True
    bbox_inches: str = "tight"
    marker_size: float = 18.0
    line_width: float = 1.0
    alpha: float = 0.9
    jitter: float = 0.12
    yscale: str = "linear"
    show_top_spine: bool = False
    show_right_spine: bool = False



def resolve_plot_config(config: PlotConfig | Mapping[str, Any] | None = None) -> PlotConfig:
    if config is None:
        return PlotConfig()
    if isinstance(config, PlotConfig):
        return config
    allowed = PlotConfig.__dataclass_fields__.keys()
    updates = {k: v for k, v in dict(config).items() if k in allowed}
    return replace(PlotConfig(), **updates)



def normalize_out_prefix(out_prefix: str | Path | None) -> Path | None:
    if out_prefix is None:
        return None
    return Path(out_prefix)



def resolve_group_colors(
    groups: list[str] | tuple[str, ...],
    *,
    control_group: str | None = None,
    color_overrides: Mapping[str, str] | None = None,
) -> dict[str, str]:
    resolved_groups = [str(g) for g in groups]
    overrides = {str(k): str(v) for k, v in dict(color_overrides or {}).items()}

    if control_group is not None and str(control_group) not in resolved_groups:
        raise ValueError(f"control_group is not present in groups: {control_group}")

    color_map: dict[str, str] = {}
    remaining = [g for g in resolved_groups if g not in overrides]

    if control_group is not None and str(control_group) in remaining:
        color_map[str(control_group)] = DEFAULT_CONTROL_COLOR
        remaining = [g for g in remaining if g != str(control_group)]
        if len(remaining) == 1:
            color_map[remaining[0]] = DEFAULT_TREATMENT_COLOR
            remaining = []

    palette_iter = iter(TOL_BRIGHT)
    for group in remaining:
        color_map[group] = next(palette_iter)

    color_map.update(overrides)
    return {group: color_map[group] for group in resolved_groups}
