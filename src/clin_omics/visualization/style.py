from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping


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
