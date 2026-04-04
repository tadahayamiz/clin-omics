from __future__ import annotations

import argparse
import json
from pathlib import Path

from clin_omics.analysis import prepare_feature_vs_obs_comparison
from clin_omics.dataset import CanonicalDataset
from clin_omics.visualization import plot_feature_vs_obs
from clin_omics.visualization.style import PlotConfig, resolve_plot_config


def _parse_group_color(values: list[str] | None) -> dict[str, str] | None:
    if not values:
        return None
    parsed: dict[str, str] = {}
    for item in values:
        if "=" not in item:
            raise ValueError(f"Invalid --group-color entry: {item}. Expected GROUP=COLOR")
        group, color = item.split("=", 1)
        group = group.strip()
        color = color.strip()
        if not group or not color:
            raise ValueError(f"Invalid --group-color entry: {item}. Expected GROUP=COLOR")
        parsed[group] = color
    return parsed


def _build_plot_config(args: argparse.Namespace) -> PlotConfig | dict:
    config: dict[str, object] = {}
    if args.fontsize is not None:
        fontsize = float(args.fontsize)
        config["fontsize"] = fontsize
        config["title_fontsize"] = fontsize + 2
        config["label_fontsize"] = fontsize
        config["tick_fontsize"] = max(fontsize - 2, 1)
        config["legend_fontsize"] = max(fontsize - 2, 1)
    if args.dpi is not None:
        config["dpi"] = int(args.dpi)
    if args.width is not None or args.height is not None:
        config["figsize"] = (float(args.width or 6.0), float(args.height or 5.0))
    if args.save_png is not None:
        config["save_png"] = bool(args.save_png)
    if args.save_svg is not None:
        config["save_svg"] = bool(args.save_svg)
    if args.marker_size is not None:
        config["marker_size"] = float(args.marker_size)
    if args.line_width is not None:
        config["line_width"] = float(args.line_width)
    if args.alpha is not None:
        config["alpha"] = float(args.alpha)
    if args.jitter is not None:
        config["jitter"] = float(args.jitter)
    if args.yscale is not None:
        config["yscale"] = str(args.yscale)
    if args.show_top_spine is not None:
        config["show_top_spine"] = bool(args.show_top_spine)
    if args.show_right_spine is not None:
        config["show_right_spine"] = bool(args.show_right_spine)
    return resolve_plot_config(config)



def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot one feature against one obs grouping field from an H5 dataset.")
    parser.add_argument("--dataset-h5", required=True, type=Path, help="Input dataset H5 path.")
    parser.add_argument("--feature", required=True, help="Feature name from X or the selected layer.")
    parser.add_argument("--obs-field", required=True, help="Categorical-like obs field used for grouping.")
    parser.add_argument("--feature-lookup-col", default=None, help="Optional var column used to resolve --feature to a unique feature_id.")
    parser.add_argument("--outdir", required=True, type=Path, help="Output directory.")
    parser.add_argument("--layer", default=None, help="Optional source layer. If omitted, use X.")
    parser.add_argument("--title", default=None, help="Optional plot title override.")
    parser.add_argument("--xlabel", default=None, help="Optional x-axis label override.")
    parser.add_argument("--ylabel", default=None, help="Optional y-axis label override.")
    parser.add_argument("--out-prefix-name", default=None, help="Optional output prefix file stem.")
    parser.add_argument("--control-group", default=None, help="Optional control group name for default gray assignment.")
    parser.add_argument("--group-order", nargs="+", default=None, help="Optional explicit group order.")
    parser.add_argument("--group-color", action="append", default=None, help="Override a group color as GROUP=COLOR. Repeatable.")
    parser.add_argument("--show-box", dest="show_box", action="store_true", default=True, help="Overlay a box summary.")
    parser.add_argument("--no-box", dest="show_box", action="store_false", help="Disable box overlay.")
    parser.add_argument("--annotate-mann-whitney", action="store_true", default=False, help="Add a two-group Mann-Whitney p-value annotation.")
    parser.add_argument("--fontsize", type=float, default=None, help="Base font size.")
    parser.add_argument("--dpi", type=int, default=None, help="Figure DPI.")
    parser.add_argument("--width", type=float, default=None, help="Figure width in inches.")
    parser.add_argument("--height", type=float, default=None, help="Figure height in inches.")
    parser.add_argument("--png", dest="save_png", action="store_true", default=None, help="Save PNG output.")
    parser.add_argument("--no-png", dest="save_png", action="store_false", help="Disable PNG output.")
    parser.add_argument("--svg", dest="save_svg", action="store_true", default=None, help="Save SVG output.")
    parser.add_argument("--no-svg", dest="save_svg", action="store_false", help="Disable SVG output.")
    parser.add_argument("--marker-size", type=float, default=None, help="Marker size for strip points.")
    parser.add_argument("--line-width", type=float, default=None, help="Line width for box and summary elements.")
    parser.add_argument("--alpha", type=float, default=None, help="Point alpha transparency.")
    parser.add_argument("--jitter", type=float, default=None, help="Horizontal jitter strength.")
    parser.add_argument("--yscale", choices=["linear", "log", "symlog", "logit"], default=None, help="Y-axis scale.")
    parser.add_argument("--show-top-spine", dest="show_top_spine", action="store_true", default=None, help="Show the top spine.")
    parser.add_argument("--hide-top-spine", dest="show_top_spine", action="store_false", help="Hide the top spine.")
    parser.add_argument("--show-right-spine", dest="show_right_spine", action="store_true", default=None, help="Show the right spine.")
    parser.add_argument("--hide-right-spine", dest="show_right_spine", action="store_false", help="Hide the right spine.")
    return parser



def run_plot_feature_vs_obs_from_h5(args: argparse.Namespace) -> dict[str, object]:
    args.outdir.mkdir(parents=True, exist_ok=True)
    dataset = CanonicalDataset.load_h5(args.dataset_h5)
    comparison = prepare_feature_vs_obs_comparison(
        dataset,
        feature=args.feature,
        obs_field=args.obs_field,
        layer=args.layer,
        group_order=args.group_order,
        feature_lookup_col=getattr(args, "feature_lookup_col", None),
    )
    color_overrides = _parse_group_color(args.group_color)
    prefix_name = args.out_prefix_name or f"feature_vs_obs_{args.obs_field}_{args.feature}"
    out_prefix = args.outdir / prefix_name
    _, _, plot_summary = plot_feature_vs_obs(
        comparison,
        title=args.title,
        xlabel=args.xlabel,
        ylabel=args.ylabel,
        show_box=bool(args.show_box),
        control_group=args.control_group,
        color_overrides=color_overrides,
        config=_build_plot_config(args),
        out_prefix=out_prefix,
        annotate_mann_whitney=bool(getattr(args, "annotate_mann_whitney", False)),
    )
    result = {
        "input_dataset": str(args.dataset_h5),
        "feature": args.feature,
        "obs_field": args.obs_field,
        "feature_lookup_col": getattr(args, "feature_lookup_col", None),
        "layer": args.layer,
        "out_prefix": str(out_prefix),
        **plot_summary,
    }
    summary_path = args.outdir / f"{prefix_name}_summary.json"
    summary_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    return result



def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    summary = run_plot_feature_vs_obs_from_h5(args)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
