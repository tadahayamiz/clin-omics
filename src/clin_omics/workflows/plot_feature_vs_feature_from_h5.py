from __future__ import annotations

import argparse
import json
from pathlib import Path

from clin_omics.analysis import prepare_two_feature_scatter_data
from clin_omics.dataset import CanonicalDataset
from clin_omics.visualization import plot_feature_vs_feature_scatter
from clin_omics.visualization.style import PlotConfig, resolve_plot_config
from clin_omics.workflows.plot_feature_vs_obs_from_h5 import _parse_group_color


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
    if args.marker is not None:
        config["marker"] = str(args.marker)
    if args.marker_edge_width is not None:
        config["marker_edge_width"] = float(args.marker_edge_width)
    if args.alpha is not None:
        config["alpha"] = float(args.alpha)
    if getattr(args, "xscale", None) is not None:
        config["xscale"] = str(args.xscale)
    if getattr(args, "yscale", None) is not None:
        config["yscale"] = str(args.yscale)
    if args.show_top_spine is not None:
        config["show_top_spine"] = bool(args.show_top_spine)
    if args.show_right_spine is not None:
        config["show_right_spine"] = bool(args.show_right_spine)
    return resolve_plot_config(config)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot one feature against another feature from an H5 dataset.")
    parser.add_argument("--dataset-h5", required=True, type=Path, help="Input dataset H5 path.")
    parser.add_argument("--x-feature", required=True, help="Feature name for the x-axis.")
    parser.add_argument("--y-feature", required=True, help="Feature name for the y-axis.")
    parser.add_argument("--x-feature-lookup-col", default=None, help="Optional var column used to resolve --x-feature to a unique feature_id.")
    parser.add_argument("--y-feature-lookup-col", default=None, help="Optional var column used to resolve --y-feature to a unique feature_id.")
    parser.add_argument("--label-field", default=None, help="Optional categorical-like obs field used for point colors.")
    parser.add_argument("--label-order", nargs="+", default=None, help="Optional explicit label order.")
    parser.add_argument("--outdir", required=True, type=Path, help="Output directory.")
    parser.add_argument("--layer", default=None, help="Optional source layer. If omitted, use X.")
    parser.add_argument("--title", default=None, help="Optional plot title override.")
    parser.add_argument("--xlabel", default=None, help="Optional x-axis label override.")
    parser.add_argument("--ylabel", default=None, help="Optional y-axis label override.")
    parser.add_argument("--out-prefix-name", default=None, help="Optional output prefix file stem.")
    parser.add_argument("--control-label", default=None, help="Optional label name for default gray assignment.")
    parser.add_argument("--label-color", action="append", default=None, help="Override a label color as LABEL=COLOR. Repeatable.")
    parser.add_argument("--legend", dest="show_legend", action="store_true", default=True, help="Show legend for labeled scatter.")
    parser.add_argument("--no-legend", dest="show_legend", action="store_false", help="Hide legend.")
    parser.add_argument("--fontsize", type=float, default=None, help="Base font size.")
    parser.add_argument("--dpi", type=int, default=None, help="Figure DPI.")
    parser.add_argument("--width", type=float, default=None, help="Figure width in inches.")
    parser.add_argument("--height", type=float, default=None, help="Figure height in inches.")
    parser.add_argument("--png", dest="save_png", action="store_true", default=None, help="Save PNG output.")
    parser.add_argument("--no-png", dest="save_png", action="store_false", help="Disable PNG output.")
    parser.add_argument("--svg", dest="save_svg", action="store_true", default=None, help="Save SVG output.")
    parser.add_argument("--no-svg", dest="save_svg", action="store_false", help="Disable SVG output.")
    parser.add_argument("--marker-size", type=float, default=None, help="Marker size for points.")
    parser.add_argument("--marker", default=None, help="Marker style passed to matplotlib.")
    parser.add_argument("--marker-edge-width", type=float, default=None, help="Marker edge width.")
    parser.add_argument("--alpha", type=float, default=None, help="Point alpha transparency.")
    parser.add_argument("--show-top-spine", dest="show_top_spine", action="store_true", default=None, help="Show the top spine.")
    parser.add_argument("--hide-top-spine", dest="show_top_spine", action="store_false", help="Hide the top spine.")
    parser.add_argument("--show-right-spine", dest="show_right_spine", action="store_true", default=None, help="Show the right spine.")
    parser.add_argument("--hide-right-spine", dest="show_right_spine", action="store_false", help="Hide the right spine.")
    return parser


def run_plot_feature_vs_feature_from_h5(args: argparse.Namespace) -> dict[str, object]:
    args.outdir.mkdir(parents=True, exist_ok=True)
    dataset = CanonicalDataset.load_h5(args.dataset_h5)
    scatter_data = prepare_two_feature_scatter_data(
        dataset,
        x_feature=args.x_feature,
        y_feature=args.y_feature,
        layer=args.layer,
        label_field=args.label_field,
        label_order=args.label_order,
        x_feature_lookup_col=args.x_feature_lookup_col,
        y_feature_lookup_col=args.y_feature_lookup_col,
    )
    color_overrides = _parse_group_color(args.label_color)
    prefix_name = args.out_prefix_name or f"feature_scatter_{args.x_feature}_vs_{args.y_feature}"
    out_prefix = args.outdir / prefix_name
    _, _, plot_summary = plot_feature_vs_feature_scatter(
        scatter_data,
        title=args.title,
        xlabel=args.xlabel,
        ylabel=args.ylabel,
        control_label=args.control_label,
        color_overrides=color_overrides,
        config=_build_plot_config(args),
        out_prefix=out_prefix,
        show_legend=bool(args.show_legend),
    )
    result = {
        "input_dataset": str(args.dataset_h5),
        "x_feature": args.x_feature,
        "y_feature": args.y_feature,
        "label_field": args.label_field,
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
    summary = run_plot_feature_vs_feature_from_h5(args)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
