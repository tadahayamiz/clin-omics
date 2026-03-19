from __future__ import annotations

import argparse
import json
from pathlib import Path

from clin_omics.dataset import CanonicalDataset
from clin_omics.visualization import plot_obs_field


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot a summary chart for an obs field from an H5 dataset.")
    parser.add_argument("--dataset-h5", required=True, type=Path, help="Input dataset H5 path.")
    parser.add_argument("--field", required=True, help="obs field to summarize.")
    parser.add_argument("--outdir", required=True, type=Path, help="Output directory.")
    parser.add_argument("--title", default=None, help="Optional plot title override.")
    parser.add_argument("--bins", type=int, default=30, help="Number of histogram bins for continuous fields.")
    parser.add_argument("--max-categories", type=int, default=12, help="Maximum categories to show before collapsing the remainder into 'Other'.")
    parser.add_argument("--rotate-xticks", type=float, default=45.0, help="Rotation angle for categorical x tick labels.")
    parser.add_argument("--fontsize", type=float, default=14.0, help="Base font size.")
    parser.add_argument("--dpi", type=int, default=150, help="PNG DPI.")
    return parser


def run_plot_obs_field_from_h5(args: argparse.Namespace) -> dict[str, object]:
    args.outdir.mkdir(parents=True, exist_ok=True)
    dataset = CanonicalDataset.load_h5(args.dataset_h5)
    _, _, summary = plot_obs_field(
        dataset,
        field=args.field,
        title=args.title,
        bins=args.bins,
        max_categories=args.max_categories,
        rotate_xticks=args.rotate_xticks,
        out_prefix=args.outdir / f"obs_{args.field}",
        config={
            "fontsize": args.fontsize,
            "title_fontsize": args.fontsize + 2,
            "label_fontsize": args.fontsize,
            "tick_fontsize": max(args.fontsize - 2, 1),
            "legend_fontsize": max(args.fontsize - 2, 1),
            "dpi": args.dpi,
            "save_png": True,
            "save_svg": True,
        },
    )
    result = {
        "input_dataset": str(args.dataset_h5),
        "field": args.field,
        "out_prefix": str(args.outdir / f"obs_{args.field}"),
        **summary,
    }
    (args.outdir / f"obs_{args.field}_summary.json").write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    summary = run_plot_obs_field_from_h5(args)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
