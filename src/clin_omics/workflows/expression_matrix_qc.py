from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from typing import Sequence

from clin_omics.analysis import (
    append_expression_qc_to_obs,
    summarize_expression_matrix_qc,
    write_expression_qc_tables,
)
from clin_omics.dataset import CanonicalDataset
from clin_omics.visualization import plot_expression_qc_metric, plot_expression_qc_scatter


def _parse_float_list(text: str) -> list[float]:
    if text.strip() == "":
        return []
    return [float(item.strip()) for item in text.split(",") if item.strip()]


def _qc_plot_config(fontsize: float, dpi: int) -> dict[str, object]:
    return {
        "fontsize": fontsize,
        "title_fontsize": fontsize + 2,
        "label_fontsize": fontsize,
        "tick_fontsize": max(fontsize - 2, 1),
        "legend_fontsize": max(fontsize - 2, 1),
        "dpi": dpi,
        "save_png": True,
        "save_svg": True,
    }


def write_expression_matrix_qc_report(
    dataset: CanonicalDataset,
    outdir: str | Path,
    *,
    prefix: str = "expression_matrix_qc",
    counts_layer: str | None = None,
    cpm_layer: str | None = None,
    count_thresholds: Sequence[float] = (5.0, 10.0),
    cpm_thresholds: Sequence[float] = (1.0,),
    top_n_features: int = 10,
    correlation_layer: str | None = None,
    save_plots: bool = True,
    fontsize: float = 14.0,
    dpi: int = 150,
) -> dict[str, object]:
    output_dir = Path(outdir)
    output_dir.mkdir(parents=True, exist_ok=True)

    result = summarize_expression_matrix_qc(
        dataset,
        counts_layer=counts_layer,
        cpm_layer=cpm_layer,
        count_thresholds=count_thresholds,
        cpm_thresholds=cpm_thresholds,
        top_n_features=top_n_features,
        correlation_layer=correlation_layer,
    )
    paths = write_expression_qc_tables(result, output_dir, prefix=prefix)

    plot_paths: dict[str, str] = {}
    if save_plots:
        config = _qc_plot_config(fontsize=fontsize, dpi=dpi)
        total_prefix = output_dir / f"{prefix}_total_counts"
        detected_prefix = output_dir / f"{prefix}_detected_genes_count_gt0"
        top_prefix = output_dir / f"{prefix}_top_{top_n_features}_gene_fraction"
        scatter_prefix = output_dir / f"{prefix}_detected_genes_vs_total_counts"
        corr_prefix = output_dir / f"{prefix}_sample_correlation"

        fig, _ = plot_expression_qc_metric(
            result.sample_qc,
            metric="qc_total_counts",
            title="Expression QC: total counts",
            config=config,
            out_prefix=total_prefix,
        )
        plt.close(fig)
        fig, _ = plot_expression_qc_metric(
            result.sample_qc,
            metric="qc_n_detected_genes_count_gt0",
            title="Expression QC: detected genes (count > 0)",
            config=config,
            out_prefix=detected_prefix,
        )
        plt.close(fig)
        fig, _ = plot_expression_qc_metric(
            result.sample_qc,
            metric=f"qc_top_{top_n_features}_gene_fraction",
            title=f"Expression QC: top {top_n_features} gene fraction",
            config=config,
            out_prefix=top_prefix,
        )
        plt.close(fig)
        fig, _ = plot_expression_qc_scatter(
            result.sample_qc,
            x="qc_total_counts",
            y="qc_n_detected_genes_count_gt0",
            title="Expression QC: detected genes vs total counts",
            config=config,
            out_prefix=scatter_prefix,
        )
        plt.close(fig)
        fig, _ = plot_expression_qc_metric(
            result.sample_qc,
            metric="qc_median_sample_correlation",
            title="Expression QC: median sample correlation",
            config=config,
            out_prefix=corr_prefix,
        )
        plt.close(fig)

        for key, prefix_path in {
            "total_counts": total_prefix,
            "detected_genes_count_gt0": detected_prefix,
            f"top_{top_n_features}_gene_fraction": top_prefix,
            "detected_genes_vs_total_counts": scatter_prefix,
            "sample_correlation": corr_prefix,
        }.items():
            plot_paths[f"{key}_png"] = str(prefix_path.with_suffix(".png"))
            plot_paths[f"{key}_svg"] = str(prefix_path.with_suffix(".svg"))

    return {
        "sample_qc": paths["sample_qc"],
        "feature_qc": paths["feature_qc"],
        "summary": paths["summary"],
        "plots": plot_paths,
        "metrics": result.summary,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize expression matrix QC from a canonical dataset H5 file.")
    parser.add_argument("--dataset-h5", required=True, type=Path, help="Input canonical dataset H5 path.")
    parser.add_argument("--outdir", required=True, type=Path, help="Output directory.")
    parser.add_argument("--prefix", default="expression_matrix_qc", help="Output filename prefix.")
    parser.add_argument("--counts-layer", default="X", help="Count-like matrix layer to summarize. Use X for dataset.X.")
    parser.add_argument("--cpm-layer", default=None, help="Optional CPM layer. If omitted, CPM is computed from counts.")
    parser.add_argument("--count-thresholds", default="5,10", help="Comma-separated count thresholds for detection metrics.")
    parser.add_argument("--cpm-thresholds", default="1", help="Comma-separated CPM thresholds for detection metrics.")
    parser.add_argument("--top-n-features", type=int, default=10, help="Number of top genes for concentration metric.")
    parser.add_argument("--correlation-layer", default=None, help="Optional layer for sample correlation. If omitted, log1p(counts) is used.")
    parser.add_argument("--append-obs-out", default=None, type=Path, help="Optional output H5 path with sample QC columns appended to obs.")
    parser.add_argument("--overwrite-obs-qc", action="store_true", help="Overwrite existing obs QC columns when appending.")
    parser.add_argument("--no-plots", action="store_true", help="Disable PNG/SVG QC plots.")
    parser.add_argument("--fontsize", type=float, default=14.0, help="Base font size for QC plots.")
    parser.add_argument("--dpi", type=int, default=150, help="PNG DPI.")
    return parser


def run_expression_matrix_qc_flow(args: argparse.Namespace) -> dict[str, object]:
    dataset = CanonicalDataset.load_h5(args.dataset_h5)
    counts_layer = None if args.counts_layer == "X" else args.counts_layer
    cpm_layer = None if args.cpm_layer in {None, ""} else args.cpm_layer
    correlation_layer = None if args.correlation_layer in {None, ""} else args.correlation_layer
    count_thresholds = _parse_float_list(args.count_thresholds)
    cpm_thresholds = _parse_float_list(args.cpm_thresholds)

    report = write_expression_matrix_qc_report(
        dataset,
        args.outdir,
        prefix=args.prefix,
        counts_layer=counts_layer,
        cpm_layer=cpm_layer,
        count_thresholds=count_thresholds,
        cpm_thresholds=cpm_thresholds,
        top_n_features=args.top_n_features,
        correlation_layer=correlation_layer,
        save_plots=not args.no_plots,
        fontsize=args.fontsize,
        dpi=args.dpi,
    )

    if args.append_obs_out is not None:
        result = summarize_expression_matrix_qc(
            dataset,
            counts_layer=counts_layer,
            cpm_layer=cpm_layer,
            count_thresholds=count_thresholds,
            cpm_thresholds=cpm_thresholds,
            top_n_features=args.top_n_features,
            correlation_layer=correlation_layer,
        )
        updated = append_expression_qc_to_obs(dataset, result.sample_qc, overwrite=args.overwrite_obs_qc)
        updated.save_h5(args.append_obs_out)
        report["append_obs_out"] = str(args.append_obs_out)

    return report


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    summary = run_expression_matrix_qc_flow(args)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
