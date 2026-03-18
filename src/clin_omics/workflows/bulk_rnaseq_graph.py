from __future__ import annotations

import argparse
import json
from pathlib import Path

from clin_omics.analysis import KNNLeidenClustering, PCAEmbedding, UMAPEmbedding
from clin_omics.export import export_assignments_table
from clin_omics.preprocess import BulkRNASeqPreprocessor
from clin_omics.visualization import plot_embedding

from .bulk_rnaseq_basic import _prepare_dataset


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the graph-based bulk RNA-seq flow on count tables.",
    )
    parser.add_argument("--x", required=True, type=Path, help="Samples x features table path.")
    parser.add_argument("--obs", required=True, type=Path, help="Sample metadata table path.")
    parser.add_argument("--var", required=True, type=Path, help="Feature metadata table path.")
    parser.add_argument("--outdir", required=True, type=Path, help="Output directory.")
    parser.add_argument("--sample-id-col", default="sample_id", help="Sample ID column in X and obs.")
    parser.add_argument("--feature-id-col", default="feature_id", help="Feature ID column in var.")
    parser.add_argument("--min-count", type=float, default=10.0, help="Minimum count threshold for filtering.")
    parser.add_argument("--min-samples", type=int, default=1, help="Minimum number of samples meeting min-count.")
    parser.add_argument("--prior-count", type=float, default=1.0, help="Prior count for logCPM.")
    parser.add_argument("--make-zscore", action="store_true", help="Also generate zscore_log_cpm.")
    parser.add_argument("--plot-color", default=None, help="Optional obs column for plot coloring. Defaults to Leiden cluster labels.")
    parser.add_argument("--n-pca-components", type=int, default=10, help="Number of PCA components before graph construction.")
    parser.add_argument("--leiden-neighbors", type=int, default=15, help="kNN size for graph construction.")
    parser.add_argument("--leiden-resolution", type=float, default=1.0, help="Leiden resolution parameter.")
    parser.add_argument("--run-umap", action="store_true", help="Also run UMAP on the PCA embedding.")
    parser.add_argument("--n-umap-components", type=int, default=2, help="Number of UMAP components.")
    parser.add_argument("--umap-neighbors", type=int, default=15, help="UMAP number of neighbors.")
    parser.add_argument("--umap-min-dist", type=float, default=0.1, help="UMAP minimum distance.")
    parser.add_argument("--random-state", type=int, default=0, help="Random seed.")
    parser.add_argument("--fontsize", type=float, default=14.0, help="Base font size for graph-flow figures.")
    parser.add_argument("--dpi", type=int, default=150, help="PNG DPI.")
    return parser


def run_graph_flow(args: argparse.Namespace) -> dict[str, object]:
    outdir = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    dataset = _prepare_dataset(
        x_path=args.x,
        obs_path=args.obs,
        var_path=args.var,
        sample_id_col=args.sample_id_col,
        feature_id_col=args.feature_id_col,
    )
    raw_path = outdir / "bulk_rnaseq_graph_input_dataset.h5"
    dataset.save_h5(raw_path)

    processed = BulkRNASeqPreprocessor(
        min_count=args.min_count,
        min_samples=args.min_samples,
        prior_count=args.prior_count,
        make_zscore=args.make_zscore,
    ).fit_transform(dataset)

    processed = PCAEmbedding(
        n_components=args.n_pca_components,
        source_layer="log_cpm",
        key="pca_graph",
    ).fit_transform(processed)

    processed = KNNLeidenClustering(
        n_neighbors=args.leiden_neighbors,
        resolution=args.leiden_resolution,
        embedding_key="pca_graph",
        key="cluster_leiden_graph",
        random_state=args.random_state,
    ).fit_predict(processed)

    plot_color = args.plot_color or "cluster_leiden_graph"
    plot_embedding(
        processed,
        embedding_key="pca_graph",
        color=plot_color,
        title="PCA graph",
        out_prefix=outdir / "pca_graph",
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

    if args.run_umap:
        processed = UMAPEmbedding(
            n_components=args.n_umap_components,
            source_layer="pca_graph",
            key="umap_graph",
            n_neighbors=args.umap_neighbors,
            min_dist=args.umap_min_dist,
            random_state=args.random_state,
        ).fit_transform(processed)
        plot_embedding(
            processed,
            embedding_key="umap_graph",
            color=plot_color,
            title="UMAP graph",
            out_prefix=outdir / "umap_graph",
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

    final_path = outdir / "bulk_rnaseq_graph_processed_dataset.h5"
    processed.save_h5(final_path)

    assignment_table = export_assignments_table(processed, key="cluster_leiden_graph")
    assignment_table.to_csv(outdir / "cluster_leiden_graph.csv", index=False)

    summary = {
        "input_dataset": str(raw_path),
        "output_dataset": str(final_path),
        "n_samples": int(processed.X.shape[0]),
        "n_features": int(processed.X.shape[1]),
        "layers": sorted(processed.layers.keys()),
        "embeddings": sorted(processed.embeddings.keys()),
        "assignments": sorted(processed.assignments.keys()),
        "plot_color": plot_color,
        "run_umap": bool(args.run_umap),
        "leiden_neighbors": int(args.leiden_neighbors),
        "leiden_resolution": float(args.leiden_resolution),
    }
    (outdir / "bulk_rnaseq_graph_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    summary = run_graph_flow(args)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
