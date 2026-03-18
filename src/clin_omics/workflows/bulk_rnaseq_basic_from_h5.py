from __future__ import annotations

import argparse
import json
from pathlib import Path

from clin_omics.analysis import KMeansClustering, PCAEmbedding, UMAPEmbedding
from clin_omics.dataset import CanonicalDataset
from clin_omics.export import export_assignments_table
from clin_omics.visualization import plot_embedding


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the basic bulk RNA-seq downstream flow from a preprocessed H5 dataset.",
    )
    parser.add_argument("--dataset-h5", required=True, type=Path, help="Preprocessed dataset H5 path.")
    parser.add_argument("--outdir", required=True, type=Path, help="Output directory.")
    parser.add_argument("--source-layer", default="log_cpm", help="Dataset layer used for PCA.")
    parser.add_argument("--plot-color", default=None, help="Optional obs column for plot coloring. Defaults to basic cluster labels when k-means is enabled.")
    parser.add_argument("--n-pca-components", type=int, default=2, help="Number of PCA components.")
    parser.add_argument("--n-umap-components", type=int, default=2, help="Number of UMAP components.")
    parser.add_argument("--umap-neighbors", type=int, default=15, help="UMAP number of neighbors.")
    parser.add_argument("--umap-min-dist", type=float, default=0.1, help="UMAP minimum distance.")
    parser.add_argument("--kmeans-clusters", type=int, default=2, help="Number of k-means clusters for basic clustering.")
    parser.add_argument("--skip-kmeans", action="store_true", help="Skip k-means clustering and related assignment export.")
    parser.add_argument("--run-umap", action="store_true", help="Also run UMAP on the selected source layer.")
    parser.add_argument("--random-state", type=int, default=0, help="Random seed.")
    parser.add_argument("--fontsize", type=float, default=14.0, help="Base font size for basic-flow figures.")
    parser.add_argument("--dpi", type=int, default=150, help="PNG DPI.")
    return parser


def run_basic_from_h5_flow(args: argparse.Namespace) -> dict[str, object]:
    outdir = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    dataset = CanonicalDataset.load_h5(args.dataset_h5)
    if args.source_layer not in dataset.layers:
        raise ValueError(f"Layer '{args.source_layer}' not found in dataset. Available: {sorted(dataset.layers.keys())}")

    processed = PCAEmbedding(
        n_components=args.n_pca_components,
        source_layer=args.source_layer,
        key="pca_basic",
    ).fit_transform(dataset)
    if not args.skip_kmeans:
        processed = KMeansClustering(
            n_clusters=args.kmeans_clusters,
            embedding_key="pca_basic",
            key="cluster_kmeans_basic",
            random_state=args.random_state,
        ).fit_predict(processed)

    plot_color = args.plot_color or ("cluster_kmeans_basic" if not args.skip_kmeans else None)
    plot_embedding(
        processed,
        embedding_key="pca_basic",
        color=plot_color,
        title="PCA basic",
        out_prefix=outdir / "pca_basic",
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
            source_layer=args.source_layer,
            key="umap_basic",
            n_neighbors=args.umap_neighbors,
            min_dist=args.umap_min_dist,
            random_state=args.random_state,
        ).fit_transform(processed)
        plot_embedding(
            processed,
            embedding_key="umap_basic",
            color=plot_color,
            title="UMAP basic",
            out_prefix=outdir / "umap_basic",
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

    final_path = outdir / "bulk_rnaseq_basic_from_h5_output_dataset.h5"
    processed.save_h5(final_path)

    if not args.skip_kmeans:
        assignment_table = export_assignments_table(processed, key="cluster_kmeans_basic")
        assignment_table.to_csv(outdir / "cluster_kmeans_basic.csv", index=False)

    summary = {
        "input_dataset": str(args.dataset_h5),
        "output_dataset": str(final_path),
        "source_layer": args.source_layer,
        "n_samples": int(processed.X.shape[0]),
        "n_features": int(processed.X.shape[1]),
        "layers": sorted(processed.layers.keys()),
        "embeddings": sorted(processed.embeddings.keys()),
        "assignments": sorted(processed.assignments.keys()),
        "plot_color": plot_color,
        "run_kmeans": bool(not args.skip_kmeans),
        "run_umap": bool(args.run_umap),
    }
    (outdir / "bulk_rnaseq_basic_from_h5_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    return summary



def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    summary = run_basic_from_h5_flow(args)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
