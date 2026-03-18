from __future__ import annotations

import argparse
import json
from pathlib import Path

from clin_omics.analysis import KNNLeidenClustering, PCAEmbedding, UMAPEmbedding
from clin_omics.dataset import CanonicalDataset
from clin_omics.export import export_assignments_table
from clin_omics.visualization import plot_embedding


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the graph-based bulk RNA-seq downstream flow from a preprocessed H5 dataset.",
    )
    parser.add_argument("--dataset-h5", required=True, type=Path, help="Preprocessed dataset H5 path.")
    parser.add_argument("--outdir", required=True, type=Path, help="Output directory.")
    parser.add_argument("--source-layer", default="log_cpm", help="Dataset layer used for PCA before graph construction.")
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


def run_graph_from_h5_flow(args: argparse.Namespace) -> dict[str, object]:
    outdir = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    dataset = CanonicalDataset.load_h5(args.dataset_h5)
    if args.source_layer not in dataset.layers:
        raise ValueError(f"Layer '{args.source_layer}' not found in dataset. Available: {sorted(dataset.layers.keys())}")

    processed = PCAEmbedding(
        n_components=args.n_pca_components,
        source_layer=args.source_layer,
        key="pca_graph",
    ).fit_transform(dataset)

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

    final_path = outdir / "bulk_rnaseq_graph_from_h5_output_dataset.h5"
    processed.save_h5(final_path)

    assignment_table = export_assignments_table(processed, key="cluster_leiden_graph")
    assignment_table.to_csv(outdir / "cluster_leiden_graph.csv", index=False)

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
        "run_umap": bool(args.run_umap),
        "leiden_neighbors": int(args.leiden_neighbors),
        "leiden_resolution": float(args.leiden_resolution),
    }
    (outdir / "bulk_rnaseq_graph_from_h5_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    return summary



def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    summary = run_graph_from_h5_flow(args)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
