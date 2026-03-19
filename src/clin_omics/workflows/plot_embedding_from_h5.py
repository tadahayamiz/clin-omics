from __future__ import annotations

import argparse
import json
from pathlib import Path

from clin_omics.dataset import CanonicalDataset
from clin_omics.visualization import plot_embedding


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot an existing embedding from an H5 dataset without recomputing PCA/UMAP/clustering.",
    )
    parser.add_argument("--dataset-h5", required=True, type=Path, help="Input dataset H5 path.")
    parser.add_argument("--outdir", required=True, type=Path, help="Output directory.")
    parser.add_argument(
        "--embedding-key",
        required=True,
        help="Existing embedding key in the dataset, e.g. pca_graph or umap_graph.",
    )
    parser.add_argument(
        "--plot-color",
        default=None,
        help="Optional obs/assignment key used for coloring. Defaults to a cluster assignment when available.",
    )
    parser.add_argument(
        "--title",
        default=None,
        help="Optional figure title. Defaults to the embedding key.",
    )
    parser.add_argument(
        "--out-prefix-name",
        default=None,
        help="Optional output filename stem. Defaults to <embedding-key>_plot.",
    )
    parser.add_argument("--fontsize", type=float, default=14.0, help="Base font size for figures.")
    parser.add_argument("--dpi", type=int, default=150, help="PNG DPI.")
    return parser


def _default_plot_color(dataset: CanonicalDataset, embedding_key: str) -> str | None:
    assignments = dataset.assignments
    if embedding_key.endswith("_graph") and "cluster_leiden_graph" in assignments:
        return "cluster_leiden_graph"
    if embedding_key.endswith("_basic") and "cluster_kmeans_basic" in assignments:
        return "cluster_kmeans_basic"
    for key in ("cluster_leiden_graph", "cluster_kmeans_basic"):
        if key in assignments:
            return key
    if assignments:
        return sorted(assignments.keys())[0]
    return None


def run_plot_embedding_from_h5(args: argparse.Namespace) -> dict[str, object]:
    outdir = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    dataset = CanonicalDataset.load_h5(args.dataset_h5)
    if args.embedding_key not in dataset.embeddings:
        raise ValueError(
            f"Embedding '{args.embedding_key}' not found in dataset. Available: {sorted(dataset.embeddings.keys())}"
        )

    plot_color = args.plot_color or _default_plot_color(dataset, args.embedding_key)
    out_prefix_name = args.out_prefix_name or f"{args.embedding_key}_plot"
    out_prefix = outdir / out_prefix_name

    plot_embedding(
        dataset,
        embedding_key=args.embedding_key,
        color=plot_color,
        title=args.title or args.embedding_key,
        out_prefix=out_prefix,
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

    summary = {
        "input_dataset": str(args.dataset_h5),
        "embedding_key": args.embedding_key,
        "plot_color": plot_color,
        "out_prefix": str(out_prefix),
        "available_embeddings": sorted(dataset.embeddings.keys()),
        "available_assignments": sorted(dataset.assignments.keys()),
    }
    (outdir / "plot_embedding_from_h5_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    return summary



def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    summary = run_plot_embedding_from_h5(args)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
