from __future__ import annotations

import argparse
import json
from pathlib import Path

from clin_omics.analysis import KMeansClustering, PCAEmbedding, UMAPEmbedding
from clin_omics.dataset import CanonicalDataset
from clin_omics.export import export_assignments_table
from clin_omics.io import read_table
from clin_omics.preprocess import BulkRNASeqPreprocessor
from clin_omics.visualization import plot_embedding


def _normalize_x_table(X, sample_id_col: str):
    if sample_id_col in X.columns:
        X_norm = X.set_index(sample_id_col)
    else:
        first_col = X.columns[0]
        candidate = X[first_col]
        if candidate.isna().any():
            raise ValueError(f"Missing sample IDs in X column: {first_col}")
        if candidate.duplicated().any():
            dup = candidate[candidate.duplicated()].astype(str).tolist()[:5]
            raise ValueError(
                "Unable to infer sample IDs from X. "
                f"Column '{first_col}' contains duplicates. Example: {dup}"
            )
        X_norm = X.set_index(first_col)

    if X_norm.index.isna().any():
        raise ValueError("X index contains missing sample IDs.")
    if X_norm.index.duplicated().any():
        dup = X_norm.index[X_norm.index.duplicated()].astype(str).tolist()[:5]
        raise ValueError(f"X contains duplicate sample IDs. Example: {dup}")

    X_norm.index = X_norm.index.astype(str)
    X_norm.index.name = "sample_id"
    return X_norm


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the basic bulk RNA-seq flow on count tables.",
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
    parser.add_argument("--run-umap", action="store_true", help="Also run UMAP on log_cpm.")
    parser.add_argument("--plot-color", default=None, help="Optional obs column for plot coloring. Defaults to basic cluster labels when k-means is enabled.")
    parser.add_argument("--n-pca-components", type=int, default=2, help="Number of PCA components.")
    parser.add_argument("--n-umap-components", type=int, default=2, help="Number of UMAP components.")
    parser.add_argument("--umap-neighbors", type=int, default=15, help="UMAP number of neighbors.")
    parser.add_argument("--umap-min-dist", type=float, default=0.1, help="UMAP minimum distance.")
    parser.add_argument("--kmeans-clusters", type=int, default=2, help="Number of k-means clusters for basic clustering.")
    parser.add_argument("--skip-kmeans", action="store_true", help="Skip k-means clustering and related assignment export.")
    parser.add_argument("--random-state", type=int, default=0, help="Random seed.")
    parser.add_argument("--fontsize", type=float, default=14.0, help="Base font size for basic-flow figures.")
    parser.add_argument("--dpi", type=int, default=150, help="PNG DPI.")
    return parser


def _prepare_dataset(
    x_path: Path,
    obs_path: Path,
    var_path: Path,
    sample_id_col: str,
    feature_id_col: str,
) -> CanonicalDataset:
    X = read_table(x_path)
    obs = read_table(obs_path)
    var = read_table(var_path)

    if sample_id_col not in obs.columns:
        raise ValueError(f"Missing sample ID column in obs: {sample_id_col}")
    if feature_id_col not in var.columns:
        raise ValueError(f"Missing feature ID column in var: {feature_id_col}")

    X = _normalize_x_table(X, sample_id_col)
    obs_indexed = obs.set_index(sample_id_col)
    obs_indexed.index = obs_indexed.index.astype(str)
    obs_indexed.index.name = "sample_id"

    missing_samples = [sample for sample in X.index if sample not in obs_indexed.index]
    if missing_samples:
        preview = ", ".join(missing_samples[:5])
        raise ValueError(
            f"obs is missing samples present in X ({len(missing_samples)} missing). Example: {preview}"
        )

    obs_indexed = obs_indexed.loc[X.index].reset_index()

    missing_features = [feature for feature in X.columns if feature not in set(var[feature_id_col].tolist())]
    if missing_features:
        preview = ", ".join(missing_features[:5])
        raise ValueError(
            f"X contains features absent from var ({len(missing_features)} missing). Example: {preview}"
        )

    var_indexed = var.set_index(feature_id_col)
    var_indexed.index.name = "feature_id"
    var_indexed = var_indexed.loc[X.columns].copy()
    var_indexed.index.name = "feature_id"
    var_indexed = var_indexed.reset_index()

    return CanonicalDataset(X=X, obs=obs_indexed, var=var_indexed)


def run_basic_flow(args: argparse.Namespace) -> dict[str, object]:
    outdir = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    dataset = _prepare_dataset(
        x_path=args.x,
        obs_path=args.obs,
        var_path=args.var,
        sample_id_col=args.sample_id_col,
        feature_id_col=args.feature_id_col,
    )
    raw_path = outdir / "bulk_rnaseq_basic_input_dataset.h5"
    dataset.save_h5(raw_path)

    bulk = BulkRNASeqPreprocessor(
        min_count=args.min_count,
        min_samples=args.min_samples,
        prior_count=args.prior_count,
        make_zscore=args.make_zscore,
    )
    processed = bulk.fit_transform(dataset)
    processed = PCAEmbedding(
        n_components=args.n_pca_components,
        source_layer="log_cpm",
        key="pca_basic",
    ).fit_transform(processed)
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
            source_layer="log_cpm",
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

    final_path = outdir / "bulk_rnaseq_basic_processed_dataset.h5"
    processed.save_h5(final_path)

    if not args.skip_kmeans:
        assignment_table = export_assignments_table(processed, key="cluster_kmeans_basic")
        assignment_table.to_csv(outdir / "cluster_kmeans_basic.csv", index=False)

    summary = {
        "input_dataset": str(raw_path),
        "output_dataset": str(final_path),
        "n_samples": int(processed.X.shape[0]),
        "n_features": int(processed.X.shape[1]),
        "layers": sorted(processed.layers.keys()),
        "embeddings": sorted(processed.embeddings.keys()),
        "assignments": sorted(processed.assignments.keys()),
        "plot_color": plot_color,
        "run_kmeans": bool(not args.skip_kmeans),
        "run_umap": bool(args.run_umap),
    }
    (outdir / "bulk_rnaseq_basic_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    summary = run_basic_flow(args)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
