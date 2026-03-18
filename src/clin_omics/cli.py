from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from clin_omics import SCHEMA_VERSION, __version__
from clin_omics.analysis import (
    FactorAnalysisEmbedding,
    KMeansClustering,
    KNNLeidenClustering,
    PCAEmbedding,
    UMAPEmbedding,
)
from clin_omics.dataset import CanonicalDataset
from clin_omics.export import export_assignments_table
from clin_omics.io import read_dataset_h5, read_table
from clin_omics.visualization import (
    plot_confusion_matrix,
    plot_embedding,
    plot_pr_curve,
    plot_regression_residuals,
    plot_roc_curve,
)




def _add_plot_config_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--fontsize", type=float, default=None, help="Base font size.")
    parser.add_argument("--dpi", type=int, default=None, help="Figure DPI.")
    parser.add_argument("--width", type=float, default=None, help="Figure width in inches.")
    parser.add_argument("--height", type=float, default=None, help="Figure height in inches.")
    parser.add_argument("--png", dest="save_png", action="store_true", default=None, help="Save PNG output.")
    parser.add_argument("--no-png", dest="save_png", action="store_false", help="Disable PNG output.")
    parser.add_argument("--svg", dest="save_svg", action="store_true", default=None, help="Save SVG output.")
    parser.add_argument("--no-svg", dest="save_svg", action="store_false", help="Disable SVG output.")


def _plot_config_from_args(args: argparse.Namespace) -> dict:
    config: dict[str, object] = {}
    if getattr(args, "fontsize", None) is not None:
        fontsize = float(args.fontsize)
        config["fontsize"] = fontsize
        config["title_fontsize"] = fontsize + 2
        config["label_fontsize"] = fontsize
        config["tick_fontsize"] = max(fontsize - 2, 1)
        config["legend_fontsize"] = max(fontsize - 2, 1)
    if getattr(args, "dpi", None) is not None:
        config["dpi"] = int(args.dpi)
    width = getattr(args, "width", None)
    height = getattr(args, "height", None)
    if width is not None or height is not None:
        config["figsize"] = (float(width or 6.0), float(height or 5.0))
    if getattr(args, "save_png", None) is not None:
        config["save_png"] = bool(args.save_png)
    if getattr(args, "save_svg", None) is not None:
        config["save_svg"] = bool(args.save_svg)
    return config

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="clin-omics")
    subparsers = parser.add_subparsers(dest="command", required=True)

    version_parser = subparsers.add_parser("version", help="Show package version.")
    version_parser.set_defaults(func=_cmd_version)

    inspect_parser = subparsers.add_parser(
        "inspect", help="Inspect a canonical dataset hdf5 file."
    )
    inspect_parser.add_argument("path", type=Path)
    inspect_parser.set_defaults(func=_cmd_inspect)

    validate_parser = subparsers.add_parser(
        "validate", help="Validate a canonical dataset hdf5 file."
    )
    validate_parser.add_argument("path", type=Path)
    validate_parser.set_defaults(func=_cmd_validate)

    build_parser = subparsers.add_parser(
        "build-dataset",
        help="Build a canonical dataset hdf5 file from curated X/obs/var tables.",
    )
    build_parser.add_argument("--x", required=True, type=Path, help="Path to samples x features table.")
    build_parser.add_argument("--obs", required=True, type=Path, help="Path to sample metadata table.")
    build_parser.add_argument("--var", required=True, type=Path, help="Path to feature metadata table.")
    build_parser.add_argument("--out", required=True, type=Path, help="Output .h5 path.")
    build_parser.add_argument(
        "--sample-id-col", default="sample_id", help="Sample ID column name in X/obs tables."
    )
    build_parser.add_argument(
        "--feature-id-col", default="feature_id", help="Feature ID column name in var table."
    )
    build_parser.set_defaults(func=_cmd_build_dataset)

    pca_parser = subparsers.add_parser("pca", help="Run PCA and save an updated dataset.")
    pca_parser.add_argument("--in", dest="input_path", required=True, type=Path, help="Input dataset .h5 path.")
    pca_parser.add_argument("--out", required=True, type=Path, help="Output dataset .h5 path.")
    pca_parser.add_argument("--n-components", type=int, default=2, help="Number of PCA components.")
    pca_parser.add_argument("--source-layer", default=None, help="Optional source layer name. If omitted, use X.")
    pca_parser.add_argument("--key", default="pca", help="Embedding key name.")
    pca_parser.set_defaults(func=_cmd_pca)

    factor_analysis_parser = subparsers.add_parser("factor-analysis", help="Run factor analysis and save an updated dataset.")
    factor_analysis_parser.add_argument("--in", dest="input_path", required=True, type=Path, help="Input dataset .h5 path.")
    factor_analysis_parser.add_argument("--out", required=True, type=Path, help="Output dataset .h5 path.")
    factor_analysis_parser.add_argument("--n-components", type=int, default=2, help="Number of latent factors.")
    factor_analysis_parser.add_argument("--source-layer", default=None, help="Optional source layer name. If omitted, use X.")
    factor_analysis_parser.add_argument("--key", default="factor_analysis", help="Embedding key name.")
    factor_analysis_parser.set_defaults(func=_cmd_factor_analysis)

    umap_parser = subparsers.add_parser("umap", help="Run UMAP and save an updated dataset.")
    umap_parser.add_argument("--in", dest="input_path", required=True, type=Path, help="Input dataset .h5 path.")
    umap_parser.add_argument("--out", required=True, type=Path, help="Output dataset .h5 path.")
    umap_parser.add_argument("--n-components", type=int, default=2, help="Number of UMAP components.")
    umap_input_group = umap_parser.add_mutually_exclusive_group()
    umap_input_group.add_argument("--embedding-key", default=None, help="Optional embedding key to use as input.")
    umap_input_group.add_argument("--source-layer", default=None, help="Optional source layer name.")
    umap_parser.add_argument("--key", default="umap", help="Embedding key name.")
    umap_parser.add_argument("--neighbors", type=int, default=15, help="Number of nearest neighbors.")
    umap_parser.add_argument("--min-dist", type=float, default=0.1, help="Minimum distance parameter.")
    umap_parser.add_argument("--random-state", type=int, default=0, help="Random seed.")
    umap_parser.set_defaults(func=_cmd_umap)

    kmeans_parser = subparsers.add_parser("cluster-kmeans", help="Run k-means clustering and save an updated dataset.")
    kmeans_parser.add_argument("--in", dest="input_path", required=True, type=Path, help="Input dataset .h5 path.")
    kmeans_parser.add_argument("--out", required=True, type=Path, help="Output dataset .h5 path.")
    kmeans_parser.add_argument("--n-clusters", type=int, required=True, help="Number of clusters.")
    kmeans_input_group = kmeans_parser.add_mutually_exclusive_group()
    kmeans_input_group.add_argument("--embedding-key", default=None, help="Optional embedding key to cluster.")
    kmeans_input_group.add_argument("--source-layer", default=None, help="Optional source layer name.")
    kmeans_parser.add_argument("--key", default="cluster_kmeans", help="Assignment key name.")
    kmeans_parser.add_argument("--random-state", type=int, default=0, help="Random seed.")
    kmeans_parser.set_defaults(func=_cmd_cluster_kmeans)

    knn_leiden_parser = subparsers.add_parser("cluster-knn-leiden", help="Run kNN->Leiden clustering and save an updated dataset.")
    knn_leiden_parser.add_argument("--in", dest="input_path", required=True, type=Path, help="Input dataset .h5 path.")
    knn_leiden_parser.add_argument("--out", required=True, type=Path, help="Output dataset .h5 path.")
    knn_leiden_parser.add_argument("--neighbors", type=int, default=15, help="Number of nearest neighbors.")
    knn_leiden_parser.add_argument("--resolution", type=float, default=1.0, help="Leiden resolution parameter.")
    knn_leiden_input_group = knn_leiden_parser.add_mutually_exclusive_group()
    knn_leiden_input_group.add_argument("--embedding-key", default=None, help="Optional embedding key to cluster.")
    knn_leiden_input_group.add_argument("--source-layer", default=None, help="Optional source layer name.")
    knn_leiden_parser.add_argument("--key", default="cluster_knn_leiden", help="Assignment key name.")
    knn_leiden_parser.add_argument("--random-state", type=int, default=0, help="Random seed.")
    knn_leiden_parser.set_defaults(func=_cmd_cluster_knn_leiden)

    export_assignments_parser = subparsers.add_parser("export-assignments", help="Export assignment labels to a table.")
    export_assignments_parser.add_argument("--in", dest="input_path", required=True, type=Path, help="Input dataset .h5 path.")
    export_assignments_parser.add_argument("--key", required=True, help="Assignment key to export.")
    export_assignments_parser.add_argument("--out", required=True, type=Path, help="Output .csv/.tsv/.txt path.")
    export_assignments_parser.set_defaults(func=_cmd_export_assignments)

    plot_embedding_parser = subparsers.add_parser("plot-embedding", help="Plot a 2D embedding and save figure files.")
    plot_embedding_parser.add_argument("--in", dest="input_path", required=True, type=Path, help="Input dataset .h5 path.")
    plot_embedding_parser.add_argument("--embedding-key", required=True, help="Embedding key to plot.")
    plot_embedding_parser.add_argument("--color", default=None, help="Optional obs column or assignment key for color.")
    plot_embedding_parser.add_argument("--title", default=None, help="Optional plot title.")
    plot_embedding_parser.add_argument("--alpha", type=float, default=1.0, help="Point alpha transparency.")
    plot_embedding_parser.add_argument("--size", type=float, default=40.0, help="Point size.")
    plot_embedding_parser.add_argument("--legend", dest="show_legend", action="store_true", default=True, help="Show legend for categorical colors.")
    plot_embedding_parser.add_argument("--no-legend", dest="show_legend", action="store_false", help="Hide legend for categorical colors.")
    plot_embedding_parser.add_argument("--out-prefix", required=True, type=Path, help="Output path prefix without extension.")
    _add_plot_config_args(plot_embedding_parser)
    plot_embedding_parser.set_defaults(func=_cmd_plot_embedding)

    plot_roc_parser = subparsers.add_parser("plot-roc", help="Plot ROC curve from a table.")
    plot_roc_parser.add_argument("--input", required=True, type=Path, help="Input table path.")
    plot_roc_parser.add_argument("--y-true-col", required=True, help="Column containing true binary labels.")
    plot_roc_parser.add_argument("--y-score-col", required=True, help="Column containing prediction scores.")
    plot_roc_parser.add_argument("--title", default=None, help="Optional plot title.")
    plot_roc_parser.add_argument("--out-prefix", required=True, type=Path, help="Output path prefix without extension.")
    _add_plot_config_args(plot_roc_parser)
    plot_roc_parser.set_defaults(func=_cmd_plot_roc)

    plot_pr_parser = subparsers.add_parser("plot-pr", help="Plot precision-recall curve from a table.")
    plot_pr_parser.add_argument("--input", required=True, type=Path, help="Input table path.")
    plot_pr_parser.add_argument("--y-true-col", required=True, help="Column containing true binary labels.")
    plot_pr_parser.add_argument("--y-score-col", required=True, help="Column containing prediction scores.")
    plot_pr_parser.add_argument("--title", default=None, help="Optional plot title.")
    plot_pr_parser.add_argument("--out-prefix", required=True, type=Path, help="Output path prefix without extension.")
    _add_plot_config_args(plot_pr_parser)
    plot_pr_parser.set_defaults(func=_cmd_plot_pr)

    plot_cm_parser = subparsers.add_parser("plot-confusion-matrix", help="Plot confusion matrix from a table.")
    plot_cm_parser.add_argument("--input", required=True, type=Path, help="Input table path.")
    plot_cm_parser.add_argument("--y-true-col", required=True, help="Column containing true labels.")
    plot_cm_parser.add_argument("--y-pred-col", required=True, help="Column containing predicted labels.")
    plot_cm_parser.add_argument("--title", default=None, help="Optional plot title.")
    plot_cm_parser.add_argument("--out-prefix", required=True, type=Path, help="Output path prefix without extension.")
    _add_plot_config_args(plot_cm_parser)
    plot_cm_parser.set_defaults(func=_cmd_plot_confusion_matrix)

    plot_resid_parser = subparsers.add_parser("plot-regression-residuals", help="Plot regression residuals from a table.")
    plot_resid_parser.add_argument("--input", required=True, type=Path, help="Input table path.")
    plot_resid_parser.add_argument("--y-true-col", required=True, help="Column containing true values.")
    plot_resid_parser.add_argument("--y-pred-col", required=True, help="Column containing predicted values.")
    plot_resid_parser.add_argument("--title", default=None, help="Optional plot title.")
    plot_resid_parser.add_argument("--out-prefix", required=True, type=Path, help="Output path prefix without extension.")
    _add_plot_config_args(plot_resid_parser)
    plot_resid_parser.set_defaults(func=_cmd_plot_regression_residuals)

    return parser


def _cmd_version(_: argparse.Namespace) -> int:
    print(__version__)
    return 0


def _cmd_inspect(args: argparse.Namespace) -> int:
    dataset = read_dataset_h5(args.path)
    summary = {
        "dataset_id": dataset.dataset_id,
        "schema_version": dataset.provenance.get("schema_version", SCHEMA_VERSION),
        "shape": [int(dataset.X.shape[0]), int(dataset.X.shape[1])],
        "n_samples": int(dataset.X.shape[0]),
        "n_features": int(dataset.X.shape[1]),
        "obs_columns": dataset.obs.columns.tolist(),
        "var_columns": dataset.var.columns.tolist(),
        "layers": sorted(dataset.layers.keys()),
        "layer_shapes": {name: [int(frame.shape[0]), int(frame.shape[1])] for name, frame in dataset.layers.items()},
        "embeddings": sorted(dataset.embeddings.keys()),
        "embedding_shapes": {name: [int(frame.shape[0]), int(frame.shape[1])] for name, frame in dataset.embeddings.items()},
        "feature_scores": sorted(dataset.feature_scores.keys()),
        "feature_score_shapes": {name: [int(frame.shape[0]), int(frame.shape[1])] for name, frame in dataset.feature_scores.items()},
        "assignments": sorted(dataset.assignments.keys()),
        "assignment_lengths": {name: int(len(series)) for name, series in dataset.assignments.items()},
        "provenance_keys": sorted(dataset.provenance.keys()),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def _cmd_validate(args: argparse.Namespace) -> int:
    try:
        read_dataset_h5(args.path)
    except Exception as exc:  # pragma: no cover - exercised in tests
        print(f"INVALID: {exc}", file=sys.stderr)
        return 1

    print("VALID")
    return 0


def _normalize_x_table_for_cli(X: pd.DataFrame, sample_id_col: str) -> pd.DataFrame:
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


def _cmd_build_dataset(args: argparse.Namespace) -> int:
    try:
        X = read_table(args.x)
        obs = read_table(args.obs)
        var = read_table(args.var)

        if args.sample_id_col not in obs.columns:
            raise ValueError(f"Missing sample ID column in obs: {args.sample_id_col}")
        if args.feature_id_col not in var.columns:
            raise ValueError(f"Missing feature ID column in var: {args.feature_id_col}")

        X = _normalize_x_table_for_cli(X, args.sample_id_col)

        if args.sample_id_col != "sample_id":
            obs = obs.rename(columns={args.sample_id_col: "sample_id"})
        if args.feature_id_col != "feature_id":
            var = var.rename(columns={args.feature_id_col: "feature_id"})

        dataset = CanonicalDataset(X=X, obs=obs, var=var)
        dataset.save_h5(args.out)
    except Exception as exc:  # pragma: no cover - exercised in tests
        print(f"BUILD_FAILED: {exc}", file=sys.stderr)
        return 1

    print(str(args.out))
    return 0




def _cmd_pca(args: argparse.Namespace) -> int:
    try:
        dataset = read_dataset_h5(args.input_path)
        updated = PCAEmbedding(
            n_components=args.n_components,
            source_layer=args.source_layer,
            key=args.key,
        ).fit_transform(dataset)
        updated.save_h5(args.out)
    except Exception as exc:  # pragma: no cover - exercised in tests
        print(f"PCA_FAILED: {exc}", file=sys.stderr)
        return 1
    print(str(args.out))
    return 0


def _cmd_factor_analysis(args: argparse.Namespace) -> int:
    try:
        dataset = read_dataset_h5(args.input_path)
        updated = FactorAnalysisEmbedding(
            n_components=args.n_components,
            source_layer=args.source_layer,
            key=args.key,
        ).fit_transform(dataset)
        updated.save_h5(args.out)
    except Exception as exc:  # pragma: no cover - exercised in tests
        print(f"FACTOR_ANALYSIS_FAILED: {exc}", file=sys.stderr)
        return 1
    print(str(args.out))
    return 0


def _cmd_umap(args: argparse.Namespace) -> int:
    try:
        dataset = read_dataset_h5(args.input_path)
        source_layer = args.embedding_key if args.embedding_key is not None else args.source_layer
        updated = UMAPEmbedding(
            n_components=args.n_components,
            source_layer=source_layer,
            key=args.key,
            n_neighbors=args.neighbors,
            min_dist=args.min_dist,
            random_state=args.random_state,
        ).fit_transform(dataset)
        updated.save_h5(args.out)
    except Exception as exc:  # pragma: no cover - exercised in tests
        print(f"UMAP_FAILED: {exc}", file=sys.stderr)
        return 1
    print(str(args.out))
    return 0


def _cmd_cluster_kmeans(args: argparse.Namespace) -> int:
    try:
        dataset = read_dataset_h5(args.input_path)
        updated = KMeansClustering(
            n_clusters=args.n_clusters,
            embedding_key=args.embedding_key,
            source_layer=args.source_layer,
            key=args.key,
            random_state=args.random_state,
        ).fit_predict(dataset)
        updated.save_h5(args.out)
    except Exception as exc:  # pragma: no cover - exercised in tests
        print(f"CLUSTER_KMEANS_FAILED: {exc}", file=sys.stderr)
        return 1
    print(str(args.out))
    return 0


def _cmd_cluster_knn_leiden(args: argparse.Namespace) -> int:
    try:
        dataset = read_dataset_h5(args.input_path)
        updated = KNNLeidenClustering(
            n_neighbors=args.neighbors,
            resolution=args.resolution,
            embedding_key=args.embedding_key,
            source_layer=args.source_layer,
            key=args.key,
            random_state=args.random_state,
        ).fit_predict(dataset)
        updated.save_h5(args.out)
    except Exception as exc:  # pragma: no cover - exercised in tests
        print(f"CLUSTER_KNN_LEIDEN_FAILED: {exc}", file=sys.stderr)
        return 1
    print(str(args.out))
    return 0

def _cmd_export_assignments(args: argparse.Namespace) -> int:
    try:
        dataset = read_dataset_h5(args.input_path)
        table = export_assignments_table(dataset, args.key)
        suffix = args.out.suffix.lower()
        if suffix == ".csv":
            sep = ","
        elif suffix in {".tsv", ".txt"}:
            sep = "\t"
        else:
            raise ValueError("Unsupported output suffix. Use .csv, .tsv, or .txt")
        table.to_csv(args.out, index=False, sep=sep)
    except Exception as exc:  # pragma: no cover - exercised in tests
        print(f"EXPORT_ASSIGNMENTS_FAILED: {exc}", file=sys.stderr)
        return 1

    print(str(args.out))
    return 0


def _cmd_plot_embedding(args: argparse.Namespace) -> int:
    try:
        dataset = read_dataset_h5(args.input_path)
        plot_embedding(
            dataset,
            embedding_key=args.embedding_key,
            color=args.color,
            title=args.title,
            alpha=args.alpha,
            size=args.size,
            show_legend=args.show_legend,
            config=_plot_config_from_args(args),
            out_prefix=args.out_prefix,
        )
    except Exception as exc:  # pragma: no cover - exercised in tests
        print(f"PLOT_EMBEDDING_FAILED: {exc}", file=sys.stderr)
        return 1
    print(str(args.out_prefix))
    return 0


def _read_required_columns(path: Path, columns: list[str]):
    table = read_table(path)
    missing = [col for col in columns if col not in table.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    return table


def _cmd_plot_roc(args: argparse.Namespace) -> int:
    try:
        table = _read_required_columns(args.input, [args.y_true_col, args.y_score_col])
        plot_roc_curve(
            y_true=table[args.y_true_col],
            y_score=table[args.y_score_col],
            title=args.title,
            config=_plot_config_from_args(args),
            out_prefix=args.out_prefix,
        )
    except Exception as exc:  # pragma: no cover - exercised in tests
        print(f"PLOT_ROC_FAILED: {exc}", file=sys.stderr)
        return 1
    print(str(args.out_prefix))
    return 0


def _cmd_plot_pr(args: argparse.Namespace) -> int:
    try:
        table = _read_required_columns(args.input, [args.y_true_col, args.y_score_col])
        plot_pr_curve(
            y_true=table[args.y_true_col],
            y_score=table[args.y_score_col],
            title=args.title,
            config=_plot_config_from_args(args),
            out_prefix=args.out_prefix,
        )
    except Exception as exc:  # pragma: no cover - exercised in tests
        print(f"PLOT_PR_FAILED: {exc}", file=sys.stderr)
        return 1
    print(str(args.out_prefix))
    return 0


def _cmd_plot_confusion_matrix(args: argparse.Namespace) -> int:
    try:
        table = _read_required_columns(args.input, [args.y_true_col, args.y_pred_col])
        plot_confusion_matrix(
            y_true=table[args.y_true_col],
            y_pred=table[args.y_pred_col],
            title=args.title,
            config=_plot_config_from_args(args),
            out_prefix=args.out_prefix,
        )
    except Exception as exc:  # pragma: no cover - exercised in tests
        print(f"PLOT_CONFUSION_MATRIX_FAILED: {exc}", file=sys.stderr)
        return 1
    print(str(args.out_prefix))
    return 0


def _cmd_plot_regression_residuals(args: argparse.Namespace) -> int:
    try:
        table = _read_required_columns(args.input, [args.y_true_col, args.y_pred_col])
        plot_regression_residuals(
            y_true=table[args.y_true_col],
            y_pred=table[args.y_pred_col],
            title=args.title,
            config=_plot_config_from_args(args),
            out_prefix=args.out_prefix,
        )
    except Exception as exc:  # pragma: no cover - exercised in tests
        print(f"PLOT_REGRESSION_RESIDUALS_FAILED: {exc}", file=sys.stderr)
        return 1
    print(str(args.out_prefix))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        code = exc.code
        if isinstance(code, int):
            return code
        return 1
    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
