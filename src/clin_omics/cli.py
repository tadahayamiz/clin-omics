from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from clin_omics import SCHEMA_VERSION, __version__
from clin_omics.analysis import KMeansClustering, KNNLeidenClustering, PCAEmbedding
from clin_omics.dataset import CanonicalDataset
from clin_omics.io import read_dataset_h5, read_table


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
    pca_parser.add_argument("--source-layer", default=None, help="Optional source layer name.")
    pca_parser.add_argument("--key", default="pca", help="Embedding key name.")
    pca_parser.set_defaults(func=_cmd_pca)

    kmeans_parser = subparsers.add_parser("cluster-kmeans", help="Run k-means clustering and save an updated dataset.")
    kmeans_parser.add_argument("--in", dest="input_path", required=True, type=Path, help="Input dataset .h5 path.")
    kmeans_parser.add_argument("--out", required=True, type=Path, help="Output dataset .h5 path.")
    kmeans_parser.add_argument("--n-clusters", type=int, required=True, help="Number of clusters.")
    kmeans_parser.add_argument("--embedding-key", default=None, help="Optional embedding key to cluster.")
    kmeans_parser.add_argument("--source-layer", default=None, help="Optional source layer name.")
    kmeans_parser.add_argument("--key", default="cluster_kmeans", help="Assignment key name.")
    kmeans_parser.add_argument("--random-state", type=int, default=0, help="Random seed.")
    kmeans_parser.set_defaults(func=_cmd_cluster_kmeans)

    knn_leiden_parser = subparsers.add_parser("cluster-knn-leiden", help="Run kNN->Leiden clustering and save an updated dataset.")
    knn_leiden_parser.add_argument("--in", dest="input_path", required=True, type=Path, help="Input dataset .h5 path.")
    knn_leiden_parser.add_argument("--out", required=True, type=Path, help="Output dataset .h5 path.")
    knn_leiden_parser.add_argument("--neighbors", type=int, default=15, help="Number of nearest neighbors.")
    knn_leiden_parser.add_argument("--resolution", type=float, default=1.0, help="Leiden resolution parameter.")
    knn_leiden_parser.add_argument("--embedding-key", default=None, help="Optional embedding key to cluster.")
    knn_leiden_parser.add_argument("--source-layer", default=None, help="Optional source layer name.")
    knn_leiden_parser.add_argument("--key", default="cluster_knn_leiden", help="Assignment key name.")
    knn_leiden_parser.add_argument("--random-state", type=int, default=0, help="Random seed.")
    knn_leiden_parser.set_defaults(func=_cmd_cluster_knn_leiden)

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
        "obs_columns": dataset.obs.columns.tolist(),
        "var_columns": dataset.var.columns.tolist(),
        "layers": sorted(dataset.layers.keys()),
        "embeddings": sorted(dataset.embeddings.keys()),
        "feature_scores": sorted(dataset.feature_scores.keys()),
        "assignments": sorted(dataset.assignments.keys()),
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


def _cmd_build_dataset(args: argparse.Namespace) -> int:
    try:
        X = read_table(args.x)
        obs = read_table(args.obs)
        var = read_table(args.var)

        if args.sample_id_col not in X.columns:
            raise ValueError(f"Missing sample ID column in X: {args.sample_id_col}")
        if args.sample_id_col not in obs.columns:
            raise ValueError(f"Missing sample ID column in obs: {args.sample_id_col}")
        if args.feature_id_col not in var.columns:
            raise ValueError(f"Missing feature ID column in var: {args.feature_id_col}")

        X = X.set_index(args.sample_id_col)

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

def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
