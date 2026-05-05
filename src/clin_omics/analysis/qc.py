from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from clin_omics.dataset import CanonicalDataset


@dataclass(frozen=True)
class ExpressionMatrixQCResult:
    sample_qc: pd.DataFrame
    feature_qc: pd.DataFrame
    summary: dict[str, Any]


def summarize_dataset_qc(dataset: CanonicalDataset) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "n_samples": [dataset.X.shape[0]],
            "n_features": [dataset.X.shape[1]],
            "n_layers": [len(dataset.layers)],
            "n_embeddings": [len(dataset.embeddings)],
            "n_assignments": [len(dataset.assignments)],
        }
    )


def _resolve_matrix(dataset: CanonicalDataset, layer: str | None) -> tuple[str, pd.DataFrame]:
    if layer is None or layer == "X":
        return "X", dataset.X.copy()
    if layer not in dataset.layers:
        raise ValueError(f"Layer '{layer}' not found. Available layers: {sorted(dataset.layers.keys())}")
    return layer, dataset.layers[layer].copy()


def _as_non_negative_numeric(frame: pd.DataFrame, *, label: str) -> pd.DataFrame:
    numeric = frame.astype(float)
    if numeric.isna().any().any():
        raise ValueError(f"{label} contains missing values.")
    if (numeric < 0).any().any():
        raise ValueError(f"{label} must contain non-negative expression values.")
    return numeric


def _compute_cpm(counts: pd.DataFrame, scale_factor: float = 1_000_000.0) -> pd.DataFrame:
    library_sizes = counts.sum(axis=1).replace(0.0, np.nan)
    cpm = counts.div(library_sizes, axis=0) * float(scale_factor)
    return cpm.fillna(0.0)


def _format_threshold(value: float) -> str:
    numeric = float(value)
    if numeric.is_integer():
        return str(int(numeric))
    text = f"{numeric:g}"
    return text.replace("-", "m").replace(".", "p")


def _deduplicate_thresholds(values: Sequence[float]) -> list[float]:
    seen: set[float] = set()
    out: list[float] = []
    for value in values:
        numeric = float(value)
        if numeric < 0:
            raise ValueError("QC thresholds must be non-negative.")
        if numeric not in seen:
            seen.add(numeric)
            out.append(numeric)
    return out


def _sample_correlations(matrix: pd.DataFrame) -> pd.DataFrame:
    if matrix.shape[0] < 2:
        return pd.DataFrame(
            {
                "qc_median_sample_correlation": [np.nan] * matrix.shape[0],
                "qc_mean_sample_correlation": [np.nan] * matrix.shape[0],
            },
            index=matrix.index.copy(),
        )

    corr = matrix.T.corr(method="pearson")
    np.fill_diagonal(corr.values, np.nan)
    return pd.DataFrame(
        {
            "qc_median_sample_correlation": corr.median(axis=1, skipna=True),
            "qc_mean_sample_correlation": corr.mean(axis=1, skipna=True),
        },
        index=matrix.index.copy(),
    )


def _robust_low_flags(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    valid = numeric.dropna()
    if valid.shape[0] < 4:
        return pd.Series(False, index=values.index)
    q1 = float(valid.quantile(0.25))
    q3 = float(valid.quantile(0.75))
    iqr = q3 - q1
    if iqr == 0:
        return pd.Series(False, index=values.index)
    return numeric < (q1 - 1.5 * iqr)


def _robust_high_flags(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    valid = numeric.dropna()
    if valid.shape[0] < 4:
        return pd.Series(False, index=values.index)
    q1 = float(valid.quantile(0.25))
    q3 = float(valid.quantile(0.75))
    iqr = q3 - q1
    if iqr == 0:
        return pd.Series(False, index=values.index)
    return numeric > (q3 + 1.5 * iqr)


def _add_sample_warning_columns(sample_qc: pd.DataFrame, *, top_n_features: int) -> pd.DataFrame:
    reasons = pd.Series([[] for _ in range(sample_qc.shape[0])], index=sample_qc.index, dtype="object")

    checks = {
        "low_total_counts": _robust_low_flags(sample_qc["qc_total_counts"]),
        "low_detected_genes": _robust_low_flags(sample_qc["qc_n_detected_genes_count_gt0"]),
        f"high_top_{top_n_features}_gene_fraction": _robust_high_flags(
            sample_qc[f"qc_top_{top_n_features}_gene_fraction"]
        ),
        "low_sample_correlation": _robust_low_flags(sample_qc["qc_median_sample_correlation"]),
    }
    for label, mask in checks.items():
        for idx in sample_qc.index[mask.fillna(False)]:
            reasons.at[idx].append(label)

    out = sample_qc.copy()
    out["qc_warning_flags"] = reasons.map(len).astype(int)
    out["qc_warning_reasons"] = reasons.map(lambda items: ";".join(items))
    return out


def summarize_expression_matrix_qc(
    dataset: CanonicalDataset,
    *,
    counts_layer: str | None = None,
    cpm_layer: str | None = None,
    count_thresholds: Sequence[float] = (5.0, 10.0),
    cpm_thresholds: Sequence[float] = (1.0,),
    top_n_features: int = 10,
    correlation_layer: str | None = None,
    add_warning_flags: bool = True,
) -> ExpressionMatrixQCResult:
    """Summarize sample-level and feature-level QC from a count-like matrix."""
    if top_n_features < 1:
        raise ValueError("top_n_features must be >= 1.")

    counts_source, counts_frame = _resolve_matrix(dataset, counts_layer)
    counts = _as_non_negative_numeric(counts_frame, label=f"Counts source '{counts_source}'")
    count_thresholds = _deduplicate_thresholds(count_thresholds)
    cpm_thresholds = _deduplicate_thresholds(cpm_thresholds)

    if cpm_layer is None:
        cpm_source = "computed_from_counts"
        cpm = _compute_cpm(counts)
    else:
        cpm_source, cpm_frame = _resolve_matrix(dataset, cpm_layer)
        cpm = _as_non_negative_numeric(cpm_frame, label=f"CPM source '{cpm_source}'")

    top_k = min(int(top_n_features), counts.shape[1])
    totals = counts.sum(axis=1)
    detected_gt0 = (counts > 0).sum(axis=1)
    top_fraction = counts.apply(
        lambda row: float(row.nlargest(top_k).sum() / row.sum()) if row.sum() > 0 else 0.0,
        axis=1,
    )

    sample_qc = pd.DataFrame(
        {
            "sample_id": counts.index.astype(str),
            "qc_total_counts": totals.to_numpy(dtype=float),
            "qc_n_detected_genes_count_gt0": detected_gt0.to_numpy(dtype=int),
            "qc_fraction_zero_genes": (1.0 - detected_gt0 / counts.shape[1]).to_numpy(dtype=float),
            f"qc_top_{top_n_features}_gene_fraction": top_fraction.to_numpy(dtype=float),
        },
        index=counts.index.copy(),
    )
    for threshold in count_thresholds:
        suffix = _format_threshold(threshold)
        sample_qc[f"qc_n_detected_genes_count_ge{suffix}"] = (
            counts >= float(threshold)
        ).sum(axis=1).to_numpy(dtype=int)
    for threshold in cpm_thresholds:
        suffix = _format_threshold(threshold)
        sample_qc[f"qc_n_detected_genes_cpm_gt{suffix}"] = (
            cpm > float(threshold)
        ).sum(axis=1).to_numpy(dtype=int)

    if correlation_layer is None:
        correlation_source = "log1p_counts"
        correlation_matrix = np.log1p(counts)
    else:
        correlation_source, correlation_frame = _resolve_matrix(dataset, correlation_layer)
        correlation_matrix = correlation_frame.astype(float)
    sample_qc = sample_qc.join(_sample_correlations(correlation_matrix))
    if add_warning_flags:
        sample_qc = _add_sample_warning_columns(sample_qc, top_n_features=top_n_features)
    sample_qc = sample_qc.reset_index(drop=True)

    feature_totals = counts.sum(axis=0)
    feature_detected_gt0 = (counts > 0).sum(axis=0)
    feature_qc = pd.DataFrame(
        {
            "feature_id": counts.columns.astype(str),
            "qc_total_counts": feature_totals.to_numpy(dtype=float),
            "qc_mean_counts": counts.mean(axis=0).to_numpy(dtype=float),
            "qc_median_counts": counts.median(axis=0).to_numpy(dtype=float),
            "qc_n_detected_samples_count_gt0": feature_detected_gt0.to_numpy(dtype=int),
            "qc_detection_rate_count_gt0": (feature_detected_gt0 / counts.shape[0]).to_numpy(dtype=float),
        },
        index=counts.columns.copy(),
    )
    for threshold in count_thresholds:
        suffix = _format_threshold(threshold)
        detected = (counts >= float(threshold)).sum(axis=0)
        feature_qc[f"qc_n_detected_samples_count_ge{suffix}"] = detected.to_numpy(dtype=int)
        feature_qc[f"qc_detection_rate_count_ge{suffix}"] = (detected / counts.shape[0]).to_numpy(dtype=float)
    for threshold in cpm_thresholds:
        suffix = _format_threshold(threshold)
        detected = (cpm > float(threshold)).sum(axis=0)
        feature_qc[f"qc_n_detected_samples_cpm_gt{suffix}"] = detected.to_numpy(dtype=int)
        feature_qc[f"qc_detection_rate_cpm_gt{suffix}"] = (detected / counts.shape[0]).to_numpy(dtype=float)
    feature_qc = feature_qc.reset_index(drop=True)

    detected_cpm_key = f"qc_n_detected_genes_cpm_gt{_format_threshold(cpm_thresholds[0])}" if cpm_thresholds else None
    summary = {
        "counts_source": counts_source,
        "cpm_source": cpm_source,
        "correlation_source": correlation_source,
        "n_samples": int(counts.shape[0]),
        "n_features": int(counts.shape[1]),
        "count_thresholds": [float(v) for v in count_thresholds],
        "cpm_thresholds": [float(v) for v in cpm_thresholds],
        "top_n_features": int(top_n_features),
        "median_total_counts": float(sample_qc["qc_total_counts"].median()),
        "min_total_counts": float(sample_qc["qc_total_counts"].min()),
        "max_total_counts": float(sample_qc["qc_total_counts"].max()),
        "median_detected_genes_count_gt0": float(sample_qc["qc_n_detected_genes_count_gt0"].median()),
        "min_detected_genes_count_gt0": int(sample_qc["qc_n_detected_genes_count_gt0"].min()),
        "max_detected_genes_count_gt0": int(sample_qc["qc_n_detected_genes_count_gt0"].max()),
        "median_top_gene_fraction": float(sample_qc[f"qc_top_{top_n_features}_gene_fraction"].median()),
        "median_sample_correlation": float(sample_qc["qc_median_sample_correlation"].median(skipna=True))
        if sample_qc["qc_median_sample_correlation"].notna().any()
        else None,
        "n_samples_with_warning": int(sample_qc.get("qc_warning_flags", pd.Series(dtype=int)).gt(0).sum()),
    }
    if detected_cpm_key is not None:
        summary[f"median_detected_genes_cpm_gt{_format_threshold(cpm_thresholds[0])}"] = float(
            sample_qc[detected_cpm_key].median()
        )

    return ExpressionMatrixQCResult(sample_qc=sample_qc, feature_qc=feature_qc, summary=summary)


def append_expression_qc_to_obs(
    dataset: CanonicalDataset,
    sample_qc: pd.DataFrame,
    *,
    overwrite: bool = False,
) -> CanonicalDataset:
    """Return a dataset with sample-level expression QC columns appended to obs."""
    if "sample_id" not in sample_qc.columns:
        raise ValueError("sample_qc must contain a 'sample_id' column.")

    obs = dataset.obs.copy()
    qc_columns = [col for col in sample_qc.columns if col != "sample_id"]
    existing = [col for col in qc_columns if col in obs.columns]
    if existing and not overwrite:
        raise ValueError(f"obs already contains QC columns: {existing}")

    indexed_qc = sample_qc.set_index("sample_id")
    if indexed_qc.index.duplicated().any():
        dup = indexed_qc.index[indexed_qc.index.duplicated()].astype(str).tolist()[:5]
        raise ValueError(f"sample_qc contains duplicate sample IDs. Example: {dup}")

    missing = [sample for sample in obs["sample_id"].astype(str).tolist() if sample not in indexed_qc.index]
    if missing:
        preview = ", ".join(missing[:5])
        raise ValueError(f"sample_qc is missing obs samples ({len(missing)} missing). Example: {preview}")

    aligned = indexed_qc.loc[obs["sample_id"].astype(str).tolist(), qc_columns].reset_index(drop=True)
    if existing:
        obs = obs.drop(columns=existing)
    obs = pd.concat([obs.reset_index(drop=True), aligned.reset_index(drop=True)], axis=1)

    return CanonicalDataset(
        X=dataset.X.copy(),
        obs=obs,
        var=dataset.var.copy(),
        layers={name: frame.copy() for name, frame in dataset.layers.items()},
        provenance=dict(dataset.provenance),
        dataset_id=dataset.dataset_id,
        embeddings={name: frame.copy() for name, frame in dataset.embeddings.items()},
        feature_scores={name: frame.copy() for name, frame in dataset.feature_scores.items()},
        assignments={name: series.copy() for name, series in dataset.assignments.items()},
    )


def write_expression_qc_tables(
    result: ExpressionMatrixQCResult,
    outdir: str | Path,
    *,
    prefix: str = "expression_matrix_qc",
) -> dict[str, str]:
    output_dir = Path(outdir)
    output_dir.mkdir(parents=True, exist_ok=True)
    sample_path = output_dir / f"{prefix}_sample_qc.csv"
    feature_path = output_dir / f"{prefix}_feature_qc.csv"
    summary_path = output_dir / f"{prefix}_summary.json"
    result.sample_qc.to_csv(sample_path, index=False)
    result.feature_qc.to_csv(feature_path, index=False)
    summary_path.write_text(
        json.dumps(result.summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return {
        "sample_qc": str(sample_path),
        "feature_qc": str(feature_path),
        "summary": str(summary_path),
    }
