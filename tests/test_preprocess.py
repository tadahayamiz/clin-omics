from __future__ import annotations

import numpy as np
import pandas as pd

from clin_omics.dataset import CanonicalDataset
from clin_omics.preprocess import (
    BulkRNASeqPreprocessor,
    CPMNormalizer,
    FilterLowExpression,
    Log1pTransform,
    LogCPMTransform,
    PreprocessPipeline,
    VarianceFilter,
    ZScoreScaler,
)


def make_dataset() -> CanonicalDataset:
    X = pd.DataFrame(
        [[0.0, 1.0, 5.0], [3.0, 1.0, 7.0], [8.0, 1.0, 9.0]],
        index=["s1", "s2", "s3"],
        columns=["f1", "f2", "f3"],
    )
    obs = pd.DataFrame({"sample_id": ["s1", "s2", "s3"], "group": ["A", "A", "B"]})
    var = pd.DataFrame({"feature_id": ["f1", "f2", "f3"]})
    return CanonicalDataset(X=X, obs=obs, var=var)


def make_dataset_with_derived() -> CanonicalDataset:
    dataset = make_dataset()
    dataset.layers["baseline"] = dataset.X.copy() * 10.0
    dataset.embeddings["pca"] = pd.DataFrame(
        [[0.1, 1.0], [0.2, 2.0], [0.3, 3.0]],
        index=dataset.obs["sample_id"].tolist(),
        columns=["PC1", "PC2"],
    )
    dataset.feature_scores["importance"] = pd.DataFrame(
        {"score": [0.5, 0.2, 0.9]},
        index=dataset.var["feature_id"].tolist(),
    )
    dataset.assignments["clusters"] = pd.Series(
        [0, 0, 1],
        index=dataset.obs["sample_id"].tolist(),
        name="clusters",
    )
    return dataset


def test_log1p_transform_to_new_layer() -> None:
    dataset = make_dataset()
    transformed = Log1pTransform(target="log1p").fit_transform(dataset)

    assert "log1p" in transformed.layers
    assert transformed.X.equals(dataset.X)
    np.testing.assert_allclose(
        transformed.layers["log1p"].to_numpy(),
        np.log1p(dataset.X.to_numpy()),
    )
    assert transformed.provenance["transform_history"][0]["transform"] == "Log1pTransform"


def test_log1p_transform_rejects_negative_values() -> None:
    dataset = make_dataset()
    dataset.X.loc["s1", "f1"] = -1.0

    try:
        Log1pTransform().fit_transform(dataset)
    except ValueError as exc:
        assert "non-negative" in str(exc)
    else:
        raise AssertionError("Expected ValueError for negative input.")


def test_zscore_scaler_replaces_X() -> None:
    dataset = make_dataset()
    scaled = ZScoreScaler().fit_transform(dataset)

    np.testing.assert_allclose(scaled.X.mean(axis=0).to_numpy(), np.zeros(3), atol=1e-8)
    np.testing.assert_allclose(scaled.X.std(axis=0, ddof=0).to_numpy(), np.array([1.0, 0.0, 1.0]), atol=1e-8)
    assert scaled.X.columns.tolist() == ["f1", "f2", "f3"]


def test_variance_filter_drops_low_variance_features_and_updates_var() -> None:
    dataset = make_dataset()
    filtered = VarianceFilter(threshold=0.0).fit_transform(dataset)

    assert filtered.X.columns.tolist() == ["f1", "f3"]
    assert filtered.var["feature_id"].tolist() == ["f1", "f3"]
    assert filtered.layers == {}


def test_pipeline_applies_steps_in_order() -> None:
    dataset = make_dataset()
    pipeline = PreprocessPipeline(
        steps=[
            ("log1p", Log1pTransform()),
            ("zscore", ZScoreScaler()),
        ]
    )
    transformed = pipeline.fit_transform(dataset)

    expected = np.log1p(dataset.X.to_numpy())
    expected = (expected - expected.mean(axis=0)) / np.where(expected.std(axis=0) == 0, 1.0, expected.std(axis=0))
    np.testing.assert_allclose(transformed.X.to_numpy(), expected)
    assert [entry["transform"] for entry in transformed.provenance["transform_history"]] == [
        "Log1pTransform",
        "ZScoreScaler",
    ]


def test_filter_low_expression_drops_low_count_features_and_updates_var() -> None:
    dataset = make_dataset()
    dataset.X = pd.DataFrame(
        [[0.0, 10.0, 5.0], [1.0, 12.0, 0.0], [0.0, 8.0, 11.0]],
        index=["s1", "s2", "s3"],
        columns=["f1", "f2", "f3"],
    )

    filtered = FilterLowExpression(min_count=10.0, min_samples=2).fit_transform(dataset)

    assert filtered.X.columns.tolist() == ["f2"]
    assert filtered.var["feature_id"].tolist() == ["f2"]
    assert filtered.layers == {}


def test_cpm_normalizer_to_new_layer() -> None:
    dataset = make_dataset()

    transformed = CPMNormalizer(target="cpm").fit_transform(dataset)

    expected = dataset.X.div(dataset.X.sum(axis=1), axis=0) * 1_000_000.0
    assert transformed.X.equals(dataset.X)
    np.testing.assert_allclose(transformed.layers["cpm"].to_numpy(), expected.to_numpy())


def test_logcpm_transform_replaces_x() -> None:
    dataset = make_dataset()

    transformed = LogCPMTransform(prior_count=1.0).fit_transform(dataset)

    expected_cpm = dataset.X.div(dataset.X.sum(axis=1), axis=0) * 1_000_000.0
    expected = np.log2(expected_cpm.to_numpy() + 1.0)
    np.testing.assert_allclose(transformed.X.to_numpy(), expected)


def test_logcpm_transform_rejects_negative_values() -> None:
    dataset = make_dataset()
    dataset.X.loc["s1", "f1"] = -1.0

    try:
        LogCPMTransform().fit_transform(dataset)
    except ValueError as exc:
        assert "non-negative" in str(exc)
    else:
        raise AssertionError("Expected ValueError for negative input.")


def test_bulk_rnaseq_preprocessor_creates_expected_layers() -> None:
    dataset = make_dataset()
    dataset.X = pd.DataFrame(
        [[0.0, 10.0, 5.0], [1.0, 12.0, 0.0], [0.0, 8.0, 11.0]],
        index=["s1", "s2", "s3"],
        columns=["f1", "f2", "f3"],
    )

    transformed = BulkRNASeqPreprocessor(min_count=10.0, min_samples=2).fit_transform(dataset)

    assert transformed.X.columns.tolist() == ["f2"]
    assert transformed.var["feature_id"].tolist() == ["f2"]
    assert set(transformed.layers) == {"counts_raw", "counts_filtered", "cpm", "log_cpm"}
    np.testing.assert_allclose(transformed.layers["counts_raw"].to_numpy(), transformed.X.to_numpy())
    np.testing.assert_allclose(transformed.layers["counts_filtered"].to_numpy(), transformed.X.to_numpy())
    assert np.isfinite(transformed.layers["log_cpm"].to_numpy()).all()


def test_bulk_rnaseq_preprocessor_can_add_zscore_layer() -> None:
    dataset = make_dataset()

    transformed = BulkRNASeqPreprocessor(make_zscore=True).fit_transform(dataset)

    assert "zscore_log_cpm" in transformed.layers
    np.testing.assert_allclose(
        transformed.layers["zscore_log_cpm"].mean(axis=0).to_numpy(),
        np.zeros(transformed.layers["zscore_log_cpm"].shape[1]),
        atol=1e-8,
    )


def test_base_preprocessor_preserves_derived_artifacts() -> None:
    dataset = make_dataset_with_derived()

    transformed = Log1pTransform(target="log1p").fit_transform(dataset)

    assert transformed.embeddings["pca"].equals(dataset.embeddings["pca"])
    assert transformed.feature_scores["importance"].equals(dataset.feature_scores["importance"])
    assert transformed.assignments["clusters"].equals(dataset.assignments["clusters"])


def test_filter_low_expression_subsets_feature_artifacts_and_preserves_sample_artifacts() -> None:
    dataset = make_dataset_with_derived()
    dataset.X = pd.DataFrame(
        [[0.0, 10.0, 5.0], [1.0, 12.0, 0.0], [0.0, 8.0, 11.0]],
        index=["s1", "s2", "s3"],
        columns=["f1", "f2", "f3"],
    )
    dataset.layers["baseline"] = dataset.X.copy() * 10.0

    filtered = FilterLowExpression(min_count=10.0, min_samples=2).fit_transform(dataset)

    assert filtered.X.columns.tolist() == ["f2"]
    assert filtered.layers["baseline"].columns.tolist() == ["f2"]
    assert filtered.feature_scores["importance"].index.tolist() == ["f2"]
    assert filtered.embeddings["pca"].equals(dataset.embeddings["pca"])
    assert filtered.assignments["clusters"].equals(dataset.assignments["clusters"])


def test_preprocess_pipeline_preserves_derived_artifacts_with_bulk_steps() -> None:
    dataset = make_dataset_with_derived()
    dataset.X = pd.DataFrame(
        [[0.0, 10.0, 5.0], [1.0, 12.0, 0.0], [0.0, 8.0, 11.0]],
        index=["s1", "s2", "s3"],
        columns=["f1", "f2", "f3"],
    )
    dataset.layers["baseline"] = dataset.X.copy() * 10.0

    pipeline = PreprocessPipeline(
        steps=[
            ("filter", FilterLowExpression(min_count=10.0, min_samples=2)),
            ("logcpm", LogCPMTransform(target="log_cpm")),
        ]
    )

    transformed = pipeline.fit_transform(dataset)

    assert transformed.X.columns.tolist() == ["f2"]
    assert "log_cpm" in transformed.layers
    assert transformed.embeddings["pca"].equals(dataset.embeddings["pca"])
    assert transformed.assignments["clusters"].equals(dataset.assignments["clusters"])
    assert transformed.feature_scores["importance"].index.tolist() == ["f2"]


def test_bulk_rnaseq_preprocessor_zscore_matches_log_cpm_scaling() -> None:
    dataset = make_dataset()

    transformed = BulkRNASeqPreprocessor(make_zscore=True).fit_transform(dataset)

    log_cpm = transformed.layers["log_cpm"]
    expected = (log_cpm - log_cpm.mean(axis=0)) / log_cpm.std(axis=0, ddof=0).replace(0.0, 1.0)
    np.testing.assert_allclose(
        transformed.layers["zscore_log_cpm"].to_numpy(),
        expected.to_numpy(),
    )
    assert transformed.X.equals(transformed.layers["counts_filtered"])
