from __future__ import annotations

import numpy as np
import pandas as pd

from clin_omics.dataset import CanonicalDataset
from clin_omics.preprocess import (
    Log1pTransform,
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
