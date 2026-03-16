from pathlib import Path

import pandas as pd
import pytest

from clin_omics.dataset import CanonicalDataset
from clin_omics.exceptions import SchemaValidationError



def make_toy_dataset() -> CanonicalDataset:
    obs = pd.DataFrame(
        {
            "sample_id": ["S1", "S2"],
            "group": ["A", "B"],
        }
    )
    var = pd.DataFrame(
        {
            "feature_id": ["F1", "F2", "F3"],
            "feature_name": ["g1", "g2", "g3"],
        }
    )
    X = pd.DataFrame(
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
        index=["S1", "S2"],
        columns=["F1", "F2", "F3"],
    )
    layers = {
        "normalized": pd.DataFrame(
            [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
            index=["S1", "S2"],
            columns=["F1", "F2", "F3"],
        )
    }
    return CanonicalDataset(X=X, obs=obs, var=var, layers=layers)



def test_canonical_dataset_accepts_valid_toy_data() -> None:
    dataset = make_toy_dataset()
    assert dataset.X.shape == (2, 3)
    assert list(dataset.obs["sample_id"]) == ["S1", "S2"]
    assert "normalized" in dataset.layers



def test_canonical_dataset_rejects_misaligned_index() -> None:
    obs = pd.DataFrame({"sample_id": ["S1", "S2"]})
    var = pd.DataFrame({"feature_id": ["F1", "F2"]})
    X = pd.DataFrame([[1.0, 2.0], [3.0, 4.0]], index=["S2", "S1"], columns=["F1", "F2"])

    with pytest.raises(SchemaValidationError, match="X index"):
        CanonicalDataset(X=X, obs=obs, var=var)



def test_save_and_load_h5_roundtrip(tmp_path: Path) -> None:
    dataset = make_toy_dataset()
    output_path = tmp_path / "toy_dataset.h5"

    dataset.save_h5(output_path)
    loaded = CanonicalDataset.load_h5(output_path)

    pd.testing.assert_frame_equal(loaded.X, dataset.X)
    pd.testing.assert_frame_equal(loaded.obs, dataset.obs)
    pd.testing.assert_frame_equal(loaded.var, dataset.var)
    pd.testing.assert_frame_equal(loaded.layers["normalized"], dataset.layers["normalized"])
    assert loaded.dataset_id == dataset.dataset_id
    assert loaded.provenance["schema_version"] == dataset.provenance["schema_version"]



def test_canonical_dataset_rejects_non_numeric_X() -> None:
    obs = pd.DataFrame({"sample_id": ["S1", "S2"]})
    var = pd.DataFrame({"feature_id": ["F1", "F2"]})
    X = pd.DataFrame([["a", 2.0], ["b", 4.0]], index=["S1", "S2"], columns=["F1", "F2"])

    with pytest.raises(SchemaValidationError, match="X must contain only numeric columns"):
        CanonicalDataset(X=X, obs=obs, var=var)


def test_canonical_dataset_rejects_non_numeric_layer() -> None:
    obs = pd.DataFrame({"sample_id": ["S1", "S2"]})
    var = pd.DataFrame({"feature_id": ["F1", "F2"]})
    X = pd.DataFrame([[1.0, 2.0], [3.0, 4.0]], index=["S1", "S2"], columns=["F1", "F2"])
    layers = {
        "bad": pd.DataFrame([["x", 0.1], ["y", 0.2]], index=["S1", "S2"], columns=["F1", "F2"])
    }

    with pytest.raises(SchemaValidationError, match="Layer 'bad' must contain only numeric columns"):
        CanonicalDataset(X=X, obs=obs, var=var, layers=layers)
