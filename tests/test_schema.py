import pandas as pd
import pytest

from clin_omics.exceptions import SchemaValidationError
from clin_omics.schema import (
    validate_feature_ids,
    validate_layer_shapes,
    validate_obs_table,
    validate_sample_ids,
    validate_var_table,
)



def test_validate_sample_ids_accepts_unique_non_missing_ids() -> None:
    result = validate_sample_ids(pd.Series(["S1", "S2", "S3"]))
    assert list(result) == ["S1", "S2", "S3"]



def test_validate_sample_ids_rejects_duplicates() -> None:
    with pytest.raises(SchemaValidationError, match="duplicates"):
        validate_sample_ids(pd.Series(["S1", "S1", "S2"]))



def test_validate_feature_ids_rejects_missing() -> None:
    with pytest.raises(SchemaValidationError, match="missing values"):
        validate_feature_ids(pd.Series(["F1", None, "F3"]))



def test_validate_obs_table_requires_sample_id() -> None:
    obs = pd.DataFrame({"id": ["S1", "S2"]})
    with pytest.raises(SchemaValidationError, match="sample_id"):
        validate_obs_table(obs)



def test_validate_var_table_requires_feature_id() -> None:
    var = pd.DataFrame({"id": ["F1", "F2"]})
    with pytest.raises(SchemaValidationError, match="feature_id"):
        validate_var_table(var)



def test_validate_layer_shapes_accepts_matching_shapes() -> None:
    layers = {
        "normalized": pd.DataFrame([[1.0, 2.0], [3.0, 4.0]]),
        "scaled": pd.DataFrame([[0.1, 0.2], [0.3, 0.4]]),
    }
    specs = validate_layer_shapes(layers, n_obs=2, n_var=2)
    assert [spec.name for spec in specs] == ["normalized", "scaled"]



def test_validate_layer_shapes_rejects_mismatched_shapes() -> None:
    layers = {"normalized": pd.DataFrame([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])}
    with pytest.raises(SchemaValidationError, match="expected"):
        validate_layer_shapes(layers, n_obs=2, n_var=2)
