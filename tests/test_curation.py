import pandas as pd
import pytest

from clin_omics.constants import REQUIRED_OBS_ID_COLUMN, REQUIRED_VAR_ID_COLUMN
from clin_omics.curation import (
    build_obs_from_clinical,
    build_var_from_matrix,
    curate_clinical_table,
    curate_omics_long_to_matrix,
    curate_omics_matrix,
)
from clin_omics.exceptions import SchemaValidationError



def test_curate_omics_long_to_matrix_basic() -> None:
    long_df = pd.DataFrame(
        {
            "sample_id": ["s1", "s1", "s2", "s2"],
            "feature_id": ["g1", "g2", "g1", "g2"],
            "value": [1.0, 2.0, 3.0, 4.0],
        }
    )

    X = curate_omics_long_to_matrix(long_df)

    assert X.index.tolist() == ["s1", "s2"]
    assert X.columns.tolist() == ["g1", "g2"]
    assert X.loc["s2", "g2"] == 4.0



def test_curate_omics_long_to_matrix_duplicate_error() -> None:
    long_df = pd.DataFrame(
        {
            "sample_id": ["s1", "s1"],
            "feature_id": ["g1", "g1"],
            "value": [1.0, 2.0],
        }
    )

    with pytest.raises(SchemaValidationError):
        curate_omics_long_to_matrix(long_df)



def test_curate_omics_long_to_matrix_duplicate_mean() -> None:
    long_df = pd.DataFrame(
        {
            "sample_id": ["s1", "s1", "s2"],
            "feature_id": ["g1", "g1", "g1"],
            "value": [1.0, 3.0, 5.0],
        }
    )

    X = curate_omics_long_to_matrix(long_df, duplicate_policy="mean")

    assert X.loc["s1", "g1"] == 2.0
    assert X.loc["s2", "g1"] == 5.0



def test_curate_omics_matrix_from_sample_id_column() -> None:
    matrix_df = pd.DataFrame(
        {
            "sample_id": ["s1", "s2"],
            "g1": [1.0, 3.0],
            "g2": [2.0, 4.0],
        }
    )

    X = curate_omics_matrix(matrix_df, sample_id_col="sample_id")

    assert X.index.tolist() == ["s1", "s2"]
    assert X.columns.tolist() == ["g1", "g2"]



def test_curate_clinical_table_basic() -> None:
    clinical = pd.DataFrame(
        {
            "sid": ["s1", "s2"],
            "sex": ["M", "F"],
            "age": ["40", "52"],
        }
    )

    obs = curate_clinical_table(
        clinical,
        sample_id_col="sid",
        category_maps={"sex": {"M": "male", "F": "female"}},
        dtype_map={"age": "int64"},
    )

    assert obs.columns.tolist() == [REQUIRED_OBS_ID_COLUMN, "sex", "age"]
    assert obs["sex"].tolist() == ["male", "female"]
    assert str(obs["age"].dtype) == "int64"



def test_build_obs_from_clinical_aligns_to_matrix_order() -> None:
    X = pd.DataFrame(
        [[1.0, 2.0], [3.0, 4.0]],
        index=["s2", "s1"],
        columns=["g1", "g2"],
    )
    clinical = pd.DataFrame(
        {
            REQUIRED_OBS_ID_COLUMN: ["s1", "s2"],
            "group": ["case", "control"],
        }
    )

    obs = build_obs_from_clinical(X, clinical)

    assert obs[REQUIRED_OBS_ID_COLUMN].tolist() == ["s2", "s1"]
    assert obs["group"].tolist() == ["control", "case"]



def test_build_obs_from_clinical_missing_sample_raises() -> None:
    X = pd.DataFrame([[1.0]], index=["s2"], columns=["g1"])
    clinical = pd.DataFrame({REQUIRED_OBS_ID_COLUMN: ["s1"], "group": ["case"]})

    with pytest.raises(SchemaValidationError):
        build_obs_from_clinical(X, clinical)



def test_build_var_from_matrix() -> None:
    X = pd.DataFrame([[1.0, 2.0]], index=["s1"], columns=["g1", "g2"])

    var = build_var_from_matrix(X)

    assert var.columns.tolist() == [REQUIRED_VAR_ID_COLUMN]
    assert var[REQUIRED_VAR_ID_COLUMN].tolist() == ["g1", "g2"]
