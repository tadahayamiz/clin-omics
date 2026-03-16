from __future__ import annotations

import pandas as pd

from clin_omics.constants import REQUIRED_OBS_ID_COLUMN, REQUIRED_VAR_ID_COLUMN
from clin_omics.exceptions import SchemaValidationError
from clin_omics.schema import validate_obs_table, validate_var_table



def build_obs_from_clinical(
    X: pd.DataFrame,
    clinical: pd.DataFrame,
    *,
    require_all_samples: bool = True,
) -> pd.DataFrame:
    """Build obs aligned to an omics matrix from a curated clinical table."""
    validated_obs = validate_obs_table(clinical)

    sample_ids = pd.Index(X.index.astype(str), name=REQUIRED_OBS_ID_COLUMN)
    clinical_ids = pd.Index(validated_obs[REQUIRED_OBS_ID_COLUMN].astype(str))

    missing = sample_ids.difference(clinical_ids)
    if require_all_samples and len(missing) > 0:
        raise SchemaValidationError(
            f"Clinical table is missing metadata for sample ids: {missing.tolist()}"
        )

    filtered = validated_obs[validated_obs[REQUIRED_OBS_ID_COLUMN].isin(sample_ids)].copy()
    filtered = filtered.set_index(REQUIRED_OBS_ID_COLUMN).reindex(sample_ids).reset_index()
    filtered.columns = [REQUIRED_OBS_ID_COLUMN, *filtered.columns[1:]]
    return validate_obs_table(filtered)



def build_var_from_matrix(X: pd.DataFrame) -> pd.DataFrame:
    """Build a minimal var table from a samples x features matrix."""
    var = pd.DataFrame({REQUIRED_VAR_ID_COLUMN: pd.Index(X.columns.astype(str))})
    return validate_var_table(var)
