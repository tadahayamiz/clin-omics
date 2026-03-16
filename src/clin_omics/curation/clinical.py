from __future__ import annotations

from collections.abc import Mapping

import pandas as pd

from clin_omics.constants import REQUIRED_OBS_ID_COLUMN
from clin_omics.exceptions import SchemaValidationError
from clin_omics.schema import validate_obs_table



def curate_clinical_table(
    table: pd.DataFrame,
    *,
    sample_id_col: str = "sample_id",
    rename_map: Mapping[str, str] | None = None,
    category_maps: Mapping[str, Mapping[object, object]] | None = None,
    dtype_map: Mapping[str, str] | None = None,
) -> pd.DataFrame:
    """Normalize a clinical wide table into obs-like metadata."""
    work = table.copy()

    if rename_map:
        work = work.rename(columns=dict(rename_map))
        sample_id_col = rename_map.get(sample_id_col, sample_id_col)

    if sample_id_col not in work.columns:
        raise SchemaValidationError(
            f"Clinical table must contain sample id column '{sample_id_col}'."
        )

    if sample_id_col != REQUIRED_OBS_ID_COLUMN:
        work = work.rename(columns={sample_id_col: REQUIRED_OBS_ID_COLUMN})

    if category_maps:
        for column, mapping in category_maps.items():
            if column not in work.columns:
                raise SchemaValidationError(
                    f"category_maps specified unknown column '{column}'."
                )
            work[column] = work[column].replace(dict(mapping))

    if dtype_map:
        for column, dtype in dtype_map.items():
            if column not in work.columns:
                raise SchemaValidationError(
                    f"dtype_map specified unknown column '{column}'."
                )
            work[column] = work[column].astype(dtype)

    ordered_columns = [REQUIRED_OBS_ID_COLUMN] + [
        col for col in work.columns if col != REQUIRED_OBS_ID_COLUMN
    ]
    work = work.loc[:, ordered_columns]
    return validate_obs_table(work)
