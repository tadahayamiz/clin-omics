from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from clin_omics.constants import SCHEMA_VERSION


def default_provenance() -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }



def generate_dataset_id() -> str:
    return f"ds_{uuid4().hex}"
