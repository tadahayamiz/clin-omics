class ClinOmicsError(Exception):
    """Base exception for clin_omics."""


class SchemaValidationError(ClinOmicsError):
    """Raised when a schema validation check fails."""
