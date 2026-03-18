import clin_omics


def test_import_and_version() -> None:
    assert hasattr(clin_omics, "__version__")
    assert clin_omics.SCHEMA_VERSION == "0.1"
