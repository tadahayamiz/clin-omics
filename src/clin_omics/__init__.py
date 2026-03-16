import importlib.metadata

from .constants import SCHEMA_VERSION

try:
    __version__ = importlib.metadata.version("clin-omics")
except importlib.metadata.PackageNotFoundError:
    __version__ = "0.0.0-dev"

__all__ = ["__version__", "SCHEMA_VERSION"]
