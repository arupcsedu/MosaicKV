"""MosaicKV research infrastructure.

The package currently provides configuration, provenance, diagnostics, and
adapter scaffolding. It does not yet implement the MosaicKV research method.
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("mosaickv")
except PackageNotFoundError:  # pragma: no cover - source-tree import fallback
    __version__ = "0+unknown"

__all__ = ["__version__"]
