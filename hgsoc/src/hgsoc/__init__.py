"""HGSOC drug delivery baseline synthetic package."""

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("hgsoc-drug-delivery-baseline")
except PackageNotFoundError:  # pragma: no cover
    __version__ = "0.1.0"

__all__ = ["__version__"]
