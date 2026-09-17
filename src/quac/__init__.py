"""
.. include ../../README.md
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("quac")
except PackageNotFoundError:  # pragma: no cover
    __version__ = "uninstalled"
