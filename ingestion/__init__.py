"""
Data Ingestion Layer for DataLab

This module provides a unified interface for ingesting data from various sources
including local files, databases, APIs, and user uploads.
"""

"""This package exposes ingestion helpers lazily to avoid heavy imports
at package import time (e.g., pandas/sqlalchemy).

Attributes provided lazily: `ingest`, `get_ingestion_registry`, `Dataset`.
"""

__all__ = ['ingest', 'get_ingestion_registry', 'Dataset']


def __getattr__(name: str):
	"""Lazily import and return attributes from submodules.

	This prevents importing heavy dependencies when only a small part
	of the package is needed.
	"""
	if name == 'ingest' or name == 'get_ingestion_registry':
		from .ingestion_registry import ingest, get_ingestion_registry
		return locals()[name]

	if name == 'Dataset':
		from .dataset import Dataset
		return Dataset

	raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
	return __all__
