"""Shared polygon geometry; implementation is embedded in the standalone offline auditor."""
try:from .verify_dataset import polygon
except ImportError:from verify_dataset import polygon
