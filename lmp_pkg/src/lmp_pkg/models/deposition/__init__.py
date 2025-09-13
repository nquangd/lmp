"""Deposition models."""

from .base import DepositionModel
from .clean_lung import CleanLungDeposition

__all__ = ["DepositionModel", "CleanLungDeposition"]