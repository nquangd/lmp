"""Error definitions for the LMP package."""

from __future__ import annotations
from typing import Dict, Optional


class LMPError(Exception):
    """Base exception for all LMP package errors."""
    
    def __init__(self, message: str, details: Optional[Dict] = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}


class ConfigError(LMPError):
    """Configuration-related errors."""
    pass


class ValidationError(ConfigError):
    """Input validation errors."""
    pass


class ModelError(LMPError):
    """Model execution errors."""
    pass


class SolverError(LMPError):
    """ODE solver errors."""
    pass


class DependencyError(ModelError):
    """Missing dependency or stage coupling errors."""
    pass


class ResourceError(LMPError):
    """Resource constraint errors (memory, disk, compute)."""
    pass