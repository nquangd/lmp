"""Configuration validation utilities."""

from typing import List, Set
import structlog

from ..contracts.errors import ValidationError
from .model import AppConfig

logger = structlog.get_logger()


def validate_config(config: AppConfig) -> None:
    """Validate configuration for common issues and conflicts.
    
    Args:
        config: Configuration to validate
        
    Raises:
        ValidationError: If configuration is invalid
    """
    errors: List[str] = []
    warnings: List[str] = []
    
    # Check stage dependencies
    _validate_stage_dependencies(config, errors)
    
    # Check parameter consistency
    _validate_parameter_consistency(config, warnings)
    
    # Check resource constraints
    _validate_resource_constraints(config, warnings)
    
    # Check solver configuration
    _validate_solver_config(config, warnings)
    
    # Log warnings
    for warning in warnings:
        logger.warning(warning)
    
    # Raise on errors
    if errors:
        raise ValidationError(
            f"Configuration validation failed: {'; '.join(errors)}"
        )


def _validate_stage_dependencies(config: AppConfig, errors: List[str]) -> None:
    """Validate that stage dependencies can be satisfied."""
    stages = set(config.run.stages)
    
    # Check for impossible combinations
    if "pbbm" in stages and "deposition" not in stages:
        if config.pbbm.model != "null":
            errors.append(
                "PBBM stage requires deposition stage or null model"
            )
    
    if "pk" in stages and "pbbm" not in stages:
        if config.pk.model != "null":
            errors.append(
                "PK stage typically requires PBBM stage or null model"
            )


def _validate_parameter_consistency(config: AppConfig, warnings: List[str]) -> None:
    """Check for parameter consistency across stages."""
    
    # Check solver tolerances
    if config.pbbm.solver.rtol > 1e-3:
        warnings.append(
            f"PBBM rtol={config.pbbm.solver.rtol} may be too loose for accurate results"
        )
    
    if config.pbbm.solver.atol > config.pbbm.solver.rtol:
        warnings.append(
            "PBBM atol should typically be smaller than rtol"
        )


def _validate_resource_constraints(config: AppConfig, warnings: List[str]) -> None:
    """Check resource usage settings."""
    
    if config.run.threads > 16:
        warnings.append(
            f"threads={config.run.threads} may cause performance issues"
        )
    
    if config.run.enable_numba and config.run.threads > 1:
        warnings.append(
            "Numba with multiple threads may not provide expected speedup"
        )


def _validate_solver_config(config: AppConfig, warnings: List[str]) -> None:
    """Validate solver configuration."""
    
    solver = config.pbbm.solver
    
    # Method-specific checks
    if solver.method == "BDF" and solver.max_step < 0.1:
        warnings.append(
            "BDF method with very small max_step may be inefficient"
        )
    
    if solver.method in ("RK45", "DOP853") and solver.rtol < 1e-8:
        warnings.append(
            f"{solver.method} with very tight tolerances may be inefficient"
        )