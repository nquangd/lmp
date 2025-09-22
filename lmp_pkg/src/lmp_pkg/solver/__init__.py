"""Configurable numerical solvers with study-specific settings."""

from .solver_config import (
    SolverSettings,
    get_solver_settings, 
    create_custom_solver_settings,
    create_study_specific_settings
)

from .study_config import (
    StudyType,
    StudyConfiguration,
    DissolutionSettings,
    LumpingSettings,
    get_study_configuration
)

from .ode_solver import ODESolver, solve_ode_system
from .optimization import optimize_parameters, fit_model_to_data
from .base import SolverBase, ODESolverBase

__all__ = [
    # Main solver settings
    'SolverSettings',
    'get_solver_settings',
    'create_custom_solver_settings', 
    'create_study_specific_settings',
    
    # Study configurations
    'StudyType',
    'StudyConfiguration',
    'DissolutionSettings',
    'LumpingSettings',
    'get_study_configuration',
    
    # ODE solvers
    'ODESolver',
    'solve_ode_system',
    
    # Optimization
    'optimize_parameters',
    'fit_model_to_data',
    
    # Base classes
    'SolverBase',
    'ODESolverBase'
]