"""Configurable solver settings for PBBM simulations.

Allows flexible configuration of numerical solver parameters
based on original rtol/atol settings from lung_pbbm.py.
Integrated with study-specific configurations.
"""

from __future__ import annotations
from typing import Dict, Any, Optional
from dataclasses import dataclass
from pydantic import BaseModel, Field

from .study_config import (
    StudyConfiguration, 
    StudyType, 
    get_study_configuration,
    DissolutionSettings,
    LumpingSettings
)


class SolverSettings(BaseModel):
    """Configurable numerical solver settings with study integration.
    
    Based on original settings from lung_pbbm.py:
    - rtol = 1e-4 (original line 239)  
    - atol = 1e-8 (original line 240)
    """
    
    # Core ODE solver tolerances
    rtol: float = Field(1e-4, gt=0.0, description="Relative tolerance")
    atol: float = Field(1e-8, gt=0.0, description="Absolute tolerance")
    
    # Integration method and control
    method: str = Field("BDF", description="Integration method (BDF, RK45, LSODA, etc.)")
    max_step: Optional[float] = Field(None, gt=0.0, description="Maximum step size [s]")
    first_step: Optional[float] = Field(None, gt=0.0, description="Initial step size [s]") 
    
    # Solver-specific parameters
    max_iter: int = Field(10000, gt=0, description="Maximum iterations")
    
    # Study configuration (includes dissolution, lumping, blocking)
    study_config: StudyConfiguration = Field(default_factory=StudyConfiguration)
    
    def get_scipy_options(self) -> Dict[str, Any]:
        """Get options formatted for scipy.integrate.solve_ivp.
        
        Returns:
            Dictionary with scipy-compatible solver options
        """
        options = {
            'rtol': self.rtol,
            'atol': self.atol,
            'method': self.method
        }
        
        if self.max_step is not None:
            options['max_step'] = self.max_step
            
        if self.first_step is not None:
            options['first_step'] = self.first_step
            
        return options
    
    def get_pbbm_options(self) -> Dict[str, Any]:
        """Get PBBM-specific solver options including study configuration.
        
        Returns:
            Dictionary with PBBM model parameters
        """
        # Get study-specific parameters
        study_params = self.study_config.get_numba_params()
        
        # Combine with solver options
        return {
            **study_params,
            'study_type': self.study_config.study_type.value,
            'study_name': self.study_config.study_name
        }
    
    def get_all_options(self) -> Dict[str, Any]:
        """Get all solver and study options combined.
        
        Returns:
            Dictionary with all configuration parameters
        """
        return {
            'solver': self.get_scipy_options(),
            'pbbm': self.get_pbbm_options(),
            'study': self.study_config.get_study_info()
        }


@dataclass
class ModelSpecificSettings:
    """Model-specific solver settings for different simulation types."""
    
    # Standard PBBM settings (based on original)
    pbbm_standard = SolverSettings(
        rtol=1e-4,
        atol=1e-8,
        method="BDF",
        max_step=3600.0,  # 1 hour max step
        study_config=get_study_configuration("normal")
    )
    
    # High-precision PBBM settings
    pbbm_high_precision = SolverSettings(
        rtol=1e-6,
        atol=1e-10,
        method="BDF",
        max_step=1800.0,  # 30-minute max step
        study_config=get_study_configuration("normal")
    )
    
    # Fast PBBM settings (for parameter sweeps)
    pbbm_fast = SolverSettings(
        rtol=1e-3,
        atol=1e-6,
        method="RK45",
        max_step=7200.0,  # 2 hour max step
        study_config=get_study_configuration("fast_dissolution")
    )
    
    # Charcoal block study settings
    pbbm_charcoal = SolverSettings(
        rtol=1e-4,
        atol=1e-8,
        method="BDF",
        max_step=3600.0,
        study_config=get_study_configuration("charcoal")
    )
    
    # PK-only settings (IV reference)
    pk_only = SolverSettings(
        rtol=1e-5,
        atol=1e-8,
        method="LSODA",
        max_step=3600.0,
        study_config=get_study_configuration("iv_reference")
    )
    
    # GI + PK settings (oral reference)
    gi_pk_only = SolverSettings(
        rtol=1e-5,
        atol=1e-8,
        method="BDF",
        max_step=1800.0,
        study_config=get_study_configuration("oral_reference")
    )
    
    # Lung-only settings (no GI absorption)
    lung_only = SolverSettings(
        rtol=1e-4,
        atol=1e-8,
        method="BDF", 
        max_step=3600.0,
        study_config=get_study_configuration("lung_only")
    )


def get_solver_settings(simulation_type: str = "pbbm_standard") -> SolverSettings:
    """Get solver settings for a specific simulation type.
    
    Args:
        simulation_type: Type of simulation including study-specific types:
                        - "pbbm_standard": Normal inhalation study 
                        - "pbbm_charcoal": Charcoal block study (ET + GI blocked)
                        - "pbbm_fast": Fast dissolution for parameter sweeps
                        - "pbbm_high_precision": High precision settings
                        - "lung_only": Lung absorption only (no GI)
                        - "pk_only": IV reference study 
                        - "gi_pk_only": Oral reference study
                        
    Returns:
        SolverSettings instance configured for the simulation type
    """
    settings_map = {
        "pbbm_standard": ModelSpecificSettings.pbbm_standard,
        "pbbm_charcoal": ModelSpecificSettings.pbbm_charcoal,
        "pbbm_fast": ModelSpecificSettings.pbbm_fast,
        "pbbm_high_precision": ModelSpecificSettings.pbbm_high_precision,
        "lung_only": ModelSpecificSettings.lung_only,
        "pk_only": ModelSpecificSettings.pk_only,
        "gi_pk_only": ModelSpecificSettings.gi_pk_only
    }
    
    if simulation_type not in settings_map:
        available_types = list(settings_map.keys())
        raise ValueError(f"Unknown simulation type '{simulation_type}'. Available types: {available_types}")
        
    return settings_map[simulation_type]


def create_custom_solver_settings(**overrides) -> SolverSettings:
    """Create custom solver settings with overrides.
    
    Args:
        **overrides: Override values for any SolverSettings field
        
    Returns:
        SolverSettings instance with custom values
    """
    # Start with standard settings
    base_settings = get_solver_settings("pbbm_standard")
    
    # Apply overrides
    settings_dict = base_settings.model_dump()
    settings_dict.update(overrides)
    
    return SolverSettings(**settings_dict)


def create_study_specific_settings(
    study_type: str, 
    solver_type: str = "pbbm_standard",
    **study_overrides
) -> SolverSettings:
    """Create solver settings for a specific study with custom parameters.
    
    Args:
        study_type: Type of study ("normal", "charcoal", "iv_reference", etc.)
        solver_type: Base solver configuration ("pbbm_standard", "pbbm_fast", etc.)
        **study_overrides: Override study configuration parameters
        
    Returns:
        SolverSettings with custom study configuration
        
    Examples:
        # Charcoal study with custom dissolution rate
        settings = create_study_specific_settings(
            "charcoal",
            dissolution=DissolutionSettings(dissolution_rate_multiplier=0.1)
        )
        
        # Fast solver with no lumping
        settings = create_study_specific_settings(
            "normal",
            solver_type="pbbm_fast", 
            lumping=LumpingSettings(enable_lumping=False)
        )
    """
    # Get base solver settings
    base_settings = get_solver_settings(solver_type)
    
    # Create custom study configuration
    study_config = get_study_configuration(study_type, **study_overrides)
    
    # Combine
    return SolverSettings(
        rtol=base_settings.rtol,
        atol=base_settings.atol,
        method=base_settings.method,
        max_step=base_settings.max_step,
        first_step=base_settings.first_step,
        max_iter=base_settings.max_iter,
        study_config=study_config
    )