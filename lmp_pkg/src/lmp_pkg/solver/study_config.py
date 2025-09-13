"""Study-specific configurations for PBBM simulations.

Handles different study types including charcoal block studies,
dissolution parameters, and lumping constants based on original code.
"""

from __future__ import annotations
from typing import Dict, Any, Optional, Literal
from dataclasses import dataclass
from pydantic import BaseModel, Field
from enum import Enum


class StudyType(str, Enum):
    """Study types for PBBM simulations."""
    NORMAL = "normal"              # Standard inhalation study
    CHARCOAL = "charcoal"         # Charcoal block study (blocks ET and GI)
    IV_ONLY = "iv_only"           # IV administration only
    ORAL_ONLY = "oral_only"       # Oral administration only
    LUNG_ONLY = "lung_only"       # Lung absorption only (no GI)
    

class DissolutionSettings(BaseModel):
    """Dissolution model parameters.
    
    Based on original lung_pbbm.py dissolution logic (lines 98-119).
    """
    
    # Dissolution cutoff parameters
    dissolution_cutoff_radius: float = Field(
        1e-6, 
        gt=0.0, 
        description="Particle radius cutoff for dissolution [cm] - original default 1E-6"
    )
    
    # Alternative dissolution cutoff methods
    use_normalized_radius: bool = Field(
        False, 
        description="Use normalized radius instead of absolute cutoff (original line 104)"
    )
    
    normalized_radius_cutoff: float = Field(
        0.1,
        gt=0.0,
        le=1.0, 
        description="Normalized radius cutoff (Rsolid/initial_radius <= cutoff)"
    )
    
    # Dissolution rate enhancement factors
    dissolution_rate_multiplier: float = Field(
        1.0,
        gt=0.0,
        description="Multiplier for dissolution rate (for sensitivity analysis)"
    )
    
    # Initial particle size distribution (from original line 102)
    initial_particle_radii_cm: list[float] = Field(
        default=[
            0.1643e-4/2, 0.2063e-4/2, 0.2591e-4/2, 0.3255e-4/2, 0.4088e-4/2,
            0.5135e-4/2, 0.6449e-4/2, 0.8100e-4/2, 1.0173e-4/2, 1.2777e-4/2,
            1.6048e-4/2, 2.0156e-4/2, 2.5315e-4/2, 3.1795e-4/2, 3.9933e-4/2,
            5.0155e-4/2, 6.2993e-4/2, 7.9116e-4/2, 9.9368e-4/2
        ],
        description="Initial particle radii distribution [cm] - from original binsize_pbpk"
    )


class LumpingSettings(BaseModel):
    """Lumping model parameters for small particles.
    
    Based on original lung_pbbm.py lumping logic (lines 111-116).
    """
    
    # Lumping rate constant
    k_lump: float = Field(
        5e-4 * 1e6,  # Original: 5E-4 * 1e6 pmol
        ge=0.0,
        description="Lumping rate constant [pmol] - original default 5E-4 * 1e6"
    )
    
    # Lumping control
    enable_lumping: bool = Field(
        True,
        description="Enable/disable lumping for small particles"
    )
    
    # Lumping rate enhancement
    lumping_rate_multiplier: float = Field(
        1.0,
        gt=0.0,
        description="Multiplier for lumping rate (for sensitivity analysis)"
    )
    
    # Alternative lumping models
    lumping_model: Literal["mass_based", "number_based", "disabled"] = Field(
        "mass_based",
        description="Type of lumping model to use"
    )


class StudyConfiguration(BaseModel):
    """Complete study configuration including type and model parameters."""
    
    # Study identification
    study_type: StudyType = Field(StudyType.NORMAL, description="Type of study")
    study_name: Optional[str] = Field(None, description="Optional study identifier")
    
    # Absorption blocking (based on original charcoal study logic)
    block_ET_region: bool = Field(False, description="Block ET region absorption")
    block_GI_absorption: bool = Field(False, description="Block GI tract absorption") 
    block_lung_absorption: bool = Field(False, description="Block all lung absorption")
    
    # Model parameters
    dissolution: DissolutionSettings = Field(default_factory=DissolutionSettings)
    lumping: LumpingSettings = Field(default_factory=LumpingSettings)
    
    # Administration routes
    enable_inhalation: bool = Field(True, description="Enable inhalation dosing")
    enable_oral: bool = Field(True, description="Enable oral dosing (via GI)")
    enable_iv: bool = Field(False, description="Enable IV dosing")
    
    @classmethod
    def for_study_type(cls, study_type: StudyType, **overrides) -> 'StudyConfiguration':
        """Create configuration for specific study type.
        
        Args:
            study_type: Type of study to configure
            **overrides: Override any default parameters
            
        Returns:
            StudyConfiguration with appropriate settings
        """
        # Base configuration
        config_dict = {
            'study_type': study_type,
            'dissolution': DissolutionSettings(),
            'lumping': LumpingSettings()
        }
        
        # Study-specific configurations
        if study_type == StudyType.CHARCOAL:
            # Charcoal study blocks ET and GI (original line 291: block_ET_GI = True)
            config_dict.update({
                'block_ET_region': True,
                'block_GI_absorption': True,
                'enable_inhalation': True,
                'enable_oral': False,
                'enable_iv': False,
                'study_name': 'Charcoal Block Study'
            })
            
        elif study_type == StudyType.IV_ONLY:
            config_dict.update({
                'block_ET_region': True,
                'block_GI_absorption': True,
                'enable_inhalation': False,
                'enable_oral': False,
                'enable_iv': True,
                'study_name': 'IV Only Study'
            })
            
        elif study_type == StudyType.ORAL_ONLY:
            config_dict.update({
                'block_lung_absorption': True,
                'enable_inhalation': False,
                'enable_oral': True,
                'enable_iv': False,
                'study_name': 'Oral Only Study'
            })
            
        elif study_type == StudyType.LUNG_ONLY:
            config_dict.update({
                'block_GI_absorption': True,
                'enable_inhalation': True,
                'enable_oral': False,
                'enable_iv': False,
                'study_name': 'Lung Absorption Only Study'
            })
            
        elif study_type == StudyType.NORMAL:
            config_dict.update({
                'enable_inhalation': True,
                'enable_oral': True,
                'enable_iv': False,
                'study_name': 'Normal Inhalation Study'
            })
        
        # Apply any overrides
        config_dict.update(overrides)
        
        return cls(**config_dict)
    
    @property
    def block_ET_GI(self) -> bool:
        """Combined ET and GI blocking flag for compatibility.
        
        Returns:
            True if both ET and GI are blocked (original block_ET_GI logic)
        """
        return self.block_ET_region and self.block_GI_absorption
    
    def get_numba_params(self) -> Dict[str, Any]:
        """Get parameters formatted for numba ODE functions.
        
        Returns:
            Dictionary with dissolution and lumping parameters for numba
        """
        return {
            'dissolution_cutoff_radius': self.dissolution.dissolution_cutoff_radius,
            'k_lump': self.lumping.k_lump * self.lumping.lumping_rate_multiplier,
            'block_ET_GI': self.block_ET_GI,
            'enable_lumping': self.lumping.enable_lumping,
            'dissolution_rate_multiplier': self.dissolution.dissolution_rate_multiplier
        }
    
    def get_study_info(self) -> Dict[str, Any]:
        """Get comprehensive study configuration information.
        
        Returns:
            Dictionary with all study settings
        """
        return {
            'study_type': self.study_type.value,
            'study_name': self.study_name,
            'blocking': {
                'ET_region': self.block_ET_region,
                'GI_absorption': self.block_GI_absorption,
                'lung_absorption': self.block_lung_absorption,
                'combined_ET_GI': self.block_ET_GI
            },
            'dosing_routes': {
                'inhalation': self.enable_inhalation,
                'oral': self.enable_oral,
                'iv': self.enable_iv
            },
            'dissolution': {
                'cutoff_radius_cm': self.dissolution.dissolution_cutoff_radius,
                'rate_multiplier': self.dissolution.dissolution_rate_multiplier,
                'use_normalized_cutoff': self.dissolution.use_normalized_radius
            },
            'lumping': {
                'k_lump': self.lumping.k_lump,
                'enabled': self.lumping.enable_lumping,
                'rate_multiplier': self.lumping.lumping_rate_multiplier,
                'model_type': self.lumping.lumping_model
            }
        }


# Predefined study configurations
@dataclass
class PredefinedStudies:
    """Predefined study configurations for common scenarios."""
    
    # Standard studies
    normal = StudyConfiguration.for_study_type(StudyType.NORMAL)
    
    # Charcoal block study (blocks ET and GI absorption)
    charcoal = StudyConfiguration.for_study_type(
        StudyType.CHARCOAL,
        study_name="Activated Charcoal Block Study"
    )
    
    # Lung-only absorption (no GI)
    lung_only = StudyConfiguration.for_study_type(
        StudyType.LUNG_ONLY,
        study_name="Lung-Only Absorption Study"
    )
    
    # IV reference study  
    iv_reference = StudyConfiguration.for_study_type(
        StudyType.IV_ONLY,
        study_name="Intravenous Reference Study"
    )
    
    # Oral reference study
    oral_reference = StudyConfiguration.for_study_type(
        StudyType.ORAL_ONLY,
        study_name="Oral Reference Study"  
    )
    
    # High dissolution rate study
    fast_dissolution = StudyConfiguration.for_study_type(
        StudyType.NORMAL,
        study_name="Fast Dissolution Study",
        dissolution=DissolutionSettings(
            dissolution_rate_multiplier=10.0,
            dissolution_cutoff_radius=1e-5  # Larger cutoff = more dissolution
        )
    )
    
    # No lumping study  
    no_lumping = StudyConfiguration.for_study_type(
        StudyType.NORMAL,
        study_name="No Lumping Study",
        lumping=LumpingSettings(
            enable_lumping=False,
            k_lump=0.0
        )
    )


def get_study_configuration(study_type: str, **overrides) -> StudyConfiguration:
    """Get study configuration by name or type.
    
    Args:
        study_type: Study type name or predefined study name
        **overrides: Override any configuration parameters
        
    Returns:
        StudyConfiguration for the specified study
        
    Raises:
        ValueError: If study_type is not recognized
    """
    
    # Check predefined studies first
    predefined = {
        'normal': PredefinedStudies.normal,
        'charcoal': PredefinedStudies.charcoal,
        'lung_only': PredefinedStudies.lung_only,
        'iv_reference': PredefinedStudies.iv_reference,
        'oral_reference': PredefinedStudies.oral_reference,
        'fast_dissolution': PredefinedStudies.fast_dissolution,
        'no_lumping': PredefinedStudies.no_lumping
    }
    
    if study_type.lower() in predefined:
        base_config = predefined[study_type.lower()]
        if overrides:
            # Create new config with overrides
            config_dict = base_config.model_dump()
            config_dict.update(overrides)
            return StudyConfiguration(**config_dict)
        return base_config
    
    # Try to create from StudyType enum
    try:
        study_enum = StudyType(study_type.lower())
        return StudyConfiguration.for_study_type(study_enum, **overrides)
    except ValueError:
        available_types = list(StudyType) + list(predefined.keys())
        raise ValueError(f"Unknown study type '{study_type}'. Available types: {available_types}")