"""PBPK utility functions for parameter transformations and composition.

Consolidates the essential functionality from transforms/ and composition/
folders into a single utility module.
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
from abc import ABC, abstractmethod


# ============================================================================
# PARAMETER TRANSFORMATION UTILITIES
# ============================================================================

def calculate_regional_permeabilities(api, regions: List[str]) -> Dict[str, Dict[str, float]]:
    """Calculate regional permeabilities with scaling factors."""
    regional_perms = {}
    
    for region in regions:
        # Get base permeabilities from API
        peff_in = api.peff.get('In', 0.0) if hasattr(api, 'peff') and api.peff else 0.0
        peff_out = api.peff.get('Out', 0.0) if hasattr(api, 'peff') and api.peff else 0.0
        peff_para = api.peff_para if hasattr(api, 'peff_para') and api.peff_para else 0.0
        
        # Get scaling factors for this region
        region_pscale = api.pscale.get(region, {}) if hasattr(api, 'pscale') and api.pscale else {}
        pscale_in = region_pscale.get('In', 1.0)
        pscale_out = region_pscale.get('Out', 1.0)
        
        pscale_para = api.pscale_para.get(region, 0.0) if hasattr(api, 'pscale_para') and api.pscale_para else 0.0
        
        # Apply scaling
        regional_perms[region] = {
            'pg_in': peff_in * pscale_in,
            'pg_out': peff_out * pscale_out,  
            'pg_para': peff_para * pscale_para
        }
    
    return regional_perms


def calculate_pk_rate_constants(api) -> Dict[str, float]:
    """Convert PK parameters to rate constants for ODE solver."""
    return {
        'k12_s': api.k12_h / 3600.0 if hasattr(api, 'k12_h') and api.k12_h else 0.0,
        'k21_s': api.k21_h / 3600.0 if hasattr(api, 'k21_h') and api.k21_h else 0.0,
        'k13_s': api.k13_h / 3600.0 if hasattr(api, 'k13_h') and api.k13_h else 0.0,
        'k31_s': api.k31_h / 3600.0 if hasattr(api, 'k31_h') and api.k31_h else 0.0,
        'k10_s': ((api.clearance_L_h / 3600.0) / api.volume_central_L 
                 if hasattr(api, 'clearance_L_h') and hasattr(api, 'volume_central_L') 
                 and api.clearance_L_h and api.volume_central_L else 0.0)
    }


def calculate_physico_chemical_params(api) -> Dict[str, float]:
    """Calculate physico-chemical parameters for dissolution model."""
    # Density conversion: g/m^3 to g/cm^3
    density_g_cm3 = (api.density_g_m3 / 1e6 
                     if hasattr(api, 'density_g_m3') and api.density_g_m3 
                     else 1.2)
    
    # Molar volume: MM/density [cm^3/mol] -> [cm^3/pmol]
    vm_cm3_pmol = ((api.molecular_weight / density_g_cm3) / 1e12 
                   if hasattr(api, 'molecular_weight') and api.molecular_weight 
                   else 0.0)
    
    # Solubility: [pg/mL] -> [pmol/mL]: pg/mL ÷ (μg/μmol) ÷ (1e6 pg/μg) = pmol/mL
    sg_pmol_ml = (api.solubility_pg_ml / api.molecular_weight / 1e6 
                  if hasattr(api, 'solubility_pg_ml') and hasattr(api, 'molecular_weight')
                  and api.solubility_pg_ml and api.molecular_weight 
                  else 0.0)
    
    return {
        'vm_cm3_pmol': vm_cm3_pmol,
        'sg_pmol_ml': sg_pmol_ml,
        'diffusion_coeff': (api.diffusion_coeff 
                           if hasattr(api, 'diffusion_coeff') and api.diffusion_coeff 
                           else 0.0),
        'molecular_weight': (api.molecular_weight 
                            if hasattr(api, 'molecular_weight') and api.molecular_weight 
                            else 250.0)
    }


def calculate_frc_scaling_factors(subject, regions: List[str]) -> Dict[str, float]:
    """Calculate FRC-based scaling factors for lung geometry."""
    # Calculate FRC scaling factor
    frc_factor = 1.0
    if (hasattr(subject, 'frc_L') and hasattr(subject, 'frc_ref_L') and 
        subject.frc_ref_L and subject.frc_ref_L > 0):
        frc_factor = (subject.frc_L / subject.frc_ref_L) ** (1/3)
    
    # Apply region-specific scaling (ET is not scaled by FRC)
    scaling_factors = {}
    for region in regions:
        if region == 'ET':
            scaling_factors[region] = 1.0
        else:
            scaling_factors[region] = frc_factor
            
    return scaling_factors


# ============================================================================
# SIMPLE MODEL COMPOSITION UTILITIES
# ============================================================================

class SimpleModelComposer:
    """Simple model composer without complex dependency resolution."""
    
    def __init__(self):
        self.models = []
        self.state_mapping = {}
        
    def add_model(self, model, name: str):
        """Add a model with a given name."""
        self.models.append((name, model))
        
    def build_state_mapping(self):
        """Build simple state index mapping."""
        current_idx = 0
        for name, model in self.models:
            state_size = model.get_state_size() if hasattr(model, 'get_state_size') else model.n_states
            self.state_mapping[name] = slice(current_idx, current_idx + state_size)
            current_idx += state_size
        return current_idx
    
    def create_ode_system(self):
        """Create ODE system function."""
        total_size = self.build_state_mapping()
        
        def ode_system(t, y, **kwargs):
            derivatives = np.zeros(total_size)
            model_outputs = {}
            
            # Extract states and compute outputs
            for name, model in self.models:
                state_slice = self.state_mapping[name]
                model_state = y[state_slice]
                
                if hasattr(model, 'extract_outputs'):
                    outputs = model.extract_outputs(t, model_state)
                    model_outputs[name] = outputs
            
            # Compute derivatives
            for name, model in self.models:
                state_slice = self.state_mapping[name]
                model_state = y[state_slice]
                
                external_inputs = kwargs.get('external_inputs', {})
                
                if hasattr(model, 'compute_derivatives'):
                    derivs = model.compute_derivatives(t, model_state, model_outputs, **external_inputs)
                    derivatives[state_slice] = derivs
                    
            return derivatives
        
        return ode_system


# ============================================================================
# FLUX COMMUNICATION DATA STRUCTURE
# ============================================================================

class FluxVector:
    """Simple flux vector for inter-model communication."""
    
    def __init__(self, **kwargs):
        self.mcc_to_gi = kwargs.get('mcc_to_gi', 0.0)
        self.mcc_bb_to_BB = kwargs.get('mcc_bb_to_BB', 0.0)
        self.lung_to_systemic = kwargs.get('lung_to_systemic', 0.0)
        self.gi_to_systemic = kwargs.get('gi_to_systemic', 0.0)
        self.hepatic_clearance = kwargs.get('hepatic_clearance', 0.0)
        self.plasma_concentration = kwargs.get('plasma_concentration', 0.0)
        
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return {
            'mcc_to_gi': self.mcc_to_gi,
            'mcc_bb_to_BB': self.mcc_bb_to_BB,
            'lung_to_systemic': self.lung_to_systemic,
            'gi_to_systemic': self.gi_to_systemic,
            'hepatic_clearance': self.hepatic_clearance,
            'plasma_concentration': self.plasma_concentration
        }
        
    def from_dict(self, data: Dict[str, float]):
        """Update from dictionary."""
        for key, value in data.items():
            if hasattr(self, key):
                setattr(self, key, value)


# ============================================================================
# VALIDATION UTILITIES
# ============================================================================

def validate_parameters(params: Dict[str, Any], requirements: Dict[str, type]) -> bool:
    """Simple parameter validation."""
    for param_name, param_type in requirements.items():
        if param_name not in params:
            raise ValueError(f"Missing required parameter: {param_name}")
        if not isinstance(params[param_name], param_type):
            raise TypeError(f"Parameter {param_name} must be of type {param_type}")
    return True


def extract_array_parameter(params: Dict[str, Any], key: str, 
                          default_size: int, default_value: float = 0.0) -> np.ndarray:
    """Extract array parameter with size validation."""
    if key in params:
        arr = np.array(params[key])
        if len(arr) != default_size:
            arr = np.resize(arr, default_size)
        return arr
    else:
        return np.full(default_size, default_value)