"""
GI Tract PBBM Models

This module contains GI tract absorption models including:
1. Numba-optimized GIModel class
2. Factory functions for different GI configurations
3. GI component wrapper for model composition

Based on original GI tract logic from lung_pbbm.py lines 187-210.
"""

import numpy as np
from numba.experimental import jitclass
from numba import int32, float64
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass


# ============================================================================
# NUMBA GI MODEL
# ============================================================================

gi_model_spec = [
    ('n_compartments', int32),
    ('volumes', float64[:]),         # mL
    ('areas', float64[:]),           # cm²
    ('transit_times', float64[:]),   # s
    ('peff', float64),               # cm/s
    ('hepatic_extraction', float64), # fraction
    ('fu_gi', float64),              # fraction unbound in GI
    ('fu_plasma', float64),          # fraction unbound in plasma
    ('bp_ratio', float64),           # blood:plasma ratio
    ('state', float64[:]),           # pmol
]

@jitclass(gi_model_spec)
class GIModel:
    """
    Numba-optimized GI absorption model.
    
    Implements compartmental transit with permeability-limited absorption
    and first-pass hepatic extraction.
    """
    
    def __init__(self, n_compartments, volumes, areas, transit_times,
                 peff, hepatic_extraction, fu_gi, fu_plasma, bp_ratio):
        self.n_compartments = n_compartments
        self.volumes = volumes
        self.areas = areas
        self.transit_times = transit_times
        self.peff = peff
        self.hepatic_extraction = hepatic_extraction
        self.fu_gi = fu_gi
        self.fu_plasma = fu_plasma
        self.bp_ratio = bp_ratio
        self.state = np.zeros(n_compartments)
    
    def set_state(self, gi_amounts):
        """Set current GI state."""
        for i in range(self.n_compartments):
            self.state[i] = gi_amounts[i]
    
    def get_state_size(self):
        """Return number of state variables."""
        return self.n_compartments
    
    def compute_derivatives(self, mcc_input_rate, plasma_conc):
        """
        Compute GI absorption derivatives.
        
        Args:
            mcc_input_rate: MCC input from lung (pmol/s)
            plasma_conc: Plasma concentration (pmol/mL)
            
        Returns:
            tuple: (derivatives, net_gi_absorption, hepatic_clearance)
        """
        derivatives = np.zeros(self.n_compartments)
        total_absorption = 0.0
        
        for i in range(self.n_compartments):
            current_amount = self.state[i]
            
            # Input fluxes
            input_rate = 0.0
            
            # MCC input to first compartment (stomach)
            if i == 0:
                input_rate += mcc_input_rate
            
            # Transit from previous compartment
            if i > 0 and self.transit_times[i-1] > 0:
                transit_rate = self.state[i-1] / self.transit_times[i-1]
                input_rate += transit_rate
            
            # Output fluxes
            output_rate = 0.0
            
            # Transit to next compartment
            if self.transit_times[i] > 0:
                transit_rate = current_amount / self.transit_times[i]
                output_rate += transit_rate
            
            # Permeability-limited absorption
            if current_amount > 0 and self.volumes[i] > 0:
                local_conc = current_amount / self.volumes[i]
                
                # Two-way flux accounting for protein binding
                flux_gut_to_blood = self.peff * self.areas[i] * local_conc * self.fu_gi * self.bp_ratio / self.fu_plasma
                flux_blood_to_gut = self.peff * self.areas[i] * self.bp_ratio * (plasma_conc / 1000.0)  # Convert L to mL
                
                net_absorption = flux_gut_to_blood - flux_blood_to_gut
                if net_absorption > 0:
                    output_rate += net_absorption
                    total_absorption += net_absorption
            
            # Net derivative
            derivatives[i] = input_rate - output_rate
        
        # First-pass hepatic extraction
        hepatic_clearance = total_absorption * self.hepatic_extraction
        net_gi_absorption = total_absorption * (1.0 - self.hepatic_extraction)
     
        return derivatives, net_gi_absorption, hepatic_clearance


# ============================================================================
# FACTORY FUNCTIONS
# ============================================================================

def create_gi_model(subject, api):
    """
    Create GI absorption model from subject and API domain objects.
    
    Args:
        subject: Subject domain object (from entities.py)
        api: API domain object (from entities.py)
        
    Returns:
        GIModel instance
    """
    # Use subject.gi if available, otherwise fall back to default
    if subject.gi:
        gi_params = subject.gi.get_api_parameters(api.name)
        
        # Convert permeability from m/s to cm/s if available
        #peff_cm_s = getattr(api, 'effective_permeability_gi_cm_s', 1.535e-2)
        peff_cm_s = getattr(api, 'peff_GI', 1.535e-2)

        
        hepatic_extraction = getattr(api, 'hepatic_extraction_pct', 90.0) / 100.0
        
        return GIModel(
            n_compartments=len(gi_params.get('gi_vol', [0]*9)),
            volumes=np.array(gi_params.get('gi_vol', [50.0, 80.0, 200.0, 200.0, 200.0, 300.0, 300.0, 300.0, 300.0])),
            areas=np.array(gi_params.get('gi_area', [200.0, 800.0, 2000.0, 2000.0, 2000.0, 3000.0, 3000.0, 3000.0, 3000.0])),
            transit_times=np.array(gi_params.get('gi_tg', [60.0, 600.0, 600.0, 600.0, 3000.0, 3600.0, 1044.0, 15084.0, 45252.0])),
            peff=peff_cm_s,
            hepatic_extraction=hepatic_extraction,
            fu_gi=getattr(api, 'fraction_unbound', {}).get('Tissue', 0.0433), #getattr(api, 'fu_gi', 1.0),
            fu_plasma= getattr(api, 'fraction_unbound', {}).get('Plasma', 0.0433),#,getattr(api, 'fu_plasma', 0.5),
            bp_ratio= getattr(api, 'blood_plasma_ratio', 0.516),#, getattr(api, 'blood_plasma_ratio', 0.855)
        )
    else:
        # Fall back to default GI model
        return create_gi_model_default(api)


def create_gi_model_default(api_params=None):
    """
    Create GI model with default parameters.
    
    Args:
        api_params: Optional dict with drug-specific parameters
        
    Returns:
        GIModel instance with default GI tract structure
    """
    # Default 9-compartment GI tract
    n_comp = 9
    
    # Default volumes (mL)
    volumes = np.array([46.56, 41.56, 154.2, 122.3, 94.29, 70.53, 49.8, 47.49, 50.33])
    
    # Default areas (cm²)
    areas = np.array([0, 0, 0, 0, 300, 300, 144.72, 280.02, 41.77])  # BD formulation
    
    # Default transit times (s)
    transit_times = np.array([60.0, 600.0, 600.0, 600.0, 3000.0, 3600.0, 1044.0, 15084.0, 45252.0])
    
    # Extract API parameters from domain object
    if api_params:
        peff = getattr(api_params, 'effective_permeability_gi_cm_s', 1.535e-2)
        hepatic_extraction = getattr(api_params, 'hepatic_extraction_pct', 90.0) / 100.0  # Convert from pct to fraction
        fu_gi = getattr(api_params, 'fu_gi', 1.0)
        fu_plasma = getattr(api_params, 'fu_plasma', 0.5)
        bp_ratio = getattr(api_params, 'blood_plasma_ratio', 0.855)
    else:
        peff = 1.535e-2  # cm/s
        hepatic_extraction = 0.9
        fu_gi = 1.0
        fu_plasma = 0.5
        bp_ratio = 0.855
    
    return GIModel(
        n_compartments=n_comp,
        volumes=volumes,
        areas=areas,
        transit_times=transit_times,
        peff=peff,
        hepatic_extraction=hepatic_extraction,
        fu_gi=fu_gi,
        fu_plasma=fu_plasma,
        bp_ratio=bp_ratio
    )


# ============================================================================
# MODEL PARAMETERS
# ============================================================================

@dataclass
class GIPBBMParams:
    """Parameters for GI PBBM model."""
    formulation: str = "BD"  # "BD", "GP", or "FF"
    n_compartments: int = 9
    use_hepatic_extraction: bool = True
    
    # Compartment-specific overrides (optional)
    volumes_ml: np.ndarray = None
    areas_cm2: np.ndarray = None
    transit_times_s: np.ndarray = None
    
    # Drug-specific parameters
    effective_permeability_cm_s: float = 1.535e-4
    hepatic_extraction_fraction: float = 0.965
    fu_gi: float = 0.217
    fu_plasma: float = 0.516
    blood_plasma_ratio: float = 0.664