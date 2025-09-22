"""
Systemic PK Models

This module contains pharmacokinetic models including:
1. Numba-optimized 1, 2, and 3-compartment PK models  
2. Factory functions for different PK configurations
3. PK component wrapper for model composition

Based on original PK logic from lung_pbbm.py lines 211-222.
"""

import numpy as np
from numba.experimental import jitclass
from numba import int32, float64
from typing import Dict, List, Any, Tuple, Optional, Mapping, Set
from dataclasses import dataclass

from ...contracts.stage import Stage
from ...contracts.types import PKInput, PKResult


# ============================================================================
# 1-COMPARTMENT PK MODEL
# ============================================================================

pk1c_spec = [
    ('vd_central', float64),     # L
    ('cl_systemic', float64),    # L/s
    ('k10', float64),            # 1/s
    ('state_central', float64),  # pmol
]

@jitclass(pk1c_spec)
class PK1CModel:
    """1-compartment PK model with first-order elimination."""
    
    def __init__(self, vd_central, cl_systemic):
        self.vd_central = vd_central
        self.cl_systemic = cl_systemic
        self.k10 = cl_systemic / vd_central if vd_central > 0 else 0.0
        self.state_central = 0.0
    
    def set_state(self, central):
        self.state_central = central
    
    def get_state_size(self):
        return 1
    
    def get_plasma_concentration(self):
        """Get plasma concentration (pmol/L) - for compatibility with lung model."""
        return self.state_central / self.vd_central if self.vd_central > 0 else 0.0
    
    def compute_derivatives(self, lung_absorption, gi_absorption):
        """
        Compute PK derivatives.
        
        Returns:
            float: d_central/dt
        """
        total_input = lung_absorption + gi_absorption
        elimination = self.k10 * self.state_central
        
        d_central = total_input - elimination
        
        return d_central


# ============================================================================
# 2-COMPARTMENT PK MODEL
# ============================================================================

pk2c_spec = [
    ('vd_central', float64),      # L
    ('vd_peripheral', float64),   # L
    ('cl_systemic', float64),     # L/s
    ('cl_distribution', float64), # L/s
    ('k10', float64),             # 1/s
    ('k12', float64),             # 1/s
    ('k21', float64),             # 1/s
    ('state_central', float64),   # pmol
    ('state_peripheral', float64), # pmol
]

@jitclass(pk2c_spec)
class PK2CModel:
    """2-compartment PK model with distribution to peripheral compartment."""
    
    def __init__(self, vd_central, vd_peripheral, cl_systemic, cl_distribution):
        self.vd_central = vd_central
        self.vd_peripheral = vd_peripheral
        self.cl_systemic = cl_systemic
        self.cl_distribution = cl_distribution
        
        # Calculate rate constants
        self.k10 = cl_systemic / vd_central if vd_central > 0 else 0.0
        self.k12 = cl_distribution / vd_central if vd_central > 0 else 0.0
        self.k21 = cl_distribution / vd_peripheral if vd_peripheral > 0 else 0.0
        
        self.state_central = 0.0
        self.state_peripheral = 0.0
    
    def set_state(self, central, peripheral):
        self.state_central = central
        self.state_peripheral = peripheral
    
    def get_state_size(self):
        return 2
    
    def get_plasma_concentration(self):
        """Get plasma concentration (pmol/L) - for compatibility with lung model."""
        return self.state_central / self.vd_central if self.vd_central > 0 else 0.0
    
    def compute_derivatives(self, lung_absorption, gi_absorption):
        """
        Compute PK derivatives.
        
        Returns:
            tuple: (d_central/dt, d_peripheral/dt)
        """
        total_input = lung_absorption + gi_absorption
        
        # Distribution fluxes
        distribution_c_to_p = self.k12 * self.state_central
        distribution_p_to_c = self.k21 * self.state_peripheral
        
        # Elimination (only from central)
        elimination = self.k10 * self.state_central
        
        d_central = total_input - distribution_c_to_p + distribution_p_to_c - elimination
        d_peripheral = distribution_c_to_p - distribution_p_to_c
        
        return d_central, d_peripheral


# ============================================================================
# 3-COMPARTMENT PK MODEL
# ============================================================================

pk3c_spec = [
    ('vd_central', float64),       # L
    ('vd_peripheral1', float64),   # L
    ('vd_peripheral2', float64),   # L
    ('cl_systemic', float64),      # L/s
    ('cl_distribution1', float64), # L/s
    ('cl_distribution2', float64), # L/s
    ('k10', float64),              # 1/s
    ('k12', float64),              # 1/s
    ('k21', float64),              # 1/s
    ('k13', float64),              # 1/s
    ('k31', float64),              # 1/s
    ('state_central', float64),    # pmol
    ('state_peripheral1', float64), # pmol
    ('state_peripheral2', float64), # pmol
]

@jitclass(pk3c_spec)
class PK3CModel:
    """3-compartment PK model with two peripheral compartments."""
    
    def __init__(self, vd_central, vd_peripheral1, vd_peripheral2, 
                 cl_systemic, cl_distribution1, cl_distribution2):
        self.vd_central = vd_central
        self.vd_peripheral1 = vd_peripheral1
        self.vd_peripheral2 = vd_peripheral2
        self.cl_systemic = cl_systemic
        self.cl_distribution1 = cl_distribution1
        self.cl_distribution2 = cl_distribution2
        
        # Calculate rate constants
        self.k10 = cl_systemic / vd_central if vd_central > 0 else 0.0
        self.k12 = cl_distribution1 / vd_central if vd_central > 0 else 0.0
        self.k21 = cl_distribution1 / vd_peripheral1 if vd_peripheral1 > 0 else 0.0
        self.k13 = cl_distribution2 / vd_central if vd_central > 0 else 0.0
        self.k31 = cl_distribution2 / vd_peripheral2 if vd_peripheral2 > 0 else 0.0
        
        self.state_central = 0.0
        self.state_peripheral1 = 0.0
        self.state_peripheral2 = 0.0
    
    def set_state(self, central, peripheral1, peripheral2):
        self.state_central = central
        self.state_peripheral1 = peripheral1
        self.state_peripheral2 = peripheral2
    
    def get_state_size(self):
        return 3
    
    def get_plasma_concentration(self):
        """Get plasma concentration (pmol/L) - for compatibility with lung model."""
        return self.state_central / self.vd_central if self.vd_central > 0 else 0.0
    
    def compute_derivatives(self, lung_absorption, gi_absorption):
        """
        Compute PK derivatives.
        
        Returns:
            tuple: (d_central/dt, d_peripheral1/dt, d_peripheral2/dt)
        """
        total_input = lung_absorption + gi_absorption
        
        # Distribution fluxes
        distribution_c_to_p1 = self.k12 * self.state_central
        distribution_p1_to_c = self.k21 * self.state_peripheral1
        distribution_c_to_p2 = self.k13 * self.state_central
        distribution_p2_to_c = self.k31 * self.state_peripheral2
        
        # Elimination (only from central)
        elimination = self.k10 * self.state_central

      
        d_central = (total_input 
                    - distribution_c_to_p1 + distribution_p1_to_c 
                    - distribution_c_to_p2 + distribution_p2_to_c 
                    - elimination)
        d_peripheral1 = distribution_c_to_p1 - distribution_p1_to_c
        d_peripheral2 = distribution_c_to_p2 - distribution_p2_to_c
        
        return d_central, d_peripheral1, d_peripheral2


# ============================================================================
# FACTORY FUNCTIONS
# ============================================================================

def create_pk1c_from_clearance(vd_central_L: float, cl_systemic_L_h: float):
    """
    Create 1-compartment PK model from clearance parameters.
    
    Args:
        vd_central_L: Central volume of distribution (L)
        cl_systemic_L_h: Systemic clearance (L/h)
        
    Returns:
        PK1CModel instance
    """
    cl_systemic_L_s = cl_systemic_L_h / 3600.0  # Convert to L/s
    
    return PK1CModel(
        vd_central=vd_central_L,
        cl_systemic=cl_systemic_L_s
    )


def create_pk2c_from_clearance(vd_central_L: float, vd_peripheral_L: float,
                              cl_systemic_L_h: float, cl_distribution_L_h: float):
    """
    Create 2-compartment PK model from clearance parameters.
    
    Args:
        vd_central_L: Central volume of distribution (L)
        vd_peripheral_L: Peripheral volume of distribution (L)
        cl_systemic_L_h: Systemic clearance (L/h)
        cl_distribution_L_h: Distribution clearance (L/h)
        
    Returns:
        PK2CModel instance
    """
    cl_systemic_L_s = cl_systemic_L_h / 3600.0
    cl_distribution_L_s = cl_distribution_L_h / 3600.0
    
    return PK2CModel(
        vd_central=vd_central_L,
        vd_peripheral=vd_peripheral_L,
        cl_systemic=cl_systemic_L_s,
        cl_distribution=cl_distribution_L_s
    )


def create_pk3c_from_clearance(vd_central_L: float, vd_peripheral1_L: float, vd_peripheral2_L: float,
                              cl_systemic_L_h: float, cl_distribution1_L_h: float, cl_distribution2_L_h: float):
    """
    Create 3-compartment PK model from clearance parameters.
    
    Args:
        vd_central_L: Central volume of distribution (L)
        vd_peripheral1_L: Peripheral 1 volume of distribution (L)
        vd_peripheral2_L: Peripheral 2 volume of distribution (L)
        cl_systemic_L_h: Systemic clearance (L/h)
        cl_distribution1_L_h: Distribution clearance to peripheral 1 (L/h)
        cl_distribution2_L_h: Distribution clearance to peripheral 2 (L/h)
        
    Returns:
        PK3CModel instance
    """
    cl_systemic_L_s = cl_systemic_L_h / 3600.0
    cl_distribution1_L_s = cl_distribution1_L_h / 3600.0
    cl_distribution2_L_s = cl_distribution2_L_h / 3600.0
    
    return PK3CModel(
        vd_central=vd_central_L,
        vd_peripheral1=vd_peripheral1_L,
        vd_peripheral2=vd_peripheral2_L,
        cl_systemic=cl_systemic_L_s,
        cl_distribution1=cl_distribution1_L_s,
        cl_distribution2=cl_distribution2_L_s
    )


def create_pk3c_from_api(api):
    """
    Create 3-compartment PK model from API object (matching original).
    
    Args:
        api: API object with PK parameters
        
    Returns:
        PK3CModel instance
    """
    # Extract parameters exactly like original (lines 289-290)
    V_central_L = api.V_central_L
    k12_s = api.k12_h / 3600.0
    k21_s = api.k21_h / 3600.0  
    k13_s = api.k13_h / 3600.0
    k31_s = api.k31_h / 3600.0
    k10_s = (api.CL_h / 3600.0) / V_central_L
    
    # Calculate volumes from rate constants (reverse engineering)
    # This assumes default peripheral volumes if not provided
    cl_distribution1 = k12_s * V_central_L
    cl_distribution2 = k13_s * V_central_L
    vd_peripheral1 = cl_distribution1 / k21_s if k21_s > 0 else V_central_L
    vd_peripheral2 = cl_distribution2 / k31_s if k31_s > 0 else V_central_L * 5.0
    
    cl_systemic = k10_s * V_central_L
    
    return PK3CModel(
        vd_central=V_central_L,
        vd_peripheral1=vd_peripheral1,
        vd_peripheral2=vd_peripheral2,
        cl_systemic=cl_systemic,
        cl_distribution1=cl_distribution1,
        cl_distribution2=cl_distribution2
    )


# ============================================================================
# MODEL PARAMETERS
# ============================================================================

@dataclass
class PK1CParams:
    """Parameters for 1-compartment PK model."""
    vd_central_L: float
    cl_systemic_L_h: float
    
    def validate(self):
        if self.vd_central_L <= 0:
            raise ValueError("Central volume must be positive")
        if self.cl_systemic_L_h <= 0:
            raise ValueError("Systemic clearance must be positive")


@dataclass
class PK2CParams:
    """Parameters for 2-compartment PK model."""
    vd_central_L: float
    vd_peripheral_L: float
    cl_systemic_L_h: float
    cl_distribution_L_h: float
    
    def validate(self):
        if self.vd_central_L <= 0:
            raise ValueError("Central volume must be positive")
        if self.vd_peripheral_L <= 0:
            raise ValueError("Peripheral volume must be positive")
        if self.cl_systemic_L_h <= 0:
            raise ValueError("Systemic clearance must be positive")
        if self.cl_distribution_L_h <= 0:
            raise ValueError("Distribution clearance must be positive")


@dataclass
class PK3CParams:
    """Parameters for 3-compartment PK model."""
    vd_central_L: float
    vd_peripheral1_L: float
    vd_peripheral2_L: float
    cl_systemic_L_h: float
    cl_distribution1_L_h: float
    cl_distribution2_L_h: float
    
    def validate(self):
        if self.vd_central_L <= 0:
            raise ValueError("Central volume must be positive")
        if self.vd_peripheral1_L <= 0:
            raise ValueError("Peripheral 1 volume must be positive")
        if self.vd_peripheral2_L <= 0:
            raise ValueError("Peripheral 2 volume must be positive")
        if self.cl_systemic_L_h <= 0:
            raise ValueError("Systemic clearance must be positive")
        if self.cl_distribution1_L_h <= 0:
            raise ValueError("Distribution clearance 1 must be positive")
        if self.cl_distribution2_L_h <= 0:
            raise ValueError("Distribution clearance 2 must be positive")


# ============================================================================
# UNIFIED PK MODEL SELECTOR
# ============================================================================

def create_pk_model(model_type: str, params: Dict[str, Any]):
    """
    Factory function to create any PK model type.

    Args:
        model_type: "1c", "2c", or "3c"
        params: Dictionary of model parameters

    Returns:
        PK model instance
    """
    if model_type == "1c":
        return create_pk1c_from_clearance(
            params['vd_central_L'],
            params['cl_systemic_L_h']
        )
    elif model_type == "2c":
        return create_pk2c_from_clearance(
            params['vd_central_L'],
            params['vd_peripheral_L'],
            params['cl_systemic_L_h'],
            params['cl_distribution_L_h']
        )
    elif model_type == "3c":
        return create_pk3c_from_clearance(
            params['vd_central_L'],
            params['vd_peripheral1_L'],
            params['vd_peripheral2_L'],
            params['cl_systemic_L_h'],
            params['cl_distribution1_L_h'],
            params['cl_distribution2_L_h']
        )
    else:
        raise ValueError(f"Unknown PK model type: {model_type}")


# ============================================================================
# STAGE WRAPPERS FOR PIPELINE REGISTRATION
# ============================================================================


class _BaseSystemicPKStage(Stage[PKInput, PKResult]):
    """Stage that adapts PBPK comprehensive results into PKResult outputs."""

    name: str = "systemic_pk"

    def __init__(self, model_type: str = "3c") -> None:
        self.model_type = model_type

    @property
    def provides(self) -> Set[str]:
        return {"pk"}

    @property
    def requires(self) -> Set[str]:
        return {"pbbm"}

    def run(self, data: PKInput) -> PKResult:
        params: Dict[str, Any] = dict(data.params or {})

        pbbm_comprehensive = params.get('pbbm_comprehensive')
        pbbm_result = params.get('pbbm_result')
        if pbbm_comprehensive is None and pbbm_result is not None:
            pbbm_comprehensive = getattr(pbbm_result, 'comprehensive', None)

        if pbbm_comprehensive is None:
            raise ValueError("Systemic PK stage requires PBPK comprehensive results in params['pbbm_comprehensive'].")

        time_s = getattr(pbbm_comprehensive, 'time_s', None)
        if time_s is None or (hasattr(time_s, '__len__') and len(time_s) == 0):
            time_candidate = params.get('pbbm_time_s')
            if time_candidate is None and pbbm_result is not None:
                time_candidate = getattr(pbbm_result, 't', None)
            if time_candidate is None and data.pulmonary_input is not None:
                pul = np.asarray(data.pulmonary_input)
                if pul.ndim == 2 and pul.shape[1] >= 1:
                    time_candidate = pul[:, 0]
            if time_candidate is None:
                raise ValueError("Systemic PK stage could not determine time vector from PBPK results.")
            time_s = np.asarray(time_candidate, dtype=float)
        else:
            time_s = np.asarray(time_s, dtype=float)

        pk_data = getattr(pbbm_comprehensive, 'pk_data', None)
        if pk_data is None:
            raise ValueError("PBPK comprehensive results do not contain pk_data.")

        central_amounts = np.asarray(getattr(pk_data, 'central_amounts', np.zeros_like(time_s)), dtype=float)

        plasma_ng = getattr(pk_data, 'plasma_concentration_ng_per_ml', None)
        if plasma_ng is None or len(plasma_ng) == 0:
            plasma_pmol = getattr(pk_data, 'plasma_concentration', None)
            if plasma_pmol is None or len(plasma_pmol) == 0:
                raise ValueError("PBPK PK data missing plasma concentration information.")
            plasma_ng = np.asarray(plasma_pmol, dtype=float)
            mw = None
            if hasattr(pk_data, 'molecular_weight') and pk_data.molecular_weight:
                mw = float(pk_data.molecular_weight)
            elif data.api is not None and hasattr(data.api, 'molecular_weight'):
                mw = float(getattr(data.api, 'molecular_weight'))
            if mw:
                plasma_ng = plasma_ng * (mw / 1000.0)
        else:
            plasma_ng = np.asarray(plasma_ng, dtype=float)

        compartments: Dict[str, np.ndarray] = {
            'central': central_amounts
        }
        peripheral_amounts = getattr(pk_data, 'peripheral_amounts', None)
        if peripheral_amounts:
            for name, values in peripheral_amounts.items():
                compartments[name] = np.asarray(values, dtype=float)

        metadata = {
            'source': 'pbpk_comprehensive',
            'model_type': self.model_type,
            'units': 'ng_per_ml',
            'pk_metrics': {
                'auc_pmol_h_per_ml': getattr(pk_data, 'auc_pmol_h_per_ml', None),
                'cmax_pmol_per_ml': getattr(pk_data, 'cmax_pmol_per_ml', None),
                'tmax_h': getattr(pk_data, 'tmax_h', None),
            }
        }

        return PKResult(
            t=time_s,
            conc_plasma=plasma_ng,
            compartments=compartments,
            metadata=metadata,
        )


class SystemicPKStage1C(_BaseSystemicPKStage):
    def __init__(self) -> None:
        super().__init__(model_type="1c")


class SystemicPKStage2C(_BaseSystemicPKStage):
    def __init__(self) -> None:
        super().__init__(model_type="2c")


class SystemicPKStage3C(_BaseSystemicPKStage):
    def __init__(self) -> None:
        super().__init__(model_type="3c")


class NullSystemicPKStage(Stage[PKInput, PKResult]):
    """Fallback systemic PK stage that returns zero concentrations."""

    name: str = "systemic_pk_null"

    @property
    def provides(self) -> Set[str]:
        return {"pk"}

    @property
    def requires(self) -> Set[str]:
        return set()

    def run(self, data: PKInput) -> PKResult:
        params: Dict[str, Any] = dict(data.params or {})

        time_s = params.get('pbbm_time_s')
        if time_s is None and data.pulmonary_input is not None:
            pul = np.asarray(data.pulmonary_input)
            if pul.ndim == 2 and pul.shape[1] >= 1:
                time_s = pul[:, 0]
        if time_s is None:
            duration_h = float(params.get('duration_h', 24.0))
            time_s = np.linspace(0.0, duration_h * 3600.0, 2)
        else:
            time_s = np.asarray(time_s, dtype=float)

        zeros = np.zeros_like(time_s, dtype=float)
        compartments = {'central': zeros.copy()}

        return PKResult(
            t=time_s,
            conc_plasma=zeros,
            compartments=compartments,
            metadata={'source': 'null_systemic_pk'}
        )
