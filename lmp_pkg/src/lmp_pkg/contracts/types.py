"""Type definitions for stage inputs and outputs."""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Mapping, Optional
import numpy as np


@dataclass(frozen=True)
class DepositionInput:
    """Input data for deposition models."""
    
    subject: dict[str, Any]
    """Subject parameters (age, height, weight, etc.)"""
    
    product: dict[str, Any] 
    """Product parameters (formulation, device properties, etc.)"""
    
    maneuver: dict[str, Any]
    """Inhalation profile parameters (flow rate, volume, etc.)"""

    api: Optional[dict[str, Any]] = None
    """API parameters when deposition needs systemic metadata"""
    
    particle_grid: Optional[np.ndarray] = None
    """Optional particle size grid for detailed calculations"""
    
    params: Optional[Mapping[str, float]] = None
    """Additional model-specific parameters"""

    cfd_result: Optional['CFDResult'] = None
    """Optional CFD surrogate output for throat/lung fractions"""


@dataclass(frozen=True) 
class DepositionResult:
    """Output from deposition models."""
    
    region_ids: np.ndarray
    """Array of lung region identifiers"""
    
    elf_initial_amounts: np.ndarray
    """Initial drug amounts in epithelial lining fluid by region"""
    
    metadata: Optional[Mapping[str, Any]] = None
    """Additional model outputs and diagnostics"""


@dataclass(frozen=True)
class CFDInput:
    """Input data for CFD surrogate models."""

    subject: dict[str, Any]
    product: dict[str, Any]
    maneuver: dict[str, Any]
    api: Optional[dict[str, Any]] = None
    params: Optional[Mapping[str, Any]] = None


@dataclass(frozen=True)
class CFDResult:
    """Output from CFD surrogate models describing global deposition metrics."""

    mmad: float
    gsd: float
    mt_deposition_fraction: float
    metadata: Optional[Mapping[str, Any]] = None


@dataclass(frozen=True)
class PBBKInput:
    """Input data for lung PBPK models."""

    subject: dict[str, Any]
    """Subject parameters"""
    
    api: dict[str, Any]
    """Active pharmaceutical ingredient properties"""
    
    lung_seed: Optional[DepositionResult] = None
    """Optional deposition results to seed lung compartments"""
    
    params: Optional[Mapping[str, float]] = None
    """Model-specific parameters"""


@dataclass(frozen=True)
class PBBKResult:
    """Output from lung PBPK models."""

    t: Optional[np.ndarray] = None
    """Time points"""

    y: Optional[np.ndarray] = None
    """Solution matrix [time, compartments]"""

    region_slices: Optional[dict[str, slice]] = None
    """Mapping from compartment names to solution matrix slices"""

    pulmonary_outflow: Optional[np.ndarray] = None
    """Time series of drug absorption into systemic circulation"""

    metadata: Optional[Mapping[str, Any]] = None
    """Solver diagnostics and additional outputs"""

    comprehensive: Optional[Any] = None
    """Comprehensive PBPK results (optional rich structure)."""


@dataclass(frozen=True)
class PKInput:
    """Input data for systemic PK models."""
    
    subject: dict[str, Any]
    """Subject parameters"""
    
    api: dict[str, Any]
    """Active pharmaceutical ingredient properties"""
    
    pulmonary_input: Optional[np.ndarray] = None
    """Time series of pulmonary absorption"""
    
    gi_input: Optional[np.ndarray] = None
    """Time series of GI absorption (for swallowed fraction)"""
    
    params: Optional[Mapping[str, float]] = None
    """Model-specific parameters"""


@dataclass(frozen=True)
class PKResult:
    """Output from systemic PK models."""
    
    t: np.ndarray
    """Time points"""
    
    conc_plasma: np.ndarray
    """Plasma concentration time series"""
    
    compartments: dict[str, np.ndarray]
    """Time series for each compartment (central, peripheral, etc.)"""
    
    metadata: Optional[Mapping[str, Any]] = None
    """Solver diagnostics and additional outputs"""


@dataclass(frozen=True)
class EfficacyInput:
    """Input data for efficacy models."""
    
    subject: dict[str, Any]
    """Subject parameters"""
    
    api: dict[str, Any]
    """Active pharmaceutical ingredient properties"""
    
    pk_results: Optional[PKResult] = None
    """PK model results for exposure-response modeling"""
    
    params: Optional[Mapping[str, float]] = None
    """Model-specific parameters (doses, study design, etc.)"""


@dataclass(frozen=True)
class EfficacyResult:
    """Output from efficacy models."""
    
    t: np.ndarray
    """Time points for efficacy endpoint"""
    
    efficacy_endpoint: np.ndarray
    """Time series of efficacy endpoint values"""
    
    endpoint_name: str
    """Name of the efficacy endpoint (e.g., 'fev1_change_L')"""
    
    baseline_value: float
    """Baseline value of the efficacy endpoint"""
    
    metadata: Optional[Mapping[str, Any]] = None
    """Model diagnostics and additional outputs"""


@dataclass(frozen=True)
class RunResult:
    """Complete simulation results."""

    run_id: str
    """Unique identifier for this simulation run"""
    
    config: dict[str, Any]
    """Configuration used for this run"""
    
    cfd: Optional[CFDResult] = None
    """CFD surrogate results"""

    deposition: Optional[DepositionResult] = None
    """Deposition model results"""
    
    pbbk: Optional[PBBKResult] = None  
    """Lung PBPK model results"""
    
    pk: Optional[PKResult] = None
    """Systemic PK model results"""
    
    efficacy: Optional[EfficacyResult] = None
    """Efficacy model results"""
    
    analysis: Optional[dict[str, Any]] = None
    """Analysis results (AUC, Cmax, etc.)"""
    
    runtime_seconds: float = 0.0
    """Total execution time"""
    
    metadata: Optional[Mapping[str, Any]] = None
    """Run-level diagnostics and information"""
