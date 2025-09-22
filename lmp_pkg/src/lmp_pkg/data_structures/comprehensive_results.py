"""Comprehensive PBBM results data structure.

This consolidates and cleans up the results handling from the original
comprehensive_data.py with better organization and validation.
"""

from __future__ import annotations
import numpy as np
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass, field
import warnings

from ..base import DataStructure


@dataclass
class BindingStateData:
    """Container for drug binding state data (bound, unbound, total)."""
    
    time_s: np.ndarray
    time_h: np.ndarray
    molecular_weight: float
    fu: float
    
    # Amount data [pmol]
    total_pmol: np.ndarray
    unbound_pmol: np.ndarray
    bound_pmol: np.ndarray
    
    # Concentration data [pmol/mL]
    total_conc_pmol_per_ml: np.ndarray = field(init=False)
    unbound_conc_pmol_per_ml: np.ndarray = field(init=False)
    bound_conc_pmol_per_ml: np.ndarray = field(init=False)
    
    # Mass data [pg]
    total_mass_pg: np.ndarray = field(init=False)
    unbound_mass_pg: np.ndarray = field(init=False)
    bound_mass_pg: np.ndarray = field(init=False)
    
    def __post_init__(self):
        """Calculate derived quantities."""
        # Concentrations (would need volume for this - simplified here)
        self.total_conc_pmol_per_ml = self.total_pmol  # Placeholder
        self.unbound_conc_pmol_per_ml = self.unbound_pmol
        self.bound_conc_pmol_per_ml = self.bound_pmol
        
        # Convert to mass [pg]: pmol × (μg/μmol) × (1e6 pg/μg) = pg
        self.total_mass_pg = self.total_pmol * self.molecular_weight * 1e6
        self.unbound_mass_pg = self.unbound_pmol * self.molecular_weight * 1e6
        self.bound_mass_pg = self.bound_pmol * self.molecular_weight * 1e6


@dataclass
class RegionalAmountData:
    """Regional drug distribution data."""
    
    region_name: str
    time_s: np.ndarray
    time_h: np.ndarray
    
    # Compartment amounts [pmol]
    epithelium_amounts: np.ndarray
    tissue_amounts: np.ndarray
    # Shallow-layer amounts for unbound calculations per reference
    epithelium_shallow_amounts: Optional[np.ndarray] = None
    tissue_shallow_amounts: Optional[np.ndarray] = None
    elf_amounts: Optional[np.ndarray] = None
    solid_drug_amounts: Optional[np.ndarray] = None

    # Total amounts
    total_amounts: np.ndarray = field(init=False)

    # Volumes for concentration conversion [mL]
    epithelium_volume_ml: Optional[float] = None
    tissue_volume_ml: Optional[float] = None

    # Derived concentrations [pmol/mL]
    epithelium_concentration_pmol_per_ml: Optional[np.ndarray] = field(init=False, default=None)
    tissue_concentration_pmol_per_ml: Optional[np.ndarray] = field(init=False, default=None)
    epithelium_tissue_concentration_pmol_per_ml: Optional[np.ndarray] = field(init=False, default=None)

    # Unbound fractions used for unbound concentration conversion
    fu_epithelium_calc: Optional[float] = None
    fu_tissue_calc: Optional[float] = None

    # Unbound concentrations [pmol/mL]
    epithelium_unbound_concentration_pmol_per_ml: Optional[np.ndarray] = field(init=False, default=None)
    tissue_unbound_concentration_pmol_per_ml: Optional[np.ndarray] = field(init=False, default=None)
    epithelium_tissue_unbound_concentration_pmol_per_ml: Optional[np.ndarray] = field(init=False, default=None)

    # Unbound amounts [pmol] for checking/reference alignment
    epithelium_unbound_amounts_pmol: Optional[np.ndarray] = field(init=False, default=None)
    tissue_unbound_amounts_pmol: Optional[np.ndarray] = field(init=False, default=None)

    # Regional exposure metrics (AUC over t_eval) [pmol·h/mL]
    auc_epithelium_pmol_h_per_ml: Optional[float] = field(init=False, default=None)
    auc_tissue_pmol_h_per_ml: Optional[float] = field(init=False, default=None)
    auc_epithelium_tissue_pmol_h_per_ml: Optional[float] = field(init=False, default=None)
    auc_epithelium_unbound_pmol_h_per_ml: Optional[float] = field(init=False, default=None)
    auc_tissue_unbound_pmol_h_per_ml: Optional[float] = field(init=False, default=None)
    auc_epithelium_tissue_unbound_pmol_h_per_ml: Optional[float] = field(init=False, default=None)

    # Binding states (if applicable)
    epithelium_binding: Optional[BindingStateData] = None
    tissue_binding: Optional[BindingStateData] = None

    def __post_init__(self):
        """Calculate total amounts."""
        self.total_amounts = self.epithelium_amounts + self.tissue_amounts
        if self.elf_amounts is not None:
            self.total_amounts += self.elf_amounts
        if self.solid_drug_amounts is not None:
            self.total_amounts += self.solid_drug_amounts

        # Compute concentrations if volumes are available
        try:
            if self.epithelium_volume_ml and self.epithelium_volume_ml > 0:
                self.epithelium_concentration_pmol_per_ml = self.epithelium_amounts / float(self.epithelium_volume_ml)
            if self.tissue_volume_ml and self.tissue_volume_ml > 0:
                self.tissue_concentration_pmol_per_ml = self.tissue_amounts / float(self.tissue_volume_ml)
            if (self.epithelium_volume_ml and self.epithelium_volume_ml > 0) and (self.tissue_volume_ml and self.tissue_volume_ml > 0):
                total_amt = self.epithelium_amounts + self.tissue_amounts
                total_vol = float(self.epithelium_volume_ml) + float(self.tissue_volume_ml)
                if total_vol > 0:
                    self.epithelium_tissue_concentration_pmol_per_ml = total_amt / total_vol
            # Compute AUCs over provided time_h grid
            if self.time_h is not None:
                import numpy as _np
                if self.epithelium_concentration_pmol_per_ml is not None:
                    self.auc_epithelium_pmol_h_per_ml = float(_np.trapz(self.epithelium_concentration_pmol_per_ml, self.time_h))
                if self.tissue_concentration_pmol_per_ml is not None:
                    self.auc_tissue_pmol_h_per_ml = float(_np.trapz(self.tissue_concentration_pmol_per_ml, self.time_h))
                if self.epithelium_tissue_concentration_pmol_per_ml is not None:
                    self.auc_epithelium_tissue_pmol_h_per_ml = float(_np.trapz(self.epithelium_tissue_concentration_pmol_per_ml, self.time_h))

            # Unbound per reference:
            # Epithelium unbound concentration = shallow_amount * fu_epi / V_epi
            if self.fu_epithelium_calc is not None and self.epithelium_shallow_amounts is not None and \
               self.epithelium_volume_ml and self.epithelium_volume_ml > 0:
                self.epithelium_unbound_concentration_pmol_per_ml = (
                    self.epithelium_shallow_amounts * float(self.fu_epithelium_calc) / float(self.epithelium_volume_ml)
                )
                # Unbound amount for epithelium = shallow_amount * fu_epi
                self.epithelium_unbound_amounts_pmol = self.epithelium_shallow_amounts * float(self.fu_epithelium_calc)
            # Tissue unbound concentration = shallow_amount * fu_tissue / V_tissue
            if self.fu_tissue_calc is not None and self.tissue_shallow_amounts is not None and \
               self.tissue_volume_ml and self.tissue_volume_ml > 0:
                self.tissue_unbound_concentration_pmol_per_ml = (
                    self.tissue_shallow_amounts * float(self.fu_tissue_calc) / float(self.tissue_volume_ml)
                )
                # Unbound amount for tissue = total_amount * fu_tissue
                self.tissue_unbound_amounts_pmol = self.tissue_amounts * float(self.fu_tissue_calc)

            # Combined unbound concentration based on shallow unbound amounts over combined volume
            if (self.epithelium_volume_ml and self.tissue_volume_ml and \
                self.epithelium_shallow_amounts is not None and self.tissue_shallow_amounts is not None and \
                self.fu_epithelium_calc is not None and self.fu_tissue_calc is not None):
                total_vol = float(self.epithelium_volume_ml) + float(self.tissue_volume_ml)
                if total_vol > 0:
                    total_unbound_amt_shallow = (
                        self.epithelium_shallow_amounts * float(self.fu_epithelium_calc) +
                        self.tissue_shallow_amounts * float(self.fu_tissue_calc)
                    )
                    self.epithelium_tissue_unbound_concentration_pmol_per_ml = total_unbound_amt_shallow / total_vol

            if self.time_h is not None:
                import numpy as _np
                if self.epithelium_unbound_concentration_pmol_per_ml is not None:
                    self.auc_epithelium_unbound_pmol_h_per_ml = float(_np.trapz(self.epithelium_unbound_concentration_pmol_per_ml, self.time_h))
                if self.tissue_unbound_concentration_pmol_per_ml is not None:
                    self.auc_tissue_unbound_pmol_h_per_ml = float(_np.trapz(self.tissue_unbound_concentration_pmol_per_ml, self.time_h))
                if self.epithelium_tissue_unbound_concentration_pmol_per_ml is not None:
                    self.auc_epithelium_tissue_unbound_pmol_h_per_ml = float(_np.trapz(self.epithelium_tissue_unbound_concentration_pmol_per_ml, self.time_h))
        except Exception:
            # Keep concentrations/AUCs as None on any failure
            pass


@dataclass
class FluxData:
    """Inter-compartment flux data."""
    
    time_s: np.ndarray
    time_h: np.ndarray
    
    # Systemic fluxes [pmol/s]
    systemic_absorption_rate: np.ndarray
    mucociliary_clearance_rate: np.ndarray
    gi_absorption_rate: Optional[np.ndarray] = None
    
    # Regional fluxes [pmol/s] 
    regional_systemic_absorption: Optional[Dict[str, np.ndarray]] = None
    regional_mcc_rates: Optional[Dict[str, np.ndarray]] = None
    
    # Dissolution fluxes [pmol/s] (if applicable)
    dissolution_rates: Optional[Dict[str, np.ndarray]] = None
    
    # PK elimination [pmol/s]
    elimination_rate: Optional[np.ndarray] = None


@dataclass
class PKResultsData:
    """Systemic PK results data.

    Supports either pmol/mL or ng/mL plasma concentrations. If only ng/mL
    is provided and `molecular_weight` is available (in μg/μmol), values are
    converted to pmol/mL for consistency.
    """

    time_s: np.ndarray
    time_h: np.ndarray

    # Plasma concentrations [pmol/mL]. If not provided, will be derived from ng/mL.
    plasma_concentration: Optional[np.ndarray]
    central_amounts: np.ndarray

    # Optional fields
    plasma_concentration_unbound: Optional[np.ndarray] = None
    peripheral_amounts: Optional[Dict[str, np.ndarray]] = None

    # Additional optional input in ng/mL and MW for conversion
    plasma_concentration_ng_per_ml: Optional[np.ndarray] = None
    molecular_weight: Optional[float] = None  # μg/μmol

    # Computed fields
    total_systemic_amounts: np.ndarray = field(init=False)

    # PK metrics
    auc_pmol_h_per_ml: Optional[float] = None
    cmax_pmol_per_ml: Optional[float] = None
    tmax_h: Optional[float] = None

    def __post_init__(self):
        """Calculate total systemic amounts and normalize concentration units."""
        # Normalize concentration units if needed
        if (self.plasma_concentration is None or len(self.plasma_concentration) == 0) and \
           (self.plasma_concentration_ng_per_ml is not None) and \
           (self.molecular_weight is not None) and self.molecular_weight > 0:
            # Convert ng/mL to pmol/mL: pmol/mL = (1000/MW) * ng/mL
            self.plasma_concentration = (1000.0 / float(self.molecular_weight)) * self.plasma_concentration_ng_per_ml

        # If pmol/mL provided but ng/mL missing and MW available, populate ng/mL for convenience
        if (self.plasma_concentration_ng_per_ml is None or len(self.plasma_concentration_ng_per_ml) == 0) and \
           (self.plasma_concentration is not None) and \
           (self.molecular_weight is not None):
            # ng/mL = (MW/1000) * pmol/mL
            self.plasma_concentration_ng_per_ml = (float(self.molecular_weight) / 1000.0) * self.plasma_concentration

        # Compute total systemic amounts
        self.total_systemic_amounts = self.central_amounts.copy()
        if self.peripheral_amounts:
            for peripheral_amts in self.peripheral_amounts.values():
                self.total_systemic_amounts += peripheral_amts


@dataclass
class MassBalanceData:
    """Mass balance tracking data."""
    
    time_s: np.ndarray
    time_h: np.ndarray
    
    # Initial conditions
    initial_deposition_pmol: float
    
    # Current amounts [pmol]
    lung_amounts: np.ndarray
    systemic_amounts: np.ndarray
    
    # Cumulative losses [pmol]
    cumulative_elimination: np.ndarray
    
    # Optional fields
    gi_amounts: Optional[np.ndarray] = None
    cumulative_other_losses: Optional[np.ndarray] = None
    
    # Mass balance metrics
    total_accounted: np.ndarray = field(init=False)
    mass_balance_error: np.ndarray = field(init=False)
    relative_error: np.ndarray = field(init=False)
    is_balanced: bool = field(init=False)
    
    def __post_init__(self):
        """Calculate mass balance metrics."""
        # Total accounted mass
        self.total_accounted = self.lung_amounts + self.systemic_amounts + self.cumulative_elimination
        if self.gi_amounts is not None:
            self.total_accounted += self.gi_amounts
        if self.cumulative_other_losses is not None:
            self.total_accounted += self.cumulative_other_losses
        
        # Mass balance error
        self.mass_balance_error = self.total_accounted - self.initial_deposition_pmol
        self.relative_error = self.mass_balance_error / self.initial_deposition_pmol
        
        # Check if balanced (within 1% tolerance)
        final_error = abs(self.relative_error[-1]) if len(self.relative_error) > 0 else 1.0
        self.is_balanced = final_error < 0.01


class ComprehensivePBBMResults(DataStructure):
    """Comprehensive PBBM simulation results.
    
    This consolidates all PBBM results into a single, well-organized
    data structure with validation and conversion utilities.
    """
    
    def __init__(self,
                 time_s: np.ndarray,
                 regional_data: Dict[str, RegionalAmountData],
                 pk_data: PKResultsData,
                 flux_data: FluxData,
                 mass_balance: MassBalanceData,
                 metadata: Optional[Dict[str, Any]] = None):
        
        self.time_s = time_s
        self.time_h = time_s / 3600.0
        
        self.regional_data = regional_data
        self.pk_data = pk_data
        self.flux_data = flux_data
        self.mass_balance = mass_balance
        self.metadata = metadata or {}
        
        # Validate consistency
        self._validate_time_consistency()
    
    def _validate_time_consistency(self):
        """Validate that all data has consistent time vectors."""
        base_length = len(self.time_s)
        
        # Check regional data
        for region_name, region_data in self.regional_data.items():
            if len(region_data.time_s) != base_length:
                warnings.warn(f"Time length mismatch for region {region_name}")
        
        # Check PK data
        if len(self.pk_data.time_s) != base_length:
            warnings.warn("Time length mismatch for PK data")
        
        # Check flux data
        if len(self.flux_data.time_s) != base_length:
            warnings.warn("Time length mismatch for flux data")
    
    def get_regional_amounts(self, region: str, compartment: str = 'total') -> np.ndarray:
        """Get amounts for a specific region and compartment."""
        if region not in self.regional_data:
            raise ValueError(f"Region {region} not found in results")
        
        region_data = self.regional_data[region]
        
        if compartment == 'total':
            return region_data.total_amounts
        elif compartment == 'epithelium':
            return region_data.epithelium_amounts
        elif compartment == 'tissue':
            return region_data.tissue_amounts
        elif compartment == 'elf' and region_data.elf_amounts is not None:
            return region_data.elf_amounts
        elif compartment == 'solid' and region_data.solid_drug_amounts is not None:
            return region_data.solid_drug_amounts
        else:
            raise ValueError(f"Compartment {compartment} not found for region {region}")
    
    def get_total_lung_amount(self) -> np.ndarray:
        """Get total drug amount in all lung regions."""
        total = np.zeros_like(self.time_s)
        for region_data in self.regional_data.values():
            total += region_data.total_amounts
        return total
    
    def calculate_pk_metrics(self) -> Dict[str, float]:
        """Calculate standard PK metrics."""
        plasma_conc = self.pk_data.plasma_concentration
        time_h = self.time_h
        
        # AUC (trapezoidal rule)
        auc = np.trapz(plasma_conc, time_h)
        
        # Cmax and Tmax
        cmax_idx = np.argmax(plasma_conc)
        cmax = plasma_conc[cmax_idx]
        tmax = time_h[cmax_idx]
        
        # Update PK data
        self.pk_data.auc_pmol_h_per_ml = auc
        self.pk_data.cmax_pmol_per_ml = cmax
        self.pk_data.tmax_h = tmax
        
        return {
            'auc_pmol_h_per_ml': auc,
            'cmax_pmol_per_ml': cmax,
            'tmax_h': tmax
        }
    
    def validate(self) -> bool:
        """Validate data structure consistency."""
        try:
            self._validate_time_consistency()
            
            # Check mass balance
            if not self.mass_balance.is_balanced:
                warnings.warn(f"Mass balance error: {self.mass_balance.relative_error[-1]*100:.3f}%")
            
            return True
        except Exception as e:
            warnings.warn(f"Results validation failed: {e}")
            return False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert results to dictionary format."""
        result = {
            'time_s': self.time_s,
            'time_h': self.time_h,
            'regional_data': {},
            'pk_data': {
                'plasma_concentration': self.pk_data.plasma_concentration,
                'central_amounts': self.pk_data.central_amounts,
                'total_systemic_amounts': self.pk_data.total_systemic_amounts,
                'auc_pmol_h_per_ml': self.pk_data.auc_pmol_h_per_ml,
                'cmax_pmol_per_ml': self.pk_data.cmax_pmol_per_ml,
                'tmax_h': self.pk_data.tmax_h
            },
            'flux_data': {
                'systemic_absorption_rate': self.flux_data.systemic_absorption_rate,
                'mucociliary_clearance_rate': self.flux_data.mucociliary_clearance_rate
            },
            'mass_balance': {
                'is_balanced': self.mass_balance.is_balanced,
                'relative_error': self.mass_balance.relative_error[-1] if len(self.mass_balance.relative_error) > 0 else 0,
                'total_accounted': self.mass_balance.total_accounted
            },
            'metadata': self.metadata
        }
        
        # Add regional data
        for region_name, region_data in self.regional_data.items():
            result['regional_data'][region_name] = {
                'epithelium_amounts': region_data.epithelium_amounts,
                'tissue_amounts': region_data.tissue_amounts,
                'total_amounts': region_data.total_amounts
            }
            if region_data.elf_amounts is not None:
                result['regional_data'][region_name]['elf_amounts'] = region_data.elf_amounts
        
        return result
    
    def from_dict(self, data: Dict[str, Any]) -> 'ComprehensivePBBMResults':
        """Create from dictionary format."""
        # This would be the inverse of to_dict()
        # Implementation would reconstruct all the dataclasses
        raise NotImplementedError("from_dict not yet implemented")
    
    def export_summary(self) -> Dict[str, Any]:
        """Export key summary metrics."""
        pk_metrics = self.calculate_pk_metrics()
        
        return {
            'simulation_duration_h': self.time_h[-1],
            'pk_metrics': pk_metrics,
            'mass_balance_passed': self.mass_balance.is_balanced,
            'mass_balance_error_percent': self.mass_balance.relative_error[-1] * 100,
            'final_lung_amount_pmol': self.get_total_lung_amount()[-1],
            'final_systemic_amount_pmol': self.pk_data.total_systemic_amounts[-1],
            'total_regions': len(self.regional_data),
            'metadata': self.metadata
        }
