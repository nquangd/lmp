"""Intuitive hierarchical data structure for PBBM results.

Updated to support unit-aware access without unit-coded attribute names.
Examples:
  - subject.Al.Epithelium.Concentration.Unbound  -> returns array in default unit
  - subject.Al.Epithelium.Amount.Total          -> returns array in default unit

Use `UnitContext` to control output units globally per Subject instance.
"""

from __future__ import annotations
from typing import Dict, Any, Optional, List
import numpy as np
from dataclasses import dataclass

from .units import UnitContext


class ConcentrationData:
    """Concentration container in internal units with unit-aware accessors.

    Internal units: pmol/mL. Accessors convert to desired output units via
    `UnitContext` (default on the Subject).
    """

    def __init__(self, time_points: np.ndarray, amounts_pmol: np.ndarray, volume_ml: float,
                 fu: float = 1.0, molecular_weight: float = 250.0, units: Optional[UnitContext] = None):
        self.time_s = time_points
        self.time_h = time_points / 3600.0
        self._volume_ml = volume_ml
        self._fu = fu
        self._mw = molecular_weight
        self._units = units or UnitContext(molecular_weight_ug_per_umol=molecular_weight)

        if volume_ml > 0:
            total_pmol_ml = amounts_pmol / volume_ml
        else:
            total_pmol_ml = np.zeros_like(amounts_pmol)
        self._total_pmol_per_ml = total_pmol_ml
        self._unbound_pmol_per_ml = total_pmol_ml * fu

    @property
    def Total(self) -> np.ndarray:
        """Total concentration, converted to default unit (UnitContext)."""
        return self._units.concentration_from_pmol_per_ml(self._total_pmol_per_ml)

    @property
    def Unbound(self) -> np.ndarray:
        """Unbound concentration, converted to default unit (UnitContext)."""
        return self._units.concentration_from_pmol_per_ml(self._unbound_pmol_per_ml)

    def set_units(self, units: UnitContext) -> None:
        """Update the unit context used for conversions."""
        self._units = units


class ConcentrationAccessor:  # Back-compat shim (deprecated)
    def __init__(self, pmol_per_ml: np.ndarray, pg_per_ml: np.ndarray, ng_per_ml: np.ndarray):
        self.pmol_per_ml = pmol_per_ml
        self.pg_per_ml = pg_per_ml
        self.ng_per_ml = ng_per_ml
    def __repr__(self) -> str:
        if len(self.ng_per_ml) > 0:
            return f"Concentration(max={np.max(self.ng_per_ml):.3f} ng/mL, final={self.ng_per_ml[-1]:.3f} ng/mL)"
        return "Concentration(empty)"


class AmountData:
    """Amount container with unit-aware access.

    Internal units: pmol. Accessors convert to UnitContext output units.
    """

    def __init__(self, time_points: np.ndarray, amounts_pmol: np.ndarray, fu: float = 1.0,
                 molecular_weight: float = 250.0, units: Optional[UnitContext] = None):
        self.time_s = time_points
        self.time_h = time_points / 3600.0
        self._pmol = amounts_pmol
        self._unbound_pmol = amounts_pmol * fu
        self._units = units or UnitContext(molecular_weight_ug_per_umol=molecular_weight)

    @property
    def Total(self) -> np.ndarray:
        return self._units.amount_from_pmol(self._pmol)

    @property
    def Unbound(self) -> np.ndarray:
        return self._units.amount_from_pmol(self._unbound_pmol)

    def __repr__(self) -> str:
        vals = self.Total
        if len(vals) > 0:
            return f"Amount(max={np.max(vals):.3f}, final={vals[-1]:.3f})"
        return "Amount(empty)"

    def set_units(self, units: UnitContext) -> None:
        """Update the unit context used for conversions."""
        self._units = units


class CompartmentData:
    """Data for a single compartment (e.g., Epithelium, Tissue)."""
    
    def __init__(self, name: str, time_points: np.ndarray, amounts_pmol: np.ndarray,
                 volume_ml: float, fu: float = 1.0, molecular_weight: float = 250.0,
                 units: Optional[UnitContext] = None):
        """Initialize compartment data.
        
        Args:
            name: Compartment name (e.g., "Epithelium", "Tissue")
            time_points: Time points [s]
            amounts_pmol: Amount time series [pmol]
            volume_ml: Compartment volume [mL]
            fu: Fraction unbound
            molecular_weight: Molecular weight [g/mol]
        """
        self.name = name
        self.volume_ml = volume_ml
        self._units = units or UnitContext(molecular_weight_ug_per_umol=molecular_weight)

        # Amount and concentration data
        self.Amount = AmountData(time_points, amounts_pmol, fu, molecular_weight, self._units)
        self.Concentration = ConcentrationData(time_points, amounts_pmol, volume_ml, fu, molecular_weight, self._units)
        
    def __repr__(self) -> str:
        return f"CompartmentData({self.name}, volume={self.volume_ml:.2f} mL)"

    def set_units(self, units: UnitContext) -> None:
        self._units = units
        self.Amount.set_units(units)
        self.Concentration.set_units(units)


class RegionData:
    """Data for a lung region or generation (e.g., ET, TB, P1, P2, Al)."""
    
    def __init__(self, name: str, time_points: np.ndarray, molecular_weight: float = 250.0,
                 units: Optional[UnitContext] = None):
        """Initialize region data.
        
        Args:
            name: Region name (e.g., "ET", "TB", "P1", "P2", "Al") 
            time_points: Time points [s]
            molecular_weight: Molecular weight [g/mol]
        """
        self.name = name
        self.time_points = time_points
        self.molecular_weight = molecular_weight
        self._compartments: Dict[str, CompartmentData] = {}
        self._units = units or UnitContext(molecular_weight_ug_per_umol=molecular_weight)
        
    def add_compartment(self, compartment_name: str, amounts_pmol: np.ndarray,
                        volume_ml: float, fu: float = 1.0) -> None:
        """Add a compartment to this region.
        
        Args:
            compartment_name: Name like "ELF", "Epithelium", "Tissue"
            amounts_pmol: Amount time series [pmol]
            volume_ml: Compartment volume [mL]
            fu: Fraction unbound
        """
        compartment = CompartmentData(
            compartment_name, self.time_points, amounts_pmol, volume_ml, fu, self.molecular_weight, self._units
        )
        self._compartments[compartment_name] = compartment
        
        # Make compartment accessible as attribute
        setattr(self, compartment_name, compartment)
    
    def get_compartments(self) -> List[str]:
        """Get list of available compartments."""
        return list(self._compartments.keys())
        
    def __repr__(self) -> str:
        compartments = ", ".join(self._compartments.keys())
        return f"RegionData({self.name}, compartments=[{compartments}])"

    def set_units(self, units: UnitContext) -> None:
        self._units = units
        for comp in self._compartments.values():
            comp.set_units(units)


class SubjectPBBMData:
    """Complete PBBM data for a single subject with intuitive access patterns.

    Supports access like: subject.Al.Epithelium.Concentration.Unbound
    and subject.Al.Epithelium.Amount.Total with units controlled by UnitContext.
    """
    
    def __init__(self, subject_id: str, time_points: np.ndarray, molecular_weight: float = 250.0,
                 units: Optional[UnitContext] = None):
        """Initialize subject PBBM data.
        
        Args:
            subject_id: Subject identifier
            time_points: Time points [s]
            molecular_weight: API molecular weight [g/mol]
        """
        self.subject_id = subject_id
        self.time_points = time_points
        self.molecular_weight = molecular_weight
        self._regions: Dict[str, RegionData] = {}
        self._units = units or UnitContext(molecular_weight_ug_per_umol=molecular_weight)
        
        # Systemic compartments
        self.Plasma: Optional[CompartmentData] = None
        self.Peripheral1: Optional[CompartmentData] = None
        self.Peripheral2: Optional[CompartmentData] = None
        
        # Summary metrics
        self.Metrics: Dict[str, Any] = {}
        
    def add_region(self, region_name: str) -> RegionData:
        """Add a lung region.
        
        Args:
            region_name: Region name like "ET", "BB", "bb", "Al"
            
        Returns:
            RegionData object for further configuration
        """
        region = RegionData(region_name, self.time_points, self.molecular_weight, self._units)
        self._regions[region_name] = region
        
        # Make region accessible as attribute  
        setattr(self, region_name, region)
        
        return region
    
    def add_systemic_compartment(self, compartment_name: str, amounts_pmol: np.ndarray,
                                 volume_L: float, fu: float = 1.0) -> None:
        """Add a systemic compartment.
        
        Args:
            compartment_name: Name like "Plasma", "Peripheral1", "Peripheral2" 
            amounts_pmol: Amount time series [pmol]
            volume_L: Compartment volume [L]
            fu: Fraction unbound
        """
        volume_ml = volume_L * 1000.0  # Convert L to mL
        compartment = CompartmentData(
            compartment_name, self.time_points, amounts_pmol, volume_ml, fu, self.molecular_weight, self._units
        )
        setattr(self, compartment_name, compartment)
    
    def add_metrics(self, metrics: Dict[str, Any]) -> None:
        """Add summary metrics (Cmax, AUC, etc.).
        
        Args:
            metrics: Dictionary of PK/PD metrics
        """
        self.Metrics.update(metrics)
    
    def get_regions(self) -> List[str]:
        """Get list of available regions."""
        return list(self._regions.keys())
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of data structure."""
        return {
            'subject_id': self.subject_id,
            'regions': self.get_regions(),
            'systemic_compartments': [name for name in ['Plasma', 'Peripheral1', 'Peripheral2'] 
                                    if getattr(self, name, None) is not None],
            'metrics': list(self.Metrics.keys()),
            'time_range_h': [self.time_points[0]/3600, self.time_points[-1]/3600] if len(self.time_points) > 0 else [0, 0]
        }
    
    def __repr__(self) -> str:
        regions = ", ".join(self._regions.keys())
        return f"SubjectPBBMData({self.subject_id}, regions=[{regions}])"

    def set_unit_context(self, units: UnitContext) -> None:
        """Set a new UnitContext for this subject and all nested items."""
        self._units = units
        for region in self._regions.values():
            region.set_units(units)
        for name in ['Plasma', 'Peripheral1', 'Peripheral2']:
            comp = getattr(self, name, None)
            if comp is not None:
                comp.set_units(units)


def create_example_subject_data() -> SubjectPBBMData:
    """Create example subject data to demonstrate the access patterns."""
    
    # Example time points (6 hours, 10-minute intervals)
    time_points = np.linspace(0, 6*3600, 37)  # 0 to 6 hours in seconds
    
    # Create subject data
    subject = SubjectPBBMData("Subject_001", time_points, molecular_weight=250.0)
    
    # Add lung regions
    for region_name in ["ET", "TB", "P1", "P2", "Al"]:
        region = subject.add_region(region_name)
        
        # Add compartments to each region
        # ELF compartment
        elf_amounts = 100.0 * np.exp(-time_points / 3600.0)  # Exponential decay
        region.add_compartment("ELF", elf_amounts, volume_ml=1.0, fu=1.0)
        
        # Epithelium compartment  
        epi_amounts = 20.0 * (1 - np.exp(-time_points / 1800.0)) * np.exp(-time_points / 7200.0)
        region.add_compartment("Epithelium", epi_amounts, volume_ml=0.5, fu=0.8)
        
        # Tissue compartment
        tissue_amounts = 10.0 * (1 - np.exp(-time_points / 3600.0)) * np.exp(-time_points / 10800.0)
        region.add_compartment("Tissue", tissue_amounts, volume_ml=2.0, fu=0.6)
    
    # Add systemic compartments
    plasma_amounts = 5.0 * (1 - np.exp(-time_points / 7200.0)) * np.exp(-time_points / 14400.0)
    subject.add_systemic_compartment("Plasma", plasma_amounts, volume_L=50.0, fu=0.9)
    
    # Add metrics
    subject.add_metrics({
        "Cmax_ng_per_ml": 2.5,
        "Tmax_h": 1.2,
        "AUC_ng_h_per_ml": 15.6,
        "Lung_retention_fraction": 0.85
    })
    
    return subject


# Example usage and demonstration
if __name__ == "__main__":
    # Create example data
    subject = create_example_subject_data()
    
    print("=== Intuitive Access Patterns ===")

    # Access lung region epithelium unbound concentration (default unit)
    print("subject.Al.Epithelium.Concentration.Unbound:")
    print(f"  Max: {np.max(subject.Al.Epithelium.Concentration.Unbound):.3f}")
    print(f"  Final: {subject.Al.Epithelium.Concentration.Unbound[-1]:.3f}")
    
    # Access tissue total amount  
    print("\nsubject.P2.Tissue.Amount.Total:")
    print(f"  Max: {np.max(subject.P2.Tissue.Amount.Total):.3f}")
    print(f"  Final: {subject.P2.Tissue.Amount.Total[-1]:.3f}")
    
    # Access plasma concentration
    print("\nsubject.Plasma.Concentration.Total:")
    print(f"  Max: {np.max(subject.Plasma.Concentration.Total):.3f}")
    print(f"  Final: {subject.Plasma.Concentration.Total[-1]:.3f}")
    
    # Access metrics
    print("\nsubject.Metrics:")
    for key, value in subject.Metrics.items():
        print(f"  {key}: {value}")
    
    print("\n=== Data Structure Summary ===")
    print(subject.get_summary())
