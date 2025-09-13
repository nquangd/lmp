"""Intuitive hierarchical data structure for PBBM results.

Supports access patterns like: subject.Al.Epithelium.Concentration.Unbound
"""

from __future__ import annotations
from typing import Dict, Any, Optional, List, Union
import numpy as np
from dataclasses import dataclass


class ConcentrationData:
    """Container for concentration data in different forms."""
    
    def __init__(self, time_points: np.ndarray, amounts_pmol: np.ndarray, volume_ml: float, 
                 fu: float = 1.0, molecular_weight: float = 250.0):
        """Initialize concentration data.
        
        Args:
            time_points: Time points [s]
            amounts_pmol: Amount time series [pmol]
            volume_ml: Compartment volume [mL]
            fu: Fraction unbound [0-1]
            molecular_weight: Molecular weight [g/mol]
        """
        self.time_s = time_points
        self.time_h = time_points / 3600.0
        self.amounts_pmol = amounts_pmol
        self.volume_ml = volume_ml
        self.fu = fu
        self.mw = molecular_weight
        
        # Calculate concentrations
        self._calculate_concentrations()
    
    def _calculate_concentrations(self):
        """Calculate concentration time series in different units."""
        if self.volume_ml > 0:
            # Total concentration [pmol/mL]
            self.total_pmol_per_ml = self.amounts_pmol / self.volume_ml
            
            # Unbound concentration [pmol/mL] 
            self.unbound_pmol_per_ml = self.total_pmol_per_ml * self.fu
            
            # Convert to mass units
            self.total_pg_per_ml = self.total_pmol_per_ml * self.mw / 1e6
            self.unbound_pg_per_ml = self.unbound_pmol_per_ml * self.mw / 1e6
            
            self.total_ng_per_ml = self.total_pg_per_ml / 1000.0
            self.unbound_ng_per_ml = self.unbound_pg_per_ml / 1000.0
        else:
            # Zero concentration if no volume
            zeros = np.zeros_like(self.amounts_pmol)
            self.total_pmol_per_ml = zeros
            self.unbound_pmol_per_ml = zeros
            self.total_pg_per_ml = zeros
            self.unbound_pg_per_ml = zeros
            self.total_ng_per_ml = zeros
            self.unbound_ng_per_ml = zeros
    
    @property
    def Total(self) -> 'ConcentrationAccessor':
        """Access total concentrations."""
        return ConcentrationAccessor(
            pmol_per_ml=self.total_pmol_per_ml,
            pg_per_ml=self.total_pg_per_ml,
            ng_per_ml=self.total_ng_per_ml
        )
    
    @property
    def Unbound(self) -> 'ConcentrationAccessor':
        """Access unbound concentrations."""
        return ConcentrationAccessor(
            pmol_per_ml=self.unbound_pmol_per_ml,
            pg_per_ml=self.unbound_pg_per_ml,
            ng_per_ml=self.unbound_ng_per_ml
        )


class ConcentrationAccessor:
    """Accessor for concentration data in different units."""
    
    def __init__(self, pmol_per_ml: np.ndarray, pg_per_ml: np.ndarray, ng_per_ml: np.ndarray):
        self.pmol_per_ml = pmol_per_ml
        self.pg_per_ml = pg_per_ml  
        self.ng_per_ml = ng_per_ml
        
    def __repr__(self) -> str:
        if len(self.ng_per_ml) > 0:
            return f"Concentration(max={np.max(self.ng_per_ml):.3f} ng/mL, final={self.ng_per_ml[-1]:.3f} ng/mL)"
        return "Concentration(empty)"


class AmountData:
    """Container for amount data."""
    
    def __init__(self, time_points: np.ndarray, amounts_pmol: np.ndarray, molecular_weight: float = 250.0):
        """Initialize amount data.
        
        Args:
            time_points: Time points [s]
            amounts_pmol: Amount time series [pmol]
            molecular_weight: Molecular weight [g/mol]
        """
        self.time_s = time_points
        self.time_h = time_points / 3600.0
        self.pmol = amounts_pmol
        self.mw = molecular_weight
        
        # Convert to mass units: pmol × (μg/μmol) × (1e6 pg/μg) = pg
        self.pg = amounts_pmol * molecular_weight * 1e6
        self.ng = self.pg / 1000.0
        self.ug = self.ng / 1000.0
        
    def __repr__(self) -> str:
        if len(self.ng) > 0:
            return f"Amount(max={np.max(self.ng):.3f} ng, final={self.ng[-1]:.3f} ng)"
        return "Amount(empty)"


class CompartmentData:
    """Data for a single compartment (e.g., Epithelium, Tissue)."""
    
    def __init__(self, name: str, time_points: np.ndarray, amounts_pmol: np.ndarray,
                 volume_ml: float, fu: float = 1.0, molecular_weight: float = 250.0):
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
        
        # Amount and concentration data
        self.Amount = AmountData(time_points, amounts_pmol, molecular_weight)
        self.Concentration = ConcentrationData(time_points, amounts_pmol, volume_ml, fu, molecular_weight)
        
    def __repr__(self) -> str:
        return f"CompartmentData({self.name}, volume={self.volume_ml:.2f} mL)"


class RegionData:
    """Data for a lung region or generation (e.g., ET, TB, P1, P2, Al)."""
    
    def __init__(self, name: str, time_points: np.ndarray, molecular_weight: float = 250.0):
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
            compartment_name, self.time_points, amounts_pmol, volume_ml, fu, self.molecular_weight
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


class SubjectPBBMData:
    """Complete PBBM data for a single subject with intuitive access patterns.
    
    Supports access like: subject.Al.Epithelium.Concentration.Unbound.ng_per_ml
    """
    
    def __init__(self, subject_id: str, time_points: np.ndarray, molecular_weight: float = 250.0):
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
        region = RegionData(region_name, self.time_points, self.molecular_weight)
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
            compartment_name, self.time_points, amounts_pmol, volume_ml, fu, self.molecular_weight
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
    
    # Access lung region epithelium unbound concentration
    print("subject.Al.Epithelium.Concentration.Unbound.ng_per_ml:")
    print(f"  Max: {np.max(subject.Al.Epithelium.Concentration.Unbound.ng_per_ml):.3f} ng/mL")
    print(f"  Final: {subject.Al.Epithelium.Concentration.Unbound.ng_per_ml[-1]:.3f} ng/mL")
    
    # Access tissue total amount  
    print("\nsubject.P2.Tissue.Amount.ng:")
    print(f"  Max: {np.max(subject.P2.Tissue.Amount.ng):.3f} ng")
    print(f"  Final: {subject.P2.Tissue.Amount.ng[-1]:.3f} ng")
    
    # Access plasma concentration
    print("\nsubject.Plasma.Concentration.Total.ng_per_ml:")
    print(f"  Max: {np.max(subject.Plasma.Concentration.Total.ng_per_ml):.3f} ng/mL")
    print(f"  Final: {subject.Plasma.Concentration.Total.ng_per_ml[-1]:.3f} ng/mL")
    
    # Access metrics
    print("\nsubject.Metrics:")
    for key, value in subject.Metrics.items():
        print(f"  {key}: {value}")
    
    print("\n=== Data Structure Summary ===")
    print(subject.get_summary())