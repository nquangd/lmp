"""Optional units and quantity wrappers for enhanced type safety."""

from __future__ import annotations
from typing import Union, Optional
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class Amount:
    """Drug amount with units."""
    
    value: float
    unit: str = "mg"
    
    def to_mg(self) -> float:
        """Convert to milligrams."""
        conversions = {
            "g": 1000.0,
            "mg": 1.0, 
            "μg": 0.001,
            "ug": 0.001,
            "ng": 1e-6
        }
        
        if self.unit not in conversions:
            raise ValueError(f"Unknown unit: {self.unit}")
            
        return self.value * conversions[self.unit]
    
    def to_moles(self, molecular_weight: float) -> float:
        """Convert to moles using molecular weight."""
        mg = self.to_mg()
        return mg / molecular_weight  # molecular_weight is in μg/μmol (= g/mol)
    
    def __str__(self) -> str:
        return f"{self.value} {self.unit}"


@dataclass(frozen=True)  
class Concentration:
    """Drug concentration with units."""
    
    value: float
    unit: str = "mg/L"
    
    def to_mg_l(self) -> float:
        """Convert to mg/L."""
        conversions = {
            "g/L": 1000.0,
            "mg/L": 1.0,
            "mg/mL": 1000.0,
            "μg/mL": 1.0,
            "ug/mL": 1.0, 
            "ng/mL": 0.001,
            "μM": None,  # Requires molecular weight
            "nM": None   # Requires molecular weight
        }
        
        if self.unit not in conversions:
            raise ValueError(f"Unknown unit: {self.unit}")
            
        factor = conversions[self.unit]
        if factor is None:
            raise ValueError(f"Unit {self.unit} requires molecular weight for conversion")
            
        return self.value * factor
    
    def to_molar(self, molecular_weight: float) -> float:
        """Convert to molar concentration.""" 
        if self.unit in ["μM", "uM"]:
            return self.value * 1e-6
        elif self.unit in ["nM"]:
            return self.value * 1e-9
        else:
            mg_l = self.to_mg_l()
            return mg_l / molecular_weight  # Convert to M
    
    def __str__(self) -> str:
        return f"{self.value} {self.unit}"


@dataclass(frozen=True)
class Volume:
    """Volume with units."""
    
    value: float
    unit: str = "L"
    
    def to_liters(self) -> float:
        """Convert to liters."""
        conversions = {
            "L": 1.0,
            "mL": 0.001,
            "μL": 1e-6,
            "uL": 1e-6
        }
        
        if self.unit not in conversions:
            raise ValueError(f"Unknown unit: {self.unit}")
            
        return self.value * conversions[self.unit]
    
    def to_ml(self) -> float:
        """Convert to milliliters."""
        return self.to_liters() * 1000.0
    
    def __str__(self) -> str:
        return f"{self.value} {self.unit}"


@dataclass(frozen=True)
class Time:
    """Time duration with units."""
    
    value: float
    unit: str = "h"
    
    def to_hours(self) -> float:
        """Convert to hours."""
        conversions = {
            "h": 1.0,
            "min": 1/60.0,
            "s": 1/3600.0,
            "d": 24.0
        }
        
        if self.unit not in conversions:
            raise ValueError(f"Unknown unit: {self.unit}")
            
        return self.value * conversions[self.unit]
    
    def to_seconds(self) -> float:
        """Convert to seconds."""
        return self.to_hours() * 3600.0
    
    def __str__(self) -> str:
        return f"{self.value} {self.unit}"


def convert_units(value: float, from_unit: str, to_unit: str, molecular_weight: Optional[float] = None) -> float:
    """Generic unit conversion function.
    
    Args:
        value: Numeric value to convert
        from_unit: Source unit 
        to_unit: Target unit
        molecular_weight: Molecular weight for mass/molar conversions
        
    Returns:
        Converted value
        
    Raises:
        ValueError: If conversion is not possible
    """
    # Mass conversions
    mass_conversions = {
        ("g", "mg"): 1000.0,
        ("mg", "g"): 0.001,
        ("mg", "μg"): 1000.0,
        ("μg", "mg"): 0.001,
        ("μg", "ng"): 1000.0,
        ("ng", "μg"): 0.001
    }
    
    # Volume conversions
    volume_conversions = {
        ("L", "mL"): 1000.0,
        ("mL", "L"): 0.001,
        ("mL", "μL"): 1000.0,
        ("μL", "mL"): 0.001
    }
    
    # Time conversions  
    time_conversions = {
        ("h", "min"): 60.0,
        ("min", "h"): 1/60.0,
        ("h", "s"): 3600.0,
        ("s", "h"): 1/3600.0,
        ("min", "s"): 60.0,
        ("s", "min"): 1/60.0
    }
    
    # Check direct conversions
    conversion_key = (from_unit, to_unit)
    
    for conversion_dict in [mass_conversions, volume_conversions, time_conversions]:
        if conversion_key in conversion_dict:
            return value * conversion_dict[conversion_key]
    
    # Check reverse conversions
    reverse_key = (to_unit, from_unit) 
    for conversion_dict in [mass_conversions, volume_conversions, time_conversions]:
        if reverse_key in conversion_dict:
            return value / conversion_dict[reverse_key]
    
    # Molar conversions (require molecular weight)
    if molecular_weight is not None:
        mw_g_mol = molecular_weight  # molecular_weight is already in μg/μmol (= g/mol)
        
        if from_unit == "mg" and to_unit == "mmol":
            return value / mw_g_mol
        elif from_unit == "mmol" and to_unit == "mg":
            return value * mw_g_mol
    
    raise ValueError(f"Cannot convert from {from_unit} to {to_unit}")