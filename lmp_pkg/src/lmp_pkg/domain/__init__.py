"""Domain entities and business logic."""

# Import the complex Subject class from subject.py as SubjectComplex
# from .subject import Subject as SubjectComplex, SimulationResults, run_lung_pbbm  # Disabled - subject.py doesn't exist
# Import the new unified Subject class from entities.py
from .entities import Subject, Demographic, API, Product, LungRegional, LungGeneration, InhalationManeuver, GI, PK, EntityCollection

# Backwards compatibility alias
InhalationProfile = InhalationManeuver
from .physiology import (
    calculate_lung_volumes,
    calculate_clearance_rates,
    calculate_flow_profile,
    calculate_inhalation_maneuver_flow_profile,
    validate_physiological_consistency,
    predict_lung_volumes_from_demographics,
    calculate_metabolic_scaling,
    estimate_respiratory_parameters
)
# Variability system (still useful as standalone)
from .variability import (
    VariabilityEngine,
    VariabilityParameter
)
from .quantity import Amount, Concentration, Volume, Time, convert_units

__all__ = [
    # Unified Subject systems
    "Subject",           # New unified Subject class from entities.py
    # "SubjectComplex",    # Complex Subject class from subject.py - disabled
    # "SimulationResults", # disabled
    # "run_lung_pbbm",     # disabled
    
    # Other entities
    "Demographic",
    "API", 
    "Product",
    "LungRegional",
    "LungGeneration", 
    "InhalationManeuver",
    "InhalationProfile",
    "GI",
    "PK",
    "EntityCollection",
    
    # Physiology functions
    "calculate_lung_volumes",
    "calculate_clearance_rates",
    "calculate_flow_profile",
    "calculate_inhalation_maneuver_flow_profile",
    "validate_physiological_consistency",
    "predict_lung_volumes_from_demographics",
    "calculate_metabolic_scaling",
    "estimate_respiratory_parameters",
    
    # Variability system (standalone)
    "VariabilityEngine",
    "VariabilityParameter",
    
    # Quantities
    "Amount",
    "Concentration",
    "Volume", 
    "Time",
    "convert_units"
]
