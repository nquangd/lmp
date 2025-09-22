"""Apply variability to generate Inter and Intra subjects."""

from __future__ import annotations
from typing import Dict, Any, Optional, Tuple, Mapping
import numpy as np
import copy

from ..domain.entities import Subject
from ..domain.entities import API, Product, InhalationManeuver, VariabilitySettings
from .spec import VariabilitySpec, LayerSpec
from .factors import (
    generate_inhalation_factors,
    generate_pk_factors, 
    generate_physiology_values
)


def apply_population_variability_settings(
    variability_spec: VariabilitySpec,
    settings_overrides: Optional[Mapping[str, bool]],
) -> VariabilitySpec:
    """Return a variability spec filtered according to per-domain toggles."""

    if not settings_overrides:
        return variability_spec

    settings = VariabilitySettings.resolve(True, settings_overrides)
    filtered = variability_spec.model_copy(deep=True)

    if not settings.inhalation:
        filtered.inter.inhalation = {}
        filtered.intra.inhalation = {}

    if not settings.pk:
        filtered.inter.pk = {}
        filtered.intra.pk = {}

    if not settings.demographic:
        filtered.inter.physiology = {}
        filtered.intra.physiology = {}

    return filtered


def build_inter_subject(
    base_entities: Dict[str, Any],
    variability_spec: VariabilitySpec,
    rng: np.random.Generator,
    subject_id: Optional[str] = None
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Build Inter-subject variation from base entities.
    
    Args:
        base_entities: Dictionary with 'subject', 'api', 'product', 'maneuver' entities
        variability_spec: Complete variability specification
        rng: Random number generator for deterministic sampling
        subject_id: Optional subject identifier for logging
        
    Returns:
        Tuple of (modified_entities, inter_factors)
        - modified_entities: Entities with Inter variability applied
        - inter_factors: Generated factors for Intra layer use
    """
    if "inter" not in variability_spec.layers:
        return base_entities, {}
        
    inter_spec = variability_spec.inter
    modified_entities = copy.deepcopy(base_entities)
    
    # Generate factors for this Inter subject
    inhalation_factors = generate_inhalation_factors(inter_spec.inhalation, rng)
    pk_factors = generate_pk_factors(inter_spec.pk, rng)
    physiology_values = generate_physiology_values(inter_spec.physiology, rng)
    
    # Apply inhalation factors to maneuver
    if "maneuver" in modified_entities and inhalation_factors:
        maneuver = modified_entities["maneuver"]
        maneuver = _apply_inhalation_factors(maneuver, inhalation_factors)
        modified_entities["maneuver"] = maneuver
    
    # Apply physiology values to subject
    if "subject" in modified_entities and physiology_values:
        subject = modified_entities["subject"]
        subject = _apply_physiology_values(subject, physiology_values)
        modified_entities["subject"] = subject
    
    # Store factors for Intra layer
    inter_factors = {
        "inhalation": inhalation_factors,
        "pk": pk_factors,
        "physiology": physiology_values
    }
    
    return modified_entities, inter_factors


def build_intra_subject(
    inter_entities: Dict[str, Any],
    inter_factors: Dict[str, Any],
    variability_spec: VariabilitySpec,
    rng: np.random.Generator,
    replicate_id: Optional[int] = None
) -> Dict[str, Any]:
    """Build Intra-subject variation from Inter subject.
    
    Args:
        inter_entities: Entities with Inter variability already applied
        inter_factors: Factors from Inter layer generation
        variability_spec: Complete variability specification
        rng: Random number generator for deterministic sampling
        replicate_id: Optional replicate identifier for logging
        
    Returns:
        Entities with both Inter and Intra variability applied
    """
    if "intra" not in variability_spec.layers:
        return inter_entities
        
    intra_spec = variability_spec.intra
    modified_entities = copy.deepcopy(inter_entities)
    
    # Generate additional Intra factors
    intra_inhalation_factors = generate_inhalation_factors(intra_spec.inhalation, rng)
    intra_pk_factors = generate_pk_factors(intra_spec.pk, rng)
    intra_physiology_values = generate_physiology_values(intra_spec.physiology, rng)
    
    # Apply additional Intra inhalation factors to maneuver
    if "maneuver" in modified_entities and intra_inhalation_factors:
        maneuver = modified_entities["maneuver"]
        maneuver = _apply_inhalation_factors(maneuver, intra_inhalation_factors)
        modified_entities["maneuver"] = maneuver
    
    # Apply additional Intra physiology values to subject  
    if "subject" in modified_entities and intra_physiology_values:
        subject = modified_entities["subject"]
        subject = _apply_physiology_values(subject, intra_physiology_values)
        modified_entities["subject"] = subject
    
    # Store combined PK factors for later use
    combined_pk_factors = _combine_pk_factors(
        inter_factors.get("pk", {}),
        intra_pk_factors
    )
    
    # Add metadata for tracking
    if "metadata" not in modified_entities:
        modified_entities["metadata"] = {}
    modified_entities["metadata"]["pk_factors"] = combined_pk_factors
    modified_entities["metadata"]["replicate_id"] = replicate_id
    
    return modified_entities


def get_pk_scale(
    layer_factors: Dict[str, Dict[str, float]],
    rng: Optional[np.random.Generator] = None
) -> Dict[str, Dict[str, float]]:
    """Extract PK scaling factors in the format expected by models.
    
    Args:
        layer_factors: PK factors from Inter/Intra generation
        rng: Unused, for API compatibility
        
    Returns:
        Dictionary mapping param -> group -> factor
    """
    return copy.deepcopy(layer_factors)


def _apply_inhalation_factors(
    maneuver: InhalationManeuver,
    factors: Dict[str, float]
) -> InhalationManeuver:
    """Apply inhalation factors to maneuver parameters.
    
    Args:
        maneuver: Base inhalation profile
        factors: Multiplicative factors to apply
        
    Returns:
        Modified inhalation profile
    """
    # Create modifiable dictionary
    maneuver_dict = maneuver.model_dump()
    
    # Apply factors to matching parameters
    factor_mapping = {
        "pifr_Lpm": "pifr_Lpm",
        "rise_time_s": "rise_time_s",
        "hold_time_s": "hold_time_s",
        "breath_hold_time_s": "breath_hold_time_s",
        "exhalation_flow_Lpm": "exhalation_flow_Lpm",
        "bolus_volume_ml": "bolus_volume_ml",
        "bolus_delay_s": "bolus_delay_s",
    }
    
    for factor_name, factor_value in factors.items():
        if factor_name in factor_mapping:
            field_name = factor_mapping[factor_name]
            if field_name in maneuver_dict and maneuver_dict[field_name] is not None:
                maneuver_dict[field_name] = maneuver_dict[field_name] * factor_value
        
        # Special handling for ET (extrathoracic deposition) - stored as metadata
        elif factor_name == "ET":
            if "et_scaling_factor" not in maneuver_dict:
                maneuver_dict["et_scaling_factor"] = factor_value
    
    # Recreate InhalationProfile with modifications
    return InhalationManeuver.model_validate(maneuver_dict)


def _apply_physiology_values(
    subject: Subject,
    values: Dict[str, float]
) -> Subject:
    """Apply physiology values to subject parameters.
    
    Args:
        subject: Base subject
        values: Absolute values to apply
        
    Returns:
        Modified subject
    """
    # Create modifiable dictionary
    subject_dict = subject.model_dump()
    
    # Apply absolute values to matching parameters
    demographic_dict = subject_dict.get("demographic")
    if demographic_dict is not None:
        demographic_copy = dict(demographic_dict)
        if "FRC" in values:
            demographic_copy["frc_ml"] = values["FRC"]
        subject_dict["demographic"] = demographic_copy

    # Recreate Subject with modifications
    return Subject.model_validate(subject_dict)


def _combine_pk_factors(
    inter_factors: Dict[str, Dict[str, float]],
    intra_factors: Dict[str, Dict[str, float]]
) -> Dict[str, Dict[str, float]]:
    """Combine Inter and Intra PK factors multiplicatively.
    
    Args:
        inter_factors: Inter-subject PK factors
        intra_factors: Intra-subject PK factors
        
    Returns:
        Combined factors (Inter * Intra)
    """
    combined = copy.deepcopy(inter_factors)
    
    for param, groups in intra_factors.items():
        if param not in combined:
            combined[param] = groups.copy()
        else:
            for group, factor in groups.items():
                if group in combined[param]:
                    combined[param][group] *= factor
                else:
                    combined[param][group] = factor
                    
    return combined


def create_deterministic_rng(seed: int, subject_id: str, replicate_id: Optional[int] = None) -> np.random.Generator:
    """Create deterministic RNG based on seed and identifiers.
    
    Args:
        seed: Base random seed
        subject_id: Subject identifier
        replicate_id: Optional replicate identifier for Intra sampling
        
    Returns:
        Deterministic random number generator
    """
    # Create deterministic seed from components
    subject_hash = hash(subject_id) % (2**31)  # Keep within int32 range
    
    if replicate_id is not None:
        combined_seed = seed ^ subject_hash ^ replicate_id
    else:
        combined_seed = seed ^ subject_hash
    
    # Ensure positive seed
    combined_seed = abs(combined_seed) % (2**31)
    
    return np.random.default_rng(combined_seed)
