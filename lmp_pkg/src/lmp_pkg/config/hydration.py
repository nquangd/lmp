"""Configuration hydration - resolve entity references from catalog."""

from __future__ import annotations
from typing import Dict, Any, Optional, Union
import warnings

from ..catalog import get_default_catalog, ReadOnlyCatalog
from ..domain import Subject, API, Product, InhalationManeuver
from ..contracts.errors import ConfigError
from .model import AppConfig, EntityRef


def hydrate_config(config: AppConfig, catalog: Optional[ReadOnlyCatalog] = None) -> Dict[str, Any]:
    """Hydrate configuration by resolving entity references from catalog.
    
    Takes an AppConfig with entity references and returns a dictionary
    with fully resolved entities, ready for simulation.
    
    Args:
        config: Configuration with entity references
        catalog: Catalog to resolve from (uses default if None)
        
    Returns:
        Dictionary with resolved entities:
        {
            'subject': Subject instance,
            'api': API instance, 
            'product': Product instance,
            'maneuver': InhalationManeuver instance,
            'config': Original config dict
        }
        
    Raises:
        ConfigError: If entity resolution fails
    """
    if catalog is None:
        catalog = get_default_catalog()
    
    try:
        # Resolve each entity reference
        subject = _resolve_entity_ref(config.subject, "subject", catalog)
        api = _resolve_entity_ref(config.api, "api", catalog)
        product = _resolve_entity_ref(config.product, "product", catalog) 
        maneuver = _resolve_entity_ref(config.maneuver, "maneuver", catalog)
        
        return {
            "subject": subject,
            "api": api,
            "product": product, 
            "maneuver": maneuver,
            "config": config.model_dump()
        }
        
    except Exception as e:
        raise ConfigError(f"Failed to hydrate configuration: {e}")


def _resolve_entity_ref(
    entity_ref: EntityRef,
    category: str, 
    catalog: ReadOnlyCatalog
) -> Union[Subject, API, Product, InhalationManeuver]:
    """Resolve a single entity reference.
    
    Args:
        entity_ref: Entity reference with optional overrides
        category: Entity category name
        catalog: Catalog for resolution
        
    Returns:
        Resolved entity instance
        
    Raises:
        ConfigError: If resolution fails
    """
    # Get base entity from catalog
    if entity_ref.ref is None:
        raise ConfigError(f"No reference specified for {category}")
    
    try:
        base_entity = catalog.get_entry(category, entity_ref.ref)
    except ValueError as e:
        raise ConfigError(f"Failed to resolve {category}.{entity_ref.ref}: {e}")
    
    # Apply overrides if present
    if entity_ref.overrides:
        try:
            # Convert to dict, apply overrides, recreate entity
            entity_dict = base_entity.model_dump()
            entity_dict.update(entity_ref.overrides)
            
            # Validate and recreate entity
            entity_class = type(base_entity)
            return entity_class.model_validate(entity_dict)
            
        except Exception as e:
            raise ConfigError(f"Failed to apply overrides to {category}.{entity_ref.ref}: {e}")
    
    return base_entity


def validate_hydrated_entities(hydrated: Dict[str, Any]) -> None:
    """Validate consistency between hydrated entities.
    
    Performs cross-entity validation checks that can only be done
    after all entities are resolved.
    
    Args:
        hydrated: Dictionary from hydrate_config()
        
    Raises:
        ConfigError: If validation fails
    """
    subject = hydrated["subject"] 
    api = hydrated["api"]
    product = hydrated["product"]
    maneuver = hydrated["maneuver"]
    
    warnings_list = []
    
    # Check age-appropriate flow rates
    if subject.age_years < 12 and maneuver.peak_inspiratory_flow_l_min > 60:
        warnings_list.append(
            f"High flow rate ({maneuver.peak_inspiratory_flow_l_min} L/min) "
            f"for pediatric subject (age {subject.age_years})"
        )
    
    # Check device-maneuver compatibility
    device_type = product.device_type.lower()
    maneuver_type = maneuver.maneuver_type.lower()
    
    if device_type == "dpi" and "deep" not in maneuver_type:
        warnings_list.append(
            f"DPI devices typically require forceful inhalation, "
            f"got maneuver type: {maneuver.maneuver_type}"
        )
    
    if device_type == "pMDI".lower() and maneuver.peak_inspiratory_flow_l_min > 90:
        warnings_list.append(
            f"High flow rate ({maneuver.peak_inspiratory_flow_l_min} L/min) "
            f"may reduce pMDI efficiency"
        )
    
    # Check dose appropriateness
    label_claim = product.label_claim_mg
    if subject.age_years < 12 and label_claim > 0.2:
        warnings_list.append(
            f"High dose ({label_claim} mg) for pediatric subject"
        )
    
    # Check API-product compatibility
    if hasattr(product, 'therapeutic_class') and hasattr(api, 'therapeutic_class'):
        if product.therapeutic_class != api.therapeutic_class:
            warnings_list.append(
                f"Therapeutic class mismatch: "
                f"API={api.therapeutic_class}, Product={product.therapeutic_class}"
            )
    
    # Check coordination requirements
    if (hasattr(maneuver, 'coordination_efficiency') and 
        maneuver.coordination_efficiency is not None):
        
        coord_eff = maneuver.coordination_efficiency
        if device_type == "pmdi" and coord_eff < 0.7:
            warnings_list.append(
                f"Low coordination efficiency ({coord_eff:.2f}) "
                f"for pMDI device - consider spacer"
            )
    
    # Log warnings but don't fail
    for warning in warnings_list:
        warnings.warn(f"Entity validation warning: {warning}")


def get_entity_summary(hydrated: Dict[str, Any]) -> Dict[str, str]:
    """Generate human-readable summary of hydrated entities.
    
    Args:
        hydrated: Dictionary from hydrate_config()
        
    Returns:
        Dictionary with summary strings for each entity
    """
    subject = hydrated["subject"]
    api = hydrated["api"] 
    product = hydrated["product"]
    maneuver = hydrated["maneuver"]
    
    summaries = {
        "subject": f"{subject.name}: {subject.age_years}y, "
                  f"{subject.weight_kg}kg, {subject.sex}, "
                  f"BMI={subject.bmi_kg_m2:.1f}",
                  
        "api": f"{api.name}: MW={api.molecular_weight:.1f}μg/μmol",
        
        "product": f"{product.name}: {product.device_type}, "
                  f"{product.label_claim_mg}mg, "
                  f"MMAD={getattr(product, 'mass_median_diameter_um', 'N/A')}μm",
                  
        "maneuver": f"{maneuver.name}: {maneuver.peak_inspiratory_flow_l_min}L/min, "
                   f"{maneuver.inhaled_volume_ml}mL, "
                   f"{maneuver.inhalation_time_s}s"
    }
    
    # Add optional API properties
    if hasattr(api, 'clearance_l_h') and api.clearance_l_h:
        summaries["api"] += f", CL={api.clearance_l_h}L/h"
    
    # Add optional product properties  
    if hasattr(product, 'fine_particle_fraction') and product.fine_particle_fraction:
        summaries["product"] += f", FPF={product.fine_particle_fraction:.2f}"
    
    return summaries


def check_catalog_coverage(config: AppConfig, catalog: Optional[ReadOnlyCatalog] = None) -> Dict[str, bool]:
    """Check if all entity references can be resolved.
    
    Args:
        config: Configuration to check
        catalog: Catalog to check against
        
    Returns:
        Dictionary indicating availability of each entity reference
    """
    if catalog is None:
        catalog = get_default_catalog()
    
    entity_refs = {
        "subject": config.subject,
        "api": config.api,
        "product": config.product,
        "maneuver": config.maneuver
    }
    
    availability = {}
    
    for category, entity_ref in entity_refs.items():
        if entity_ref.ref is None:
            availability[category] = False
        else:
            availability[category] = catalog.has_entry(category, entity_ref.ref)
    
    return availability