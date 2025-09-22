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
        subject_ref = config.subject
        population_ref = getattr(getattr(config, 'study', None), 'population', None)
        subject = None
        if population_ref:
            use_population = (subject_ref.ref is None or subject_ref.ref in {'healthy_reference', population_ref})
            try:
                if use_population:
                    base_entity = catalog.get_entry('subject', population_ref)
                    if subject_ref.overrides:
                        data = base_entity.model_dump()
                        data.update(subject_ref.overrides)
                        subject = type(base_entity).model_validate(data)
                    else:
                        subject = base_entity
            except ValueError:
                subject = None
        if subject is None:
            subject = _resolve_entity_ref(subject_ref, 'subject', catalog)

        api = _resolve_entity_ref(config.api, "api", catalog)
        product = _resolve_entity_ref(config.product, "product", catalog) 
        maneuver = _resolve_entity_ref(config.maneuver, "maneuver", catalog)

        # Ensure subject is a fully built instance with all subcomponents
        # If the catalog-provided subject lacks demographic or other parts, rebuild via Subject.from_builtin
        try:
            needs_build = not isinstance(subject, Subject) or (
                isinstance(subject, Subject) and (
                    subject.demographic is None or subject.lung_generation is None or subject.inhalation_maneuver is None
                )
            )
            if needs_build:
                demo_name = subject_ref.ref or getattr(subject, 'name', None) or 'healthy_reference'
                inh_profile = getattr(config.maneuver, 'ref', None) or getattr(maneuver, 'name', None) or 'pMDI_variable_trapezoid'
                api_name = getattr(config.api, 'ref', None) or getattr(api, 'name', None) or 'BD'
                subject_built = Subject.from_builtin(
                    subject_name=demo_name,
                    demographic_name=demo_name,
                    lung_geometry_population='healthy',
                    gi_name='default',
                    inhalation_profile=inh_profile,
                    api_name=api_name,
                )
                # Apply simple demographic overrides if provided
                if subject_ref.overrides and subject_built.demographic:
                    for k, v in subject_ref.overrides.items():
                        if hasattr(subject_built.demographic, k):
                            try:
                                setattr(subject_built.demographic, k, v)
                            except Exception:
                                pass
                subject = subject_built
        except Exception as e:
            raise ConfigError(f"Failed to construct subject from builtin: {e}")
        
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
    
    # Check age-appropriate flow rates (guard if demographic missing)
    age = getattr(getattr(subject, 'demographic', None), 'age_years', None)
    pifr = getattr(maneuver, 'pifr_Lpm', None)
    if isinstance(age, (int, float)) and isinstance(pifr, (int, float)):
        if age < 12 and pifr > 60:
            warnings_list.append(
                f"High flow rate ({pifr} L/min) for pediatric subject (age {age})"
            )
    
    # Check device-maneuver compatibility
    device_type = (getattr(product, 'device', '') or '').lower()
    maneuver_type = (getattr(maneuver, 'maneuver_type', '') or '').lower()
    
    if device_type == "dpi" and "deep" not in maneuver_type:
        warnings_list.append(
            f"DPI devices typically require forceful inhalation, got maneuver type: {getattr(maneuver, 'maneuver_type', 'unknown')}"
        )
    
    if device_type == "pmdi" and isinstance(pifr, (int, float)) and pifr > 90:
        warnings_list.append(
            f"High flow rate ({pifr} L/min) may reduce pMDI efficiency"
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
    
    demo = getattr(subject, 'demographic', None)
    name = getattr(subject, 'name', 'unknown')
    age = getattr(demo, 'age_years', 'NA')
    weight = getattr(demo, 'weight_kg', 'NA')
    sex = getattr(demo, 'sex', 'NA')
    bmi = getattr(demo, 'bmi_kg_m2', None)
    bmi_str = f"{bmi:.1f}" if isinstance(bmi, (int, float)) else "NA"

    summaries = {
        "subject": f"{name}: {age}y, {weight}kg, {sex}, BMI={bmi_str}",
        "api": f"{getattr(api, 'name', 'unknown')}: MW={getattr(api, 'molecular_weight', 0.0):.1f}μg/μmol",
        "product": f"{getattr(product, 'name', 'unknown')}: {getattr(product, 'device', 'N/A')}",
        "maneuver": f"{getattr(maneuver, 'name', 'unknown')}: {getattr(maneuver, 'pifr_Lpm', 'NA')}L/min, "
                     f"{getattr(maneuver, 'inhaled_volume_L', 'NA')}L"
    }
    
    # Add optional API properties
    if hasattr(api, 'clearance_l_h') and getattr(api, 'clearance_l_h'):
        summaries["api"] += f", CL={api.clearance_l_h}L/h"
    
    # Add optional product properties  
    if hasattr(product, 'fine_particle_fraction') and getattr(product, 'fine_particle_fraction'):
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
    
    # Check age-appropriate flow rates (guard if demographic missing)
    age = getattr(getattr(subject, 'demographic', None), 'age_years', None)
    pifr = getattr(maneuver, 'pifr_Lpm', None)
    if isinstance(age, (int, float)) and isinstance(pifr, (int, float)):
        if age < 12 and pifr > 60:
            warnings_list.append(
                f"High flow rate ({pifr} L/min) for pediatric subject (age {age})"
            )
    
    # Check device-maneuver compatibility
    device_type = (getattr(product, 'device', '') or '').lower()
    maneuver_type = (getattr(maneuver, 'maneuver_type', '') or '').lower()
    
    if device_type == "dpi" and "deep" not in maneuver_type:
        warnings_list.append(
            f"DPI devices typically require forceful inhalation, got maneuver type: {getattr(maneuver, 'maneuver_type', 'unknown')}"
        )
    
    if device_type == "pmdi" and isinstance(pifr, (int, float)) and pifr > 90:
        warnings_list.append(
            f"High flow rate ({pifr} L/min) may reduce pMDI efficiency"
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
    
    demo = getattr(subject, 'demographic', None)
    name = getattr(subject, 'name', 'unknown')
    age = getattr(demo, 'age_years', 'NA')
    weight = getattr(demo, 'weight_kg', 'NA')
    sex = getattr(demo, 'sex', 'NA')
    bmi = getattr(demo, 'bmi_kg_m2', None)
    bmi_str = f"{bmi:.1f}" if isinstance(bmi, (int, float)) else "NA"

    summaries = {
        "subject": f"{name}: {age}y, {weight}kg, {sex}, BMI={bmi_str}",
        "api": f"{getattr(api, 'name', 'unknown')}: MW={getattr(api, 'molecular_weight', 0.0):.1f}μg/μmol",
        "product": f"{getattr(product, 'name', 'unknown')}: {getattr(product, 'device', 'N/A')}",
        "maneuver": f"{getattr(maneuver, 'name', 'unknown')}: {getattr(maneuver, 'pifr_Lpm', 'NA')}L/min, "
                     f"{getattr(maneuver, 'inhaled_volume_L', 'NA')}L"
    }
    
    # Add optional API properties
    if hasattr(api, 'clearance_l_h') and getattr(api, 'clearance_l_h'):
        summaries["api"] += f", CL={api.clearance_l_h}L/h"
    
    # Add optional product properties  
    if hasattr(product, 'fine_particle_fraction') and getattr(product, 'fine_particle_fraction'):
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