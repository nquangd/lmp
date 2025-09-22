"""Factor generation functions for variability."""

from __future__ import annotations
import math
from typing import Dict, Any, Union
import numpy as np
from scipy import stats

from .spec import DistributionSpec


def convert_gcv_to_sigma_log(gcv: float) -> float:
    """Convert geometric coefficient of variation to lognormal sigma.
    
    This matches the original implementation exactly.
    
    Args:
        gcv: Geometric coefficient of variation
        
    Returns:
        Sigma parameter for lognormal distribution
    """
    if gcv <= 0:
        return 0.0
    return math.sqrt(math.log(gcv**2 + 1))


def sample_multiplicative_factor(spec: DistributionSpec, rng: np.random.Generator) -> float:
    """Sample a multiplicative factor from a distribution specification.
    
    Args:
        spec: Distribution specification
        rng: Random number generator
        
    Returns:
        Multiplicative factor (mean=1 for most distributions)
        
    Raises:
        ValueError: For unsupported distribution types in multiplicative mode
    """
    if spec.mode == "absolute":
        raise ValueError("Use sample_absolute_value for absolute mode distributions")
    
    if spec.dist == "lognormal":
        sigma_log = spec.get_effective_sigma_log()
        if sigma_log == 0.0:
            return 1.0
        return rng.lognormal(mean=0.0, sigma=sigma_log)
    
    elif spec.dist == "normal":
        if spec.sd is None or spec.sd == 0.0:
            return 1.0
        return rng.normal(loc=1.0, scale=spec.sd)
    
    elif spec.dist == "uniform":
        if spec.min is None or spec.max is None:
            return 1.0
        if spec.min == spec.max:
            return spec.min
        return rng.uniform(low=spec.min, high=spec.max)
    
    else:
        raise ValueError(f"Unsupported distribution for multiplicative sampling: {spec.dist}")


def sample_absolute_value(spec: DistributionSpec, rng: np.random.Generator) -> float:
    """Sample an absolute value from a distribution specification.
    
    Args:
        spec: Distribution specification
        rng: Random number generator
        
    Returns:
        Absolute sampled value
        
    Raises:
        ValueError: For unsupported distribution types in absolute mode
    """
    if spec.mode == "multiplicative":
        raise ValueError("Use sample_multiplicative_factor for multiplicative mode distributions")
    
    if spec.dist == "normal_absolute":
        if spec.mean is None or spec.sd is None:
            raise ValueError("Normal absolute requires mean and sd")
        if spec.sd == 0.0:
            return spec.mean
        value = rng.normal(loc=spec.mean, scale=spec.sd)
        if spec.sd > 0.0 and value == spec.mean:
            # Rare but possible when underlying standard normal draw is exactly zero.
            # Resample once to ensure a deviation when variability is enabled.
            value = rng.normal(loc=spec.mean, scale=spec.sd)
        return value
    
    elif spec.dist == "uniform":
        if spec.min is None or spec.max is None:
            raise ValueError("Uniform distribution requires min and max")
        if spec.min == spec.max:
            return spec.min
        return rng.uniform(low=spec.min, high=spec.max)
    
    else:
        raise ValueError(f"Unsupported distribution for absolute sampling: {spec.dist}")


def generate_inhalation_factors(
    layer_spec: Dict[str, DistributionSpec],
    rng: np.random.Generator
) -> Dict[str, float]:
    """Generate multiplicative factors for inhalation parameters.
    
    Args:
        layer_spec: Layer specification for inhalation parameters
        rng: Random number generator
        
    Returns:
        Dictionary mapping parameter names to multiplicative factors
    """
    factors = {}
    
    for param_name, spec in layer_spec.items():
        try:
            factors[param_name] = sample_multiplicative_factor(spec, rng)
        except ValueError:
            # Handle special case for absolute parameters
            if spec.mode == "absolute":
                factors[param_name] = sample_absolute_value(spec, rng)
            else:
                # Default to 1.0 for problematic specs
                factors[param_name] = 1.0
                
    return factors


def generate_pk_factors(
    layer_spec: Dict[str, Dict[str, DistributionSpec]],
    rng: np.random.Generator
) -> Dict[str, Dict[str, float]]:
    """Generate multiplicative factors for PK parameters.
    
    Args:
        layer_spec: Layer specification for PK parameters (param -> group -> spec)
        rng: Random number generator
        
    Returns:
        Nested dictionary: param -> group -> multiplicative factor
    """
    factors = {}
    
    for param_name, group_specs in layer_spec.items():
        factors[param_name] = {}
        for group_name, spec in group_specs.items():
            try:
                factors[param_name][group_name] = sample_multiplicative_factor(spec, rng)
            except ValueError:
                # Default to 1.0 for problematic specs
                factors[param_name][group_name] = 1.0
                
    return factors


def generate_physiology_values(
    layer_spec: Dict[str, DistributionSpec],
    rng: np.random.Generator
) -> Dict[str, float]:
    """Generate absolute values for physiology parameters.
    
    Args:
        layer_spec: Layer specification for physiology parameters
        rng: Random number generator
        
    Returns:
        Dictionary mapping parameter names to absolute values
    """
    values = {}
    
    for param_name, spec in layer_spec.items():
        try:
            if spec.mode == "absolute":
                values[param_name] = sample_absolute_value(spec, rng)
            else:
                # Multiplicative physiology parameters (rare)
                values[param_name] = sample_multiplicative_factor(spec, rng)
        except ValueError:
            # Use spec mean or default
            if spec.mean is not None:
                values[param_name] = spec.mean
            else:
                values[param_name] = 1.0
                
    return values
