"""Variability engine for handling inter/intra subject variability.

This module provides a clean interface between builtin variability data
and the transformation system in physiology.py.

Architecture:
- VariabilityEngine loads data from builtin/variability/*.toml
- Provides unified interface for all variability parameters
- Supports multiple distribution types (normal, lognormal)
- Handles defaults when data is missing (scale_factor = 1.0)
"""

from __future__ import annotations
# Use json as a fallback for now since TOML may not be available
import json
from typing import Dict, Any, Optional, Literal, Union
from pathlib import Path
import numpy as np
from scipy import stats
from dataclasses import dataclass


@dataclass
class VariabilityParameter:
    """Single variability parameter specification."""
    name: str
    distribution_type: Literal['normal', 'lognormal']
    inter_value: float  # GCV or sigma depending on context
    intra_value: float  # GCV or sigma depending on context
    api_specific: Dict[str, Dict[str, float]] = None  # For PK parameters
    
    
class VariabilityEngine:
    """Engine for loading and managing variability data from builtin catalog."""
    
    def __init__(self, variability_name: str = "default"):
        """Initialize variability engine.
        
        Args:
            variability_name: Name of variability profile in builtin/variability/
        """
        self.name = variability_name
        self.data: Dict[str, Any] = {}
        self._load_builtin_data()
        
    def _load_builtin_data(self) -> None:
        """Load variability data from builtin catalog TOML files."""
        try:
            # Try to load from TOML file in catalog
            import toml
            from pathlib import Path
            
            # Get path to builtin variability TOML
            catalog_path = Path(__file__).parent.parent / 'catalog' / 'builtin' / 'variability'
            toml_file = catalog_path / f'{self.name}.toml'
            
            if toml_file.exists():
                with open(toml_file, 'r') as f:
                    self.data = toml.load(f)
            else:
                # Fallback to default structure
                self.data = self._get_default_structure()
                
        except ImportError:
            # toml not available, use default structure
            self.data = self._get_default_structure()
        except Exception:
            # Any other error, use default structure
            self.data = self._get_default_structure()
            
    def _get_default_structure(self) -> Dict[str, Any]:
        """Provide default structure when builtin data is not available."""
        return {
            'physiology': {
                'FRC': {
                    'type': 'normal',
                    'mean': 3300.0,
                    'inter_sigma': 600.0,
                    'intra_sigma': 0.0
                },
                'weight_kg': {
                    'type': 'normal', 
                    'mean': 70.0,
                    'inter_sigma': 10.0,
                    'intra_sigma': 0.0
                },
                'height_cm': {
                    'type': 'normal',
                    'mean': 175.0,
                    'inter_sigma': 8.0,
                    'intra_sigma': 0.0
                }
            },
            'inhalation': {
                'pifr_Lpm': {
                    'type': 'lognormal',
                    'inter_gcv': 0.3,
                    'intra_gcv': 0.3
                },
                'rise_time_s': {
                    'type': 'lognormal',
                    'inter_gcv': 0.1,
                    'intra_gcv': 0.1
                },
                'ET': {
                    'type': 'lognormal',
                    'inter_gcv': 0.235,
                    'intra_gcv': 0.15
                },
                'hold_time_s': {
                    'type': 'normal',
                    'inter_gcv': 0.0,
                    'intra_gcv': 0.0
                },
                'breath_hold_time_s': {
                    'type': 'normal',
                    'inter_gcv': 0.0,
                    'intra_gcv': 0.0
                },
                'exhalation_flow_Lpm': {
                    'type': 'normal',
                    'inter_gcv': 0.0,
                    'intra_gcv': 0.0
                },
                'bolus_volume_ml': {
                    'type': 'normal',
                    'inter_gcv': 0.0,
                    'intra_gcv': 0.0
                },
                'bolus_delay_s': {
                    'type': 'normal',
                    'inter_gcv': 0.0,
                    'intra_gcv': 0.0
                },
                'mmad': {
                    'type': 'normal',
                    'inter_gcv': 0.0,
                    'intra_gcv': 0.0
                },
                'gsd': {
                    'type': 'normal',
                    'inter_gcv': 0.0,
                    'intra_gcv': 0.0
                }
            },
            'pk': {
                'CL': {
                    'type': 'lognormal',
                    'inter': {
                        'BD': 0.26,
                        'GP': 0.5,
                        'FF': 0.3
                    },
                    'intra': {
                        'BD': 0.05,
                        'GP': 0.4,
                        'FF': 0.1
                    }
                },
                'Eh': {
                    'type': 'lognormal',
                    'inter': {
                        'BD': 0.114,
                        'GP': 0.5,
                        'FF': 0.4
                    },
                    'intra': {
                        'BD': 0.10,
                        'GP': 0.4,
                        'FF': 0.2
                    }
                }
            },
            'defaults': {
                'default_gcv': 0.0,
                'default_sigma': 0.0,
                'default_distribution': 'normal'
            }
        }
    
    def get_physiology_parameter(self, param_name: str) -> VariabilityParameter:
        """Get physiology parameter variability (e.g., FRC, weight_kg).
        
        Args:
            param_name: Parameter name (e.g., 'FRC', 'weight_kg')
            
        Returns:
            VariabilityParameter with distribution and sigma values
        """
        if 'physiology' in self.data and param_name in self.data['physiology']:
            param_data = self.data['physiology'][param_name]
            return VariabilityParameter(
                name=param_name,
                distribution_type=param_data.get('type', 'normal'),
                inter_value=param_data.get('inter_sigma', 0.0),
                intra_value=param_data.get('intra_sigma', 0.0)
            )
        else:
            # Return defaults
            return self._get_default_parameter(param_name)
    
    def get_inhalation_parameter(self, param_name: str) -> VariabilityParameter:
        """Get inhalation parameter variability (e.g., pifr_Lpm, rise_time_s).
        
        Args:
            param_name: Parameter name (e.g., 'pifr_Lpm', 'rise_time_s')
            
        Returns:
            VariabilityParameter with distribution and GCV values
        """
        if 'inhalation' in self.data and param_name in self.data['inhalation']:
            param_data = self.data['inhalation'][param_name]
            return VariabilityParameter(
                name=param_name,
                distribution_type=param_data.get('type', 'normal'),
                inter_value=param_data.get('inter_gcv', 0.0),
                intra_value=param_data.get('intra_gcv', 0.0)
            )
        else:
            return self._get_default_parameter(param_name)
    
    def get_pk_parameter(self, param_name: str) -> VariabilityParameter:
        """Get PK parameter variability (e.g., CL, Eh) with API-specific values.
        
        Args:
            param_name: Parameter name (e.g., 'CL', 'Eh')
            
        Returns:
            VariabilityParameter with API-specific GCV values
        """
        if 'pk' in self.data and param_name in self.data['pk']:
            param_data = self.data['pk'][param_name]
            api_specific = {
                'inter': param_data.get('inter', {}),
                'intra': param_data.get('intra', {})
            }
            return VariabilityParameter(
                name=param_name,
                distribution_type=param_data.get('type', 'lognormal'),
                inter_value=0.0,  # Not used for API-specific
                intra_value=0.0,  # Not used for API-specific
                api_specific=api_specific
            )
        else:
            return self._get_default_parameter(param_name)
    
    def _get_default_parameter(self, param_name: str) -> VariabilityParameter:
        """Get default parameter when no data is available."""
        defaults = self.data.get('defaults', {})
        return VariabilityParameter(
            name=param_name,
            distribution_type=defaults.get('default_distribution', 'normal'),
            inter_value=defaults.get('default_gcv', 0.0),
            intra_value=defaults.get('default_gcv', 0.0)
        )
    
    def sample_factor(
        self, 
        param: VariabilityParameter,
        variability_type: Literal['Inter', 'Intra'],
        api_name: Optional[str] = None
    ) -> float:
        """Sample a scaling factor from the parameter's distribution.
        
        Args:
            param: VariabilityParameter specification
            variability_type: 'Inter' or 'Intra' subject variability
            api_name: API name for PK parameters (e.g., 'BD', 'GP', 'FF')
            
        Returns:
            Scaling factor (1.0 if no variability)
        """
        # Handle API-specific parameters
        if param.api_specific and api_name:
            var_key = variability_type.lower()
            if var_key in param.api_specific and api_name in param.api_specific[var_key]:
                gcv = param.api_specific[var_key][api_name]
                if gcv == 0:
                    return 1.0
                sigma = self._convert_gcv_to_sigma_log(gcv)
                return np.random.lognormal(mean=0, sigma=sigma)
        
        # Handle regular parameters
        value = param.inter_value if variability_type == 'Inter' else param.intra_value
        if value == 0:
            return 1.0
            
        if param.distribution_type == 'lognormal':
            if param.name in ['FRC']:  # Physiology parameters use sigma directly
                sigma = value
            else:  # Inhalation parameters use GCV
                sigma = self._convert_gcv_to_sigma_log(value)
            return np.random.lognormal(mean=0, sigma=sigma)
        else:  # normal
            if param.name in ['FRC']:  # Physiology parameters
                return stats.norm.rvs(loc=1.0, scale=value)
            else:  # Inhalation parameters use GCV as scale
                return stats.norm.rvs(loc=1.0, scale=value)
    
    @staticmethod
    def _convert_gcv_to_sigma_log(gcv: float) -> float:
        """Convert Geometric CV to log-normal sigma.
        
        Args:
            gcv: Geometric coefficient of variation
            
        Returns:
            Sigma parameter for log-normal distribution
        """
        if gcv <= 0:
            return 0.0
        return np.sqrt(np.log(gcv**2 + 1))
    
    def get_all_parameter_names(self) -> Dict[str, list]:
        """Get all available parameter names by category.
        
        Returns:
            Dictionary with categories and parameter names
        """
        result = {
            'physiology': [],
            'inhalation': [], 
            'pk': []
        }
        
        if 'physiology' in self.data:
            result['physiology'] = list(self.data['physiology'].keys())
        if 'inhalation' in self.data:
            result['inhalation'] = list(self.data['inhalation'].keys())
        if 'pk' in self.data:
            result['pk'] = list(self.data['pk'].keys())
            
        return result