"""Builtin data loader for catalog data.

This module provides utilities to load various builtin data structures
from the catalog/builtin/ directory in a consistent way.
"""

from __future__ import annotations
from typing import Dict, Any, List, Optional
import copy
from pathlib import Path
import numpy as np
try:
    import tomllib
except ImportError:
    import tomli as tomllib


class BuiltinDataLoader:
    """Loader for builtin catalog data."""

    _variability_cache: Dict[tuple[str, str], Dict[str, Any]] = {}

    def __init__(self, catalog_root: Optional[Path] = None):
        """Initialize loader with catalog root path.
        
        Args:
            catalog_root: Root path to catalog directory. If None, uses relative path.
        """
        if catalog_root is None:
            # Default to builtin directory relative to this file
            self.catalog_root = Path(__file__).parent / "builtin"
        else:
            self.catalog_root = catalog_root

    def load_variability_file(self, category: str, filename: str) -> Dict[str, Any]:
        """Load raw variability data from the builtin catalog without fallback defaults.

        Args:
            category: Catalog subdirectory (e.g., 'inhalation', 'gi_tract').
            filename: File stem to load (with or without 'Variability_' prefix).

        Returns:
            Parsed TOML content as a dictionary.
        """
        toml_name = filename if filename.endswith('.toml') else f"{filename}.toml"
        key = (str(self.catalog_root), f"{category}/{toml_name}")
        if key in self._variability_cache:
            return copy.deepcopy(self._variability_cache[key])

        file_path = self.catalog_root / category / toml_name

        if not file_path.exists():
            raise FileNotFoundError(f"Variability file not found: {file_path}")

        with open(file_path, 'rb') as handle:
            data = tomllib.load(handle)

        self._variability_cache[key] = data
        return copy.deepcopy(data)
    
    def load_regional_physiology(
        self,
        name: str = "regional",
        population: str = "healthy"
    ) -> Dict[str, Any]:
        """Derive regional lung physiology parameters from generation data.

        Args:
            name: Label applied to the returned regional physiology bundle.
            population: Lung generation population key (default: "healthy").

        Returns:
            Dictionary containing regional physiology parameters derived from
            the generation-level reference table.
        """

        from ..domain.entities import LungGeneration  # Local import to avoid circular dependency

        generation_rows = self.load_lung_geometry(population)
        lung_generation = LungGeneration(
            name=f"geometry_{population}",
            description="Reference generation-level geometry",
            generations=generation_rows,
        )

        regional_params = lung_generation.compute_regional_parameters()
        regional_params['name'] = name
        regional_params.setdefault('description', f"Derived regional physiology for {population}")
        return regional_params
    
    def load_inhalation_defaults(self, name: str = "default") -> Dict[str, float]:
        """Load default inhalation maneuver parameters.
        
        Args:
            name: Name of inhalation profile (default: "default")
            
        Returns:
            Dictionary with inhalation parameters
        """
        # Matches builtin/inhalation/default.toml
        return {
            'name': name,
            'pifr_Lpm': 30.0,
            'rise_time_s': 0.4,
            'inhaled_volume_L': 2.0,
            'hold_time_s': 0.5,
            'breath_hold_time_s': 30.0,
            'exhalation_flow_Lpm': 30.0,
            'bolus_volume_ml': 200.0,
            'bolus_delay_s': 0.0
        }
    
    def load_gi_tract_defaults(self, name: str = "default") -> Dict[str, Any]:
        """Load GI tract physiology parameters.
        
        Args:
            name: Name of GI tract profile (default: "default")
            
        Returns:
            Dictionary with GI tract parameters
        """
        # Matches builtin/gi_tract/default.toml
        return {
            'name': name,
            'num_comp': 9,
            'gi_area': {
                'BD': [0.0, 0.0, 0.0, 0.0, 300.0, 300.0, 144.72, 280.02, 41.77],
                'GP': [0.0, 80.0, 400.0, 422.82, 126.07, 226.32, 144.72, 28.02, 41.77],
                'FF': [0.0, 0.0, 0.0, 0.0, 250.0, 150.0, 150.0, 28.02, 41.77]
            },
            'gi_tg': {
                'BD': [60.0, 600.0, 600.0, 600.0, 3000.0, 3600.0, 1044.0, 15084.0, 45252.0],
                'GP': [12600.0, 5400.0, 3348.0, 2664.0, 2088.0, 1512.0, 1044.0, 15084.0, 45252.0],
                'FF': [600.0, 600.0, 600.0, 600.0, 2088.0, 1512.0, 1044.0, 15084.0, 45252.0]
            },
            'gi_vol': {
                'BD': [46.56, 41.56, 154.2, 122.3, 94.29, 70.53, 49.8, 47.49, 50.33],
                'GP': [46.56, 41.56, 154.2, 122.3, 94.29, 70.53, 49.8, 47.49, 50.33],
                'FF': [46.56, 41.56, 154.2, 122.3, 94.29, 70.53, 49.8, 47.49, 50.33]
            }
        }
    
    def load_lung_geometry(self, population: str = "healthy") -> list[dict[str, Any]]:
        """Load lung generation geometry/physiology table.

        Args:
            population: Population type ("healthy", etc.)

        Returns:
            Sorted list of generation dictionaries.
        """
        lung_file = self.catalog_root / "lung_geometry" / f"{population}.toml"

        if not lung_file.exists():
            raise FileNotFoundError(f"Lung geometry file not found: {lung_file}")

        try:
            with open(lung_file, 'rb') as f:
                data = tomllib.load(f)

            if 'generations' not in data:
                raise ValueError(f"No lung geometry data found in {lung_file} (expected 'generations' key)")

            generations = data['generations']
            if not isinstance(generations, list) or not generations:
                raise ValueError(f"No generation data found in {lung_file}")

            generations_sorted = sorted(generations, key=lambda x: x.get('generation', 0))
            return generations_sorted

        except Exception as e:
            raise RuntimeError(f"Failed to load lung geometry from {lung_file}: {e}")
    
    def get_available_api_names(self) -> List[str]:
        """Get list of available API names from builtin data.
        
        Returns:
            List of API names (e.g., ['BD', 'GP', 'FF'])
        """
        # In production, this would scan builtin/api/ directory or catalog
        # For now, return the known API types from GI tract and PK data
        return ['BD', 'GP', 'FF']
    
    def get_available_populations(self) -> List[str]:
        """Get list of available population types.
        
        Returns:
            List of population names
        """
        # In production, would scan builtin/lung_geometry/ directory
        return ['Healthy', 'COPD_Opt_1', 'COPD_Opt_2', 'COPD_Opt_3', 'COPD_Opt_4', 'COPD_Opt_5']
    
    def load_api_parameters(self, api_name: str) -> Dict[str, Any]:
        """Load API parameters from builtin TOML file.
        
        Args:
            api_name: API name ('BD', 'GP', 'FF')
            
        Returns:
            Dictionary with API parameters
        """
        api_file = self.catalog_root / "api" / f"{api_name}.toml"
        
        if not api_file.exists():
            raise FileNotFoundError(f"API file not found: {api_file}")
            
        try:
            with open(api_file, 'rb') as f:
                data = tomllib.load(f)
                
            # CRITICAL FIX: Manually parse nested tables for robust loading
            # This ensures that peff, pscale, etc. are correctly loaded as dicts
            if 'peff' in data:
                data['peff'] = {k: v for k, v in data['peff'].items()}
            
            if 'pscale' in data:
                data['pscale'] = {
                    region: {k: v for k, v in table.items()}
                    for region, table in data['pscale'].items()
                }
                
            if 'pscale_para' in data:
                data['pscale_para'] = {k: v for k, v in data['pscale_para'].items()}

            if 'k_in' in data:
                data['k_in'] = {k: v for k, v in data['k_in'].items()}

            if 'k_out' in data:
                data['k_out'] = {k: v for k, v in data['k_out'].items()}
                
            if 'fraction_unbound' in data:
                data['fraction_unbound'] = {k: v for k, v in data['fraction_unbound'].items()}

            return data
        except Exception as e:
            raise RuntimeError(f"Failed to load API parameters from {api_file}: {e}")
    
    def load_product_parameters(self, product_name: str) -> Dict[str, Any]:
        """Load Product parameters from builtin TOML file.
        
        Args:
            product_name: Product name ('test_product', 'reference_product')
            
        Returns:
            Dictionary with product parameters including API-specific data
        """
        product_file = self.catalog_root / "product" / f"{product_name}.toml"
        
        if not product_file.exists():
            raise FileNotFoundError(f"Product file not found: {product_file}")
            
        try:
            with open(product_file, 'rb') as f:
                data = tomllib.load(f)
                
                # Convert apis array to dict format expected by Product class
                if 'apis' in data:
                    apis_dict = {}
                    for api_entry in data['apis']:
                        api_name = api_entry.pop('name')
                        apis_dict[api_name] = api_entry
                    data['apis'] = apis_dict
                    
                return data
        except Exception as e:
            raise RuntimeError(f"Failed to load product parameters from {product_file}: {e}")
    
    def load_inhalation_profile(self, name: str = "pMDI_variable_trapezoid") -> Dict[str, Any]:
        """Load inhalation profile parameters from builtin files.
        
        Args:
            name: Name of inhalation profile (default: "pMDI_variable_trapezoid")
            
        Returns:
            Dictionary with inhalation profile parameters
        """
        try:
            profile_path = self.catalog_root / "inhalation" / f"{name}.toml"
            with open(profile_path, 'rb') as f:
                profile_data = tomllib.load(f)
            return profile_data
        except (FileNotFoundError, Exception):
            # Fallback to default values if file not found
            return {
                'name': name,
                'pifr_Lpm': 30.0,
                'rise_time_s': 0.4,
                'inhaled_volume_L': 2.0,
                'hold_time_s': 0.5,
                'breath_hold_time_s': 30.0,
                'exhalation_flow_Lpm': 30.0,
                'bolus_volume_ml': 200.0,
                'bolus_delay_s': 0.0
            }
    
    def get_available_inhalation_profiles(self) -> List[str]:
        """Get list of available inhalation profile names.
        
        Returns:
            List of profile names (without .toml extension)
        """
        try:
            inhalation_dir = self.catalog_root / "inhalation"
            if inhalation_dir.exists():
                return [f.stem for f in inhalation_dir.glob("*.toml")]
            return []
        except Exception:
            return []
    
    def load_subject_physiology(self, name: str = "healthy_reference") -> Dict[str, Any]:
        """Load subject physiology parameters from builtin files.
        
        Args:
            name: Name of subject physiology profile
            
        Returns:
            Dictionary with subject physiology parameters
        """
        try:
            subject_path = self.catalog_root / "subject" / f"{name}.toml"
            with open(subject_path, 'rb') as f:
                subject_data = tomllib.load(f)
            return subject_data
        except (FileNotFoundError, Exception):
            # Fallback to default values
            return {
                'name': name,
                'frc_ml': 3300.0,
                'frc_ref_ml': 2999.6,
                'et_scale_factor': 1.26,
                'mt_size': 'medium',
                'enable_variability': False
            }
    
    def load_cfd_parameters(self, name: str = "mt_deposition_params") -> Dict[str, Any]:
        """Load CFD parameters from builtin files.
        
        Args:
            name: Name of CFD parameter file
            
        Returns:
            Dictionary with CFD parameters
        """
        try:
            cfd_path = self.catalog_root / "cfd" / f"{name}.toml"
            with open(cfd_path, 'rb') as f:
                cfd_data = tomllib.load(f)
            return cfd_data
        except (FileNotFoundError, Exception):
            # Fallback to hardcoded values
            return {
                'size_based_mt_deposition': {
                    'small_particle_threshold_um': 2.0,
                    'large_particle_threshold_um': 4.0,
                    'small_particle_mt_fraction': 0.30,
                    'medium_particle_mt_slope': 0.10,
                    'large_particle_mt_fraction': 0.70
                },
                'population_scaling': {
                    'Healthy': 1.26,
                    'COPD_Opt_1': 1.35,
                    'COPD_Opt_2': 1.40,
                    'COPD_Opt_3': 1.45,
                    'COPD_Opt_4': 1.50,
                    'COPD_Opt_5': 1.55
                }
            }
