"""Domain entity schemas using Pydantic.

Note: Subject class has been moved to domain/subject.py as a unified class.
This file now contains only API, Product, InhalationManeuver, and other entities.
"""

from __future__ import annotations
from typing import Optional, Dict, Any, List, Tuple, TYPE_CHECKING
from pydantic import BaseModel, Field, field_validator, computed_field
import math
import numpy as np

if TYPE_CHECKING:
    from .subject import Subject

# Note: Complex Subject class is in domain/subject.py as unified Subject class
# This is a simpler Subject entity for loading builtin demographics


class Demographic(BaseModel):
    """Subject demographics and basic parameters loaded from builtin catalog."""
    
    # Basic identification
    name: str = Field(..., description="Subject name")
    subject_id: str = Field(..., description="Subject identifier")
    description: Optional[str] = Field(None, description="Subject description")
    
    # Demographics
    age_years: float = Field(..., ge=0, le=120, description="Age in years")
    weight_kg: float = Field(..., ge=1, le=300, description="Body weight in kg")
    height_cm: float = Field(..., ge=30, le=250, description="Height in cm")
    sex: str = Field(..., description="Biological sex", pattern="^(M|F|male|female)$")
    population: str = Field("Healthy", description="Population group")
    
    # Respiratory parameters
    frc_ml: Optional[float] = Field(None, description="Functional residual capacity in mL")
    frc_ref_ml: Optional[float] = Field(None, description="Reference FRC for scaling in mL")
    tidal_volume_ml: Optional[float] = Field(None, description="Tidal volume in mL")
    respiratory_rate_bpm: Optional[float] = Field(None, description="Respiratory rate in breaths per minute")
    
    # Deposition parameters
    et_scale_factor: Optional[float] = Field(None, description="ET deposition scaling factor")
    mt_size: Optional[str] = Field(None, description="Mouth-throat size category")
    
    # Configuration
    enable_variability: bool = Field(True, description="Enable population variability")
    
    # Internal state
    is_loaded: bool = Field(False, exclude=True, description="Whether parameters are loaded from builtin")
    
    @field_validator("sex")
    @classmethod
    def normalize_sex(cls, v: str) -> str:
        """Normalize sex to M/F format."""
        return {"male": "M", "female": "F"}.get(v.lower(), v.upper())
    
    @computed_field
    @property
    def bmi_kg_m2(self) -> float:
        """Body mass index in kg/m²."""
        height_m = self.height_cm / 100
        return self.weight_kg / (height_m ** 2)
    
    @computed_field  
    @property
    def bsa_m2(self) -> float:
        """Body surface area using DuBois formula in m²."""
        return 0.007184 * (self.weight_kg ** 0.425) * (self.height_cm ** 0.725)
    
    @classmethod
    def from_builtin(cls, subject_name: str) -> 'SubjectEntity':
        """Create SubjectEntity instance with parameters loaded from builtin catalog.
        
        Args:
            subject_name: Subject name ('healthy_reference', 'adult_70kg', etc.)
            
        Returns:
            Demographic instance with all builtin parameters loaded
        """
        from ..catalog.builtin_loader import BuiltinDataLoader
        
        loader = BuiltinDataLoader()
        params = loader.load_subject_physiology(subject_name)
        
        if not params:
            raise ValueError(f"Subject '{subject_name}' not found in builtin catalog")
            
        # Create instance with all parameters
        instance = cls(**params)
        instance.is_loaded = True
        return instance
    
    @classmethod
    def get_variability(cls, enable_variability: bool = True, subject_name: str = "healthy_reference") -> 'Demographic':
        """Create Demographic instance with variability parameters loaded from Variability_ builtin catalog.
        
        Args:
            enable_variability: Whether to enable variability (if False, sets first element to 0.0)
            subject_name: Subject name ('healthy_reference', 'adult_70kg', etc.)
            
        Returns:
            Demographic instance with variability parameters loaded (parameters hold lists)
        """
        from ..catalog.builtin_loader import BuiltinDataLoader
        
        loader = BuiltinDataLoader()
        # Always load from Variability_ prefixed file
        params = loader.load_subject_physiology(f"Variability_{subject_name}")
        
        if not params:
            raise ValueError(f"Variability subject 'Variability_{subject_name}' not found in builtin catalog")
            
        # If variability disabled, set first element of all lists to 0.0
        if not enable_variability:
            # Process all demographic parameters
            demographic_params = [
                'age_years', 'weight_kg', 'height_cm', 'frc_ml', 'frc_ref_ml', 
                'tidal_volume_ml', 'respiratory_rate_bpm', 'et_scale_factor', 'enable_variability'
            ]
            for param in demographic_params:
                if param in params and isinstance(params[param], list) and len(params[param]) > 0:
                    params[param][0] = 0.0
        
        # Create instance with variability parameters
        instance = cls(**params)
        instance.is_loaded = True
        return instance


class API(BaseModel):
    """Active Pharmaceutical Ingredient properties loaded from builtin catalog."""
    
    # Basic identification
    name: str = Field(..., description="API name")
    description: Optional[str] = Field(None, description="API description")
    
    # Molecular properties (from builtin data)
    molecular_weight: Optional[float] = Field(None, gt=0, description="Molecular weight in μg/μmol (= g/mol)")
    blood_plasma_ratio: Optional[float] = Field(None, description="Blood to plasma ratio")
    cell_binding: Optional[int] = Field(None, description="Cell binding flag")
    
    # PK model parameters (from builtin data)
    n_pk_compartments: Optional[int] = Field(None, description="Number of PK compartments")
    volume_central_L: Optional[float] = Field(None, ge=0, description="Central volume in L")
    clearance_L_h: Optional[float] = Field(None, ge=0, description="Total clearance in L/h")
    k12_h: Optional[float] = Field(None, description="Rate constant 1->2 (1/h)")
    k21_h: Optional[float] = Field(None, description="Rate constant 2->1 (1/h)")
    k13_h: Optional[float] = Field(None, description="Rate constant 1->3 (1/h)")
    k31_h: Optional[float] = Field(None, description="Rate constant 3->1 (1/h)")
    
    # GI absorption
    hepatic_extraction_pct: Optional[float] = Field(None, description="Hepatic extraction (%)")
    peff_para: Optional[float] = Field(None, description="Paracellular permeability")
    peff_GI: Optional[float] = Field(None, description="GI permeability")
    
    # Physical properties
    diffusion_coeff: Optional[float] = Field(None, description="Diffusion coefficient")
    density_g_m3: Optional[float] = Field(None, description="Density (g/m³)")
    solubility_pg_ml: Optional[float] = Field(None, description="Solubility (pg/mL)")
    
    # Complex parameters (stored as dicts)
    fraction_unbound: Optional[Dict[str, float]] = Field(None, description="Fraction unbound by compartment")
    peff: Optional[Dict[str, float]] = Field(None, description="Permeability parameters")
    pscale: Optional[Dict[str, Dict[str, float]]] = Field(None, description="Regional scaling factors")
    pscale_para: Optional[Dict[str, float]] = Field(None, description="Paracellular scaling by region")
    k_in: Optional[Dict[str, float]] = Field(None, description="Influx rate constants")
    k_out: Optional[Dict[str, float]] = Field(None, description="Efflux rate constants")
    
    # Internal state
    is_loaded: bool = Field(False, exclude=True, description="Whether parameters are loaded from builtin")
    
    @classmethod
    def from_builtin(cls, api_name: str) -> 'API':
        """Create API instance with parameters loaded from builtin catalog.
        
        Args:
            api_name: API name ('BD', 'GP', 'FF')
            
        Returns:
            API instance with all builtin parameters loaded
        """
        from ..catalog.builtin_loader import BuiltinDataLoader
        
        loader = BuiltinDataLoader()
        params = loader.load_api_parameters(api_name)
        
        if not params:
            raise ValueError(f"API '{api_name}' not found in builtin catalog")
            
        # Create instance with all parameters
        instance = cls(**params)
        instance.is_loaded = True
        return instance
    
    def get_final_values(self, dose_override: Optional[float] = None):
        """Get final API instance for deposition calculations.
        
        Args:
            dose_override: Optional dose override in pmol (if None, uses built-in molecular weight conversion)
            
        Returns:
            API instance with computed attributes (final_dose_pmol, etc.)
        """
        # Default dose calculation if not overridden (assumes dose is in pg and needs conversion to pmol)
        if dose_override is not None:
            final_dose_pmol = dose_override
        else:
            # If no dose available in API directly, return molecular weight for conversion
            final_dose_pmol = None
        
        # Store computed values as attributes on the API instance
        self._final_dose_pmol = final_dose_pmol
        
        return self


class Product(BaseModel):
    """Drug product with API-specific parameters loaded from builtin catalog."""
    
    # Basic identification
    name: str = Field(..., description="Product identifier")
    description: Optional[str] = Field(None, description="Product description")
    
    # Device properties (from builtin data)
    device: Optional[str] = Field(None, description="Device type (DFP, etc.)")
    propellant: Optional[str] = Field(None, description="Propellant type (PT210, PT010, etc.)")
    
    # API-specific parameters (from builtin data)
    apis: Optional[Dict[str, Dict[str, float]]] = Field(None, description="API-specific parameters")
    
    # Internal state
    is_loaded: bool = Field(False, exclude=True, description="Whether parameters are loaded from builtin")
    
    @classmethod
    def from_builtin(cls, product_name: str) -> 'Product':
        """Create Product instance with parameters loaded from builtin catalog.
        
        Args:
            product_name: Product name ('test_product', 'reference_product')
            
        Returns:
            Product instance with all builtin parameters loaded
        """
        from ..catalog.builtin_loader import BuiltinDataLoader
        
        loader = BuiltinDataLoader()
        params = loader.load_product_parameters(product_name)
        
        if not params:
            raise ValueError(f"Product '{product_name}' not found in builtin catalog")
            
        # Create instance with all parameters
        instance = cls(**params)
        instance.is_loaded = True
        return instance
    
    def get_api_parameters(self, api_name: str) -> Dict[str, float]:
        """Get API-specific parameters for this product.
        
        Args:
            api_name: API name ('BD', 'GP', 'FF')
            
        Returns:
            Dictionary with dose_pg, mmad, gsd, usp_depo_fraction
        """
        if not self.apis or api_name not in self.apis:
            return {}
        return self.apis[api_name]
    
    def get_dose_pg(self, api_name: str) -> float:
        """Get dose in picograms for specific API.
        
        Args:
            api_name: API name
            
        Returns:
            Dose in picograms
        """
        api_params = self.get_api_parameters(api_name)
        return api_params.get('dose_pg', 0.0)
    
    def get_final_values(self, api_name: str, dose_override_pg: Optional[float] = None):
        """Get final product instance for specific API.
        
        Args:
            api_name: API name ('BD', 'GP', 'FF')
            dose_override_pg: Optional dose override in picograms
            
        Returns:
            Product instance with computed attributes (final_dose_pg, final_api_name, etc.)
        """
        api_params = self.get_api_parameters(api_name)
        if not api_params:
            raise ValueError(f"API '{api_name}' not found in product '{self.name}'")
        
        # Use override dose if provided, otherwise use builtin dose
        final_dose_pg = dose_override_pg if dose_override_pg is not None else api_params.get('dose_pg', 0.0)
        
        # Store computed values as attributes on the product instance
        self._final_api_name = api_name
        self._final_dose_pg = final_dose_pg
        self._final_mmad = api_params.get('mmad', 3.5)
        self._final_gsd = api_params.get('gsd', 1.6)
        self._final_usp_depo_fraction = api_params.get('usp_depo_fraction', 40.0)
        
        return self


class LungRegional(BaseModel):
    """Regional lung physiology parameters loaded from builtin catalog."""
    
    # Basic identification
    name: str = Field(..., description="Regional model name")
    description: Optional[str] = Field(None, description="Model description")
    
    # Region definitions
    regions: List[str] = Field(..., description="List of region names")
    
    # Epithelial lining fluid area reference values (cm²)
    A_elf_ref: Dict[str, float] = Field(..., description="ELF area reference by region")
    
    # Extra area reference values (cm²)
    extra_area_ref: Dict[str, float] = Field(..., description="Extra area reference by region")
    
    # Epithelial lining fluid thickness (cm)
    d_elf: Dict[str, float] = Field(..., description="ELF thickness by region")
    
    # Epithelial thickness (cm)
    d_epi: Dict[str, float] = Field(..., description="Epithelial thickness by region")
    
    # Tissue volume (mL)
    V_tissue: Dict[str, float] = Field(..., description="Tissue volume by region")
    
    # Tissue blood flow (mL/min)
    Q_g: Dict[str, float] = Field(..., description="Blood flow by region")
    
    # Transit time (minutes)
    tg: Dict[str, float] = Field(..., description="Transit time by region")
    
    # Volume fraction
    V_frac_g: float = Field(..., description="Volume fraction")
    
    # Number of epithelial layers
    n_epi_layer: Dict[str, int] = Field(..., description="Number of epithelial layers by region")
    
    # Internal state
    is_loaded: bool = Field(False, exclude=True, description="Whether parameters are loaded from builtin")
    
    @classmethod
    def from_builtin(cls) -> 'LungRegional':
        """Create LungRegional instance with parameters loaded from builtin catalog.
        
        Returns:
            LungRegional instance with all builtin parameters loaded
        """
        from ..catalog.builtin_loader import BuiltinDataLoader
        
        loader = BuiltinDataLoader()
        params = loader.load_regional_physiology()
        
        if not params:
            raise ValueError("Regional lung physiology parameters not found in builtin catalog")
            
        # Create instance with all parameters
        instance = cls(**params)
        instance.is_loaded = True
        return instance
    
    @classmethod
    def get_variability(cls, enable_variability: bool = True) -> 'LungRegional':
        """Create LungRegional instance with variability parameters loaded from Variability_ builtin catalog.
        
        Args:
            enable_variability: Whether to enable variability (if False, sets first element to 0.0)
            
        Returns:
            LungRegional instance with variability parameters loaded (parameters hold lists)
        """
        from ..catalog.builtin_loader import BuiltinDataLoader
        
        loader = BuiltinDataLoader()
        # Always load from Variability_ prefixed file
        params = loader.load_regional_physiology("Variability_regional")
        
        if not params:
            raise ValueError("Variability regional lung physiology parameters not found in builtin catalog")
            
        # If variability disabled, set first element of all lists to 0.0
        if not enable_variability:
            # Process dictionaries with region-specific lists
            for param_group in ['A_elf_ref', 'extra_area_ref', 'd_elf', 'd_epi', 'V_tissue', 'Q_g', 'tg', 'n_epi_layer']:
                if param_group in params and isinstance(params[param_group], dict):
                    for region, value_list in params[param_group].items():
                        if isinstance(value_list, list) and len(value_list) > 0:
                            params[param_group][region][0] = 0.0
            
            # Process single parameters
            for param in ['V_frac_g']:
                if param in params and isinstance(params[param], list) and len(params[param]) > 0:
                    params[param][0] = 0.0
        
        # Create instance with variability parameters
        instance = cls(**params)
        instance.is_loaded = True
        return instance
    
    @classmethod
    def lung_scaling(cls, ref_lung: np.ndarray, v_lung: float, numgen: int = 25) -> np.ndarray:
        """Scale lung regional parameters based on lung volume.
        
        Based on scale_lung function from models/deposition/helper_functions.py line 35.
        Scales regional lung physiology parameters using volume-based scaling factor.
        
        Args:
            ref_lung: Reference lung physiology array
            v_lung: Target lung volume in microliters
            numgen: Number of generations (default 25)
            
        Returns:
            Scaled lung parameters array with shape (numgen, 8)
            Columns: [k_expansion_frac, multi, V_alveoli, Radius, Length, V_air, Angle_preceding, Angle_gravity]
        """
        import numpy as np
        
        # Convert volume to m³
        lung_size = v_lung / 1e6  # m³
        
        # Calculate scaling factor using reference volume 2999.60e-6 m³
        sf = (lung_size / 2999.60e-6)**(1/3)
        
        # Initialize output array
        sub_lung = np.zeros((numgen, 8))
        
        # Map reference lung columns to scaled output:
        # sub_lung columns: 0: k_expansion_frac, 1: multi, 2: V_alveoli, 3: Radius, 4: Length, 5: V_air, 6: Angle_preceding, 7: Angle_gravity
        # ref_lung columns assumed: 0: multi, 1: V_alveoli_ref, 2: Length_ref, 3: Diameter_ref, 4: Angle_gravity, 5: Angle_preceding, 6: k_expansion_frac
        
        sub_lung[:, 0] = ref_lung[:, 6] / 100  # Expansion fraction (convert from %)
        sub_lung[:, 1] = ref_lung[:, 0]        # Multiplicity
        sub_lung[:, 2] = 1e-6 * ref_lung[:, 1] * (sf**3)  # Alveolar volume (scaled by volume)
        sub_lung[:, 3] = 1e-2 * 0.5 * ref_lung[:, 3] * sf  # Radius (scaled by length, convert to m)
        sub_lung[:, 4] = 1e-2 * ref_lung[:, 2] * sf       # Length (scaled by length, convert to m)
        sub_lung[:, 5] = (sub_lung[:, 3]**2) * np.pi * sub_lung[:, 4] * sub_lung[:, 1] + sub_lung[:, 2]  # Total airway volume
        sub_lung[0, 5] = 50e-6  # Override first generation volume
        sub_lung[:, 6] = ref_lung[:, 5]        # Preceding angle
        sub_lung[:, 7] = ref_lung[:, 4]        # Gravity angle
        
        return sub_lung


class LungGeneration(BaseModel):
    """Generational lung geometry parameters loaded from builtin catalog."""
    
    # Basic identification
    name: str = Field(..., description="Geometry model name")
    description: Optional[str] = Field(None, description="Model description")
    
    # Lung geometry matrix (25 generations x 7 columns)
    # Columns: [N_airways, Length, Diameter, Area, Volume, FlowRate, Other]
    lung_geometry: List[List[float]] = Field(..., description="Lung geometry matrix")
    
    # Internal state
    is_loaded: bool = Field(False, exclude=True, description="Whether parameters are loaded from builtin")
    
    @classmethod
    def from_builtin(cls, population: str = "healthy") -> 'LungGeneration':
        """Create LungGeneration instance with parameters loaded from builtin catalog.
        
        Args:
            population: Population type ('healthy', etc.)
            
        Returns:
            LungGeneration instance with all builtin parameters loaded
        """
        from ..catalog.builtin_loader import BuiltinDataLoader
        
        loader = BuiltinDataLoader()
        lung_geometry_array = loader.load_lung_geometry(population)
        
        if lung_geometry_array is None:
            raise ValueError(f"Lung geometry parameters for '{population}' not found in builtin catalog")
            
        # Convert numpy array to the format expected by LungGeneration
        lung_geometry_list = lung_geometry_array.tolist()
        
        # Create instance with lung geometry data
        instance = cls(name=f"geometry_{population}", lung_geometry=lung_geometry_list)
        instance.is_loaded = True
        return instance
    
    @classmethod
    def get_variability(cls, enable_variability: bool = True, population: str = "healthy") -> 'LungGeneration':
        """Create LungGeneration instance with variability parameters loaded from Variability_ builtin catalog.
        
        Args:
            enable_variability: Whether to enable variability (if False, sets first element to 0.0)
            population: Population type ('healthy', etc.)
            
        Returns:
            LungGeneration instance with variability parameters loaded (parameters hold lists)
        """
        from ..catalog.builtin_loader import BuiltinDataLoader
        
        loader = BuiltinDataLoader()
        # Always load from Variability_ prefixed file
        params = loader.load_lung_geometry(f"Variability_{population}")
        
        if not params:
            raise ValueError(f"Variability lung geometry parameters for 'Variability_{population}' not found in builtin catalog")
            
        # If variability disabled, set first element of all nested lists to 0.0
        if not enable_variability:
            if 'lung_geometry' in params and isinstance(params['lung_geometry'], list):
                for generation_idx, generation in enumerate(params['lung_geometry']):
                    if isinstance(generation, list):
                        for column_idx, column_list in enumerate(generation):
                            if isinstance(column_list, list) and len(column_list) > 0:
                                params['lung_geometry'][generation_idx][column_idx][0] = 0.0
        
        # Create instance with variability parameters
        instance = cls(**params)
        instance.is_loaded = True
        return instance
    
    @classmethod
    def lung_scaling(cls, ref_lung: np.ndarray, v_lung: float, numgen: int = 25) -> np.ndarray:
        """Scale lung generation geometry based on lung volume.
        
        Based on scale_lung function from models/deposition/helper_functions.py line 35.
        Scales generational lung geometry parameters using volume-based scaling factor.
        
        Args:
            ref_lung: Reference lung geometry array
            v_lung: Target lung volume in microliters
            numgen: Number of generations (default 25)
            
        Returns:
            Scaled lung geometry array with shape (numgen, 8)
            Columns: [k_expansion_frac, multi, V_alveoli, Radius, Length, V_air, Angle_preceding, Angle_gravity]
        """
        import numpy as np
        
        # Convert volume to m³
        lung_size = v_lung / 1e6  # m³
        
        # Calculate scaling factor using reference volume 2999.60e-6 m³
        sf = (lung_size / 2999.60e-6)**(1/3)
        
        # Initialize output array
        sub_lung = np.zeros((numgen, 8))
        
        # Map reference lung columns to scaled output:
        # sub_lung columns: 0: k_expansion_frac, 1: multi, 2: V_alveoli, 3: Radius, 4: Length, 5: V_air, 6: Angle_preceding, 7: Angle_gravity
        # ref_lung columns assumed: 0: multi, 1: V_alveoli_ref, 2: Length_ref, 3: Diameter_ref, 4: Angle_gravity, 5: Angle_preceding, 6: k_expansion_frac
        
        sub_lung[:, 0] = ref_lung[:, 6] / 100  # Expansion fraction (convert from %)
        sub_lung[:, 1] = ref_lung[:, 0]        # Multiplicity
        sub_lung[:, 2] = 1e-6 * ref_lung[:, 1] * (sf**3)  # Alveolar volume (scaled by volume)
        sub_lung[:, 3] = 1e-2 * 0.5 * ref_lung[:, 3] * sf  # Radius (scaled by length, convert to m)
        sub_lung[:, 4] = 1e-2 * ref_lung[:, 2] * sf       # Length (scaled by length, convert to m)
        sub_lung[:, 5] = (sub_lung[:, 3]**2) * np.pi * sub_lung[:, 4] * sub_lung[:, 1] + sub_lung[:, 2]  # Total airway volume
        sub_lung[0, 5] = 50e-6  # Override first generation volume
        sub_lung[:, 6] = ref_lung[:, 5]        # Preceding angle
        sub_lung[:, 7] = ref_lung[:, 4]        # Gravity angle
        
        return sub_lung


class InhalationManeuver(BaseModel):
    """Inhalation maneuver characteristics loaded from builtin catalog."""
    
    # Basic identification
    name: str = Field(..., description="Profile identifier")
    description: Optional[str] = Field(None, description="Profile description")
    maneuver_type: str = Field(..., description="Type of inhalation maneuver")
    
    # Peak inspiratory flow rate (L/min)
    pifr_Lpm: float = Field(..., gt=0, description="Peak inspiratory flow rate in L/min")
    
    # Rise time to peak flow (s)
    rise_time_s: float = Field(..., gt=0, description="Rise time to peak flow in seconds")
    
    # Inhaled volume (L)
    inhaled_volume_L: float = Field(..., gt=0, description="Inhaled volume in L")
    
    # Hold time at peak flow (s)
    hold_time_s: float = Field(..., ge=0, description="Hold time at peak flow in seconds")
    
    # Breath hold time after inhalation (s)
    breath_hold_time_s: float = Field(..., ge=0, description="Breath hold time after inhalation in seconds")
    
    # Exhalation flow rate (L/min)
    exhalation_flow_Lpm: float = Field(..., gt=0, description="Exhalation flow rate in L/min")
    
    # Bolus volume (mL)
    bolus_volume_ml: float = Field(..., ge=0, description="Bolus volume in mL")
    
    # Bolus delay (s)
    bolus_delay_s: float = Field(..., ge=0, description="Bolus delay in seconds")
    
    # ET deposition scaling factor
    et_scale_factor: float = Field(1.26, gt=0, description="Extrathoracic deposition scaling factor")
    
    # Internal state
    is_loaded: bool = Field(False, exclude=True, description="Whether parameters are loaded from builtin")
    
    @classmethod
    def from_builtin(cls, profile_name: str = "pMDI_variable_trapezoid") -> 'InhalationManeuver':
        """Create InhalationManeuver instance with parameters loaded from builtin catalog.
        
        Args:
            profile_name: Profile name ('pMDI_variable_trapezoid', etc.)
            
        Returns:
            InhalationManeuver instance with all builtin parameters loaded
        """
        from ..catalog.builtin_loader import BuiltinDataLoader
        
        loader = BuiltinDataLoader()
        params = loader.load_inhalation_profile(profile_name)
        
        if not params:
            raise ValueError(f"Inhalation profile '{profile_name}' not found in builtin catalog")
            
        # Create instance with all parameters
        instance = cls(**params)
        instance.is_loaded = True
        return instance
    
    @classmethod
    def get_variability(cls, enable_variability: bool = True, profile_name: str = "pMDI_variable_trapezoid") -> 'InhalationManeuver':
        """Create InhalationManeuver instance with variability parameters loaded from Variability_ builtin catalog.
        
        Args:
            enable_variability: Whether to enable variability (if False, sets first element to 0.0)
            profile_name: Profile name ('pMDI_variable_trapezoid', etc.)
            
        Returns:
            InhalationManeuver instance with variability parameters loaded (parameters hold lists)
        """
        from ..catalog.builtin_loader import BuiltinDataLoader
        
        loader = BuiltinDataLoader()
        # Always load from Variability_ prefixed file
        params = loader.load_inhalation_profile(f"Variability_{profile_name}")
        
        if not params:
            raise ValueError(f"Variability inhalation profile 'Variability_{profile_name}' not found in builtin catalog")
            
        # If variability disabled, set first element of all lists to 0.0
        if not enable_variability:
            # Process all inhalation parameters
            inhalation_params = [
                'pifr_Lpm', 'rise_time_s', 'inhaled_volume_L', 'hold_time_s', 
                'breath_hold_time_s', 'exhalation_flow_Lpm', 'bolus_volume_ml', 'bolus_delay_s'
            ]
            for param in inhalation_params:
                if param in params and isinstance(params[param], list) and len(params[param]) > 0:
                    params[param][0] = 0.0
        
        # Create instance with variability parameters
        instance = cls(**params)
        instance.is_loaded = True
        return instance
    
    def calculate_inhalation_maneuver_flow_profile(self) -> np.ndarray:
        """Calculate inhalation flow profile using the original InhalationManeuver logic.
        
        Based on original lmp_apps/population/attributes.py InhalationManeuver.inhale_profile()
        Uses constants from config.constants for N_STEPS.
        
        Returns:
            Flow profile as [time_points, flow_rates] array with shape (N_STEPS, 2)
            Flow rates in L/min
        """
        from ..config.constants import N_STEPS
        
        # Convert to L/s for calculations
        pifr_ls = self.pifr_Lpm / 60.0
        
        # Calculate hold time
        if pifr_ls <= 0 or self.rise_time_s <= 0:
            hold_time = self.inhaled_volume_L / 1e-6 if pifr_ls <= 0 else self.inhaled_volume_L / pifr_ls
        else:
            hold_time = (self.inhaled_volume_L - 2 * (0.5 * pifr_ls * self.rise_time_s)) / pifr_ls
        
        if hold_time < 0:
            hold_time = 0
            
        # Total inhalation duration
        inhaled_duration = hold_time + 2 * self.rise_time_s
        if inhaled_duration <= 0:
            inhaled_duration = 1e-6  # Avoid division by zero
            
        # Calculate slope for ramp phases
        slope = pifr_ls / self.rise_time_s if self.rise_time_s > 0 else 0
        
        # Generate time points using N_STEPS from constants
        time_points = np.linspace(0, inhaled_duration, N_STEPS)
        flowrate = np.zeros_like(time_points)
        
        # Ramp up phase
        mask1 = time_points < self.rise_time_s
        flowrate[mask1] = time_points[mask1] * slope
        
        # Hold phase  
        mask2 = (time_points >= self.rise_time_s) & (time_points < (self.rise_time_s + hold_time))
        flowrate[mask2] = pifr_ls
        
        # Ramp down phase
        mask3 = time_points >= (self.rise_time_s + hold_time)
        flowrate[mask3] = pifr_ls - (time_points[mask3] - (self.rise_time_s + hold_time)) * slope
        
        # Ensure no negative flow
        flowrate = np.maximum(0, flowrate)
        
        # Format as [time, flow_rate] array and convert back to L/min
        flow_profile = np.zeros((len(flowrate), 2))
        flow_profile[:, 0] = time_points
        flow_profile[:, 1] = flowrate * 60.0  # Back to L/min
        
        return flow_profile


class GI(BaseModel):
    """GI tract physiology parameters loaded from builtin catalog."""
    
    # Basic identification
    name: str = Field(..., description="GI tract model name")
    description: Optional[str] = Field(None, description="Model description")
    
    # Number of GI compartments
    num_comp: int = Field(..., description="Number of GI compartments")
    
    # GI parameters by API type
    gi_area: Dict[str, List[float]] = Field(..., description="GI area parameters (cm²) by API")
    gi_tg: Dict[str, List[float]] = Field(..., description="GI transit time parameters (min) by API")
    gi_vol: Dict[str, List[float]] = Field(..., description="GI volume parameters (mL) by API")
    
    # Internal state
    is_loaded: bool = Field(False, exclude=True, description="Whether parameters are loaded from builtin")
    
    @classmethod
    def from_builtin(cls, gi_name: str = "default") -> 'GI':
        """Create GI instance with parameters loaded from builtin catalog.
        
        Args:
            gi_name: GI tract model name ('default', etc.)
            
        Returns:
            GI instance with all builtin parameters loaded
        """
        from ..catalog.builtin_loader import BuiltinDataLoader
        
        loader = BuiltinDataLoader()
        params = loader.load_gi_tract_defaults(gi_name)
        
        if not params:
            raise ValueError(f"GI tract parameters '{gi_name}' not found in builtin catalog")
            
        # Create instance with all parameters
        instance = cls(**params)
        instance.is_loaded = True
        return instance
    
    def get_api_parameters(self, api_name: str) -> Dict[str, List[float]]:
        """Get GI parameters for specific API.
        
        Args:
            api_name: API name ('BD', 'GP', 'FF')
            
        Returns:
            Dictionary with area, transit time, and volume lists for the API
        """
        return {
            'gi_area': self.gi_area.get(api_name, []),
            'gi_tg': self.gi_tg.get(api_name, []),
            'gi_vol': self.gi_vol.get(api_name, [])
        }
    
    @classmethod
    def get_variability(cls, enable_variability: bool = True, gi_name: str = "default", seed: Optional[int] = None) -> 'GI':
        """Create GI instance with variability multiplicative factors computed from Variability_ builtin catalog.
        
        Args:
            enable_variability: Whether to enable variability (if False, returns factors of 1.0)
            gi_name: GI tract model name ('default', etc.)
            seed: Random seed for reproducibility
            
        Returns:
            GI instance where parameters hold [inter_factor, intra_factor, dependent_var] lists
        """
        import numpy as np
        from ..catalog.builtin_loader import BuiltinDataLoader
        
        if seed is not None:
            np.random.seed(seed)
        
        loader = BuiltinDataLoader()
        # Always load from Variability_ prefixed file
        var_params = loader.load_gi_tract_defaults(f"Variability_{gi_name}")
        
        if not var_params:
            raise ValueError(f"Variability GI tract parameters 'Variability_{gi_name}' not found in builtin catalog")
        
        # Process parameters to compute multiplicative factors
        processed_params = {}
        
        for key, value in var_params.items():
            if key in ['name', 'description']:
                processed_params[key] = value  # Keep as-is for string fields
                
            elif key == 'num_comp':
                # Single parameter: [inter_sigma, inter_dist, intra_sigma, intra_dist, dependent]
                if isinstance(value, list) and len(value) >= 5:
                    inter_sigma, inter_dist, intra_sigma, intra_dist, dependent = value[:5]
                    
                    if not enable_variability:
                        inter_sigma = 0.0
                        intra_sigma = 0.0
                    
                    # Compute multiplicative factors
                    inter_factor = cls._compute_factor(inter_sigma, inter_dist)
                    intra_factor = cls._compute_factor(intra_sigma, intra_dist)
                    
                    processed_params[key] = [inter_factor, intra_factor, dependent]
                else:
                    processed_params[key] = value
                    
            elif key in ['gi_area', 'gi_tg', 'gi_vol']:
                # API-dependent parameters
                processed_params[key] = {}
                if isinstance(value, dict):
                    for api_name, compartment_list in value.items():
                        processed_compartments = []
                        if isinstance(compartment_list, list):
                            for comp_var in compartment_list:
                                if isinstance(comp_var, list) and len(comp_var) >= 5:
                                    inter_sigma, inter_dist, intra_sigma, intra_dist, dependent = comp_var[:5]
                                    
                                    if not enable_variability:
                                        inter_sigma = 0.0
                                        intra_sigma = 0.0
                                    
                                    # Compute multiplicative factors
                                    inter_factor = cls._compute_factor(inter_sigma, inter_dist)
                                    intra_factor = cls._compute_factor(intra_sigma, intra_dist)
                                    
                                    processed_compartments.append([inter_factor, intra_factor, dependent])
                                else:
                                    processed_compartments.append(comp_var)
                        processed_params[key][api_name] = processed_compartments
            else:
                processed_params[key] = value
        
        # Create instance with processed variability factors
        instance = cls(**processed_params)
        instance.is_loaded = True
        return instance
    
    @staticmethod
    def _compute_factor(sigma: float, distribution: str) -> float:
        """Compute multiplicative factor based on distribution type and sigma.
        
        Args:
            sigma: Standard deviation parameter
            distribution: Distribution type ('lognormal' or 'normal')
            
        Returns:
            Multiplicative factor
        """
        import numpy as np
        
        if sigma == 0.0:
            return 1.0
            
        if distribution == 'lognormal':
            # For lognormal: mean=0 in log space, returns multiplicative factor
            return np.random.lognormal(mean=0, sigma=sigma)
        elif distribution == 'normal':
            # For normal: mean=1, returns additive factor around 1
            return np.random.normal(loc=1.0, scale=sigma)
        else:
            return 1.0


class PK(BaseModel):
    """Pharmacokinetic parameters loaded from builtin catalog."""
    
    # Basic identification
    name: str = Field(..., description="PK model name")
    description: Optional[str] = Field(None, description="Model description")
    
    # Core PK parameters (loaded from API toml files)
    n_pk_compartments: int = Field(3, description="Number of PK compartments")
    clearance_L_h: float = Field(..., description="Systemic clearance (L/h)")
    volume_central_L: float = Field(..., description="Central volume of distribution (L)")
    hepatic_extraction: float = Field(..., description="Hepatic extraction ratio (fraction)")
    
    # Multi-compartment rate constants (1/h)
    k12_h: Optional[float] = Field(None, description="Rate constant central to peripheral 1 (1/h)")
    k21_h: Optional[float] = Field(None, description="Rate constant peripheral 1 to central (1/h)")
    k13_h: Optional[float] = Field(None, description="Rate constant central to peripheral 2 (1/h)")
    k31_h: Optional[float] = Field(None, description="Rate constant peripheral 2 to central (1/h)")
    
    # Additional compartment volumes (calculated or specified)
    volume_peripheral1_L: Optional[float] = Field(None, description="Peripheral 1 volume (L)")
    volume_peripheral2_L: Optional[float] = Field(None, description="Peripheral 2 volume (L)")
    
    # Distribution clearances (calculated from rate constants and volumes)
    cl_distribution1_L_h: Optional[float] = Field(None, description="Distribution clearance 1 (L/h)")
    cl_distribution2_L_h: Optional[float] = Field(None, description="Distribution clearance 2 (L/h)")
    
    # Absorption parameters
    ka_h: Optional[float] = Field(None, description="Absorption rate constant (1/h)")
    f_bioavail: float = Field(1.0, description="Bioavailability fraction")
    
    # Internal state
    is_loaded: bool = Field(False, exclude=True, description="Whether parameters are loaded from builtin")
    
    @classmethod
    def from_builtin(cls, api_name: str = "BD") -> 'PK':
        """Create PK instance with PK parameters loaded from API builtin catalog.
        
        Args:
            api_name: API name ('BD', 'GP', 'FF') to load PK parameters from
            
        Returns:
            PK instance with PK parameters loaded from API catalog
        """
        # Load API instance from builtin catalog to get PK-specific parameters
        api_instance = API.from_builtin(api_name)
        
        # Extract ONLY PK-relevant parameters from API instance
        pk_params = {
            "name": f"pk_{api_name}",
            "description": f"PK parameters for {api_name}",
            
            # Core PK parameters from toml
            "n_pk_compartments": getattr(api_instance, 'n_pk_compartments', 3),
            "clearance_L_h": getattr(api_instance, 'clearance_L_h', 35.0),
            "volume_central_L": getattr(api_instance, 'volume_central_L', 5.0),
            "hepatic_extraction": getattr(api_instance, 'hepatic_extraction_pct', 87.0) / 100.0,  # Convert % to fraction
            
            # Multi-compartment rate constants from toml
            "k12_h": getattr(api_instance, 'k12_h', None),
            "k21_h": getattr(api_instance, 'k21_h', None),
            "k13_h": getattr(api_instance, 'k13_h', None),  
            "k31_h": getattr(api_instance, 'k31_h', None),
            
            # Absorption parameters
            "ka_h": getattr(api_instance, 'ka_h', None),
            "f_bioavail": getattr(api_instance, 'f_bioavail', 1.0)
        }
        
        # Calculate derived parameters if rate constants are available
        instance = cls(**pk_params)
        instance._calculate_derived_parameters()
        instance.is_loaded = True
        return instance
    
    def _calculate_derived_parameters(self):
        """Calculate derived PK parameters from rate constants."""
        if self.k12_h and self.k21_h and self.volume_central_L:
            # Calculate peripheral volume 1 from rate constants
            self.cl_distribution1_L_h = self.k12_h * self.volume_central_L
            self.volume_peripheral1_L = self.cl_distribution1_L_h / self.k21_h
            
        if self.k13_h and self.k31_h and self.volume_central_L:
            # Calculate peripheral volume 2 from rate constants  
            self.cl_distribution2_L_h = self.k13_h * self.volume_central_L
            self.volume_peripheral2_L = self.cl_distribution2_L_h / self.k31_h
    
    # Removed get_api_parameters method - now use direct attribute access  
    # e.g., pk.clearance_L_h instead of pk.get_api_parameters('BD')['clearance_L_h']
    
    @classmethod
    def get_variability(cls, enable_variability: bool = True, pk_name: str = "default", seed: Optional[int] = None) -> 'PK':
        """Create PK instance with variability multiplicative factors computed from Variability_ builtin catalog.
        
        Args:
            enable_variability: Whether to enable variability (if False, returns factors of 1.0)
            pk_name: PK model name ('default', etc.)
            seed: Random seed for reproducibility
            
        Returns:
            PK instance where parameters hold [inter_factor, intra_factor, dependent_var] lists
        """
        import numpy as np
        from ..catalog.builtin_loader import BuiltinDataLoader
        
        if seed is not None:
            np.random.seed(seed)
        
        loader = BuiltinDataLoader()
        # Always load from Variability_ prefixed file
        var_params = loader.load_pk_parameters(f"Variability_{pk_name}")
        
        if not var_params:
            raise ValueError(f"Variability PK parameters 'Variability_{pk_name}' not found in builtin catalog")
        
        # Process parameters to compute multiplicative factors
        processed_params = {}
        
        for key, value in var_params.items():
            if key in ['name', 'description']:
                processed_params[key] = value  # Keep as-is for string fields
                
            elif key in ['clearance_L_h', 'hepatic_extraction', 'volume_central_L', 'q_inter_L_h', 'ka_h', 'f_bioavail']:
                # API-dependent parameters
                processed_params[key] = {}
                if isinstance(value, dict):
                    for api_name, var_spec in value.items():
                        if isinstance(var_spec, list) and len(var_spec) >= 5:
                            inter_sigma, inter_dist, intra_sigma, intra_dist, dependent = var_spec[:5]
                            
                            if not enable_variability:
                                inter_sigma = 0.0
                                intra_sigma = 0.0
                            
                            # Compute multiplicative factors
                            inter_factor = cls._compute_factor(inter_sigma, inter_dist)
                            intra_factor = cls._compute_factor(intra_sigma, intra_dist)
                            
                            processed_params[key][api_name] = [inter_factor, intra_factor, dependent]
                        else:
                            processed_params[key][api_name] = var_spec
            else:
                processed_params[key] = value
        
        # Create instance with processed variability factors
        instance = cls(**processed_params)
        instance.is_loaded = True
        return instance
    
    @staticmethod
    def _compute_factor(sigma: float, distribution: str) -> float:
        """Compute multiplicative factor based on distribution type and sigma.
        
        Args:
            sigma: Standard deviation parameter
            distribution: Distribution type ('lognormal' or 'normal')
            
        Returns:
            Multiplicative factor
        """
        import numpy as np
        
        if sigma == 0.0:
            return 1.0
            
        if distribution == 'lognormal':
            # For lognormal: mean=0 in log space, returns multiplicative factor
            return np.random.lognormal(mean=0, sigma=sigma)
        elif distribution == 'normal':
            # For normal: mean=1, returns additive factor around 1
            return np.random.normal(loc=1.0, scale=sigma)
        else:
            return 1.0


class Subject(BaseModel):
    """Unified Subject class containing instances of all domain entities."""
    
    # Basic identification
    name: str = Field(..., description="Subject name")
    description: Optional[str] = Field(None, description="Subject description")
    
    # Domain entity instances
    demographic: Optional[Demographic] = Field(None, description="Demographic information")
    lung_regional: Optional[LungRegional] = Field(None, description="Regional lung physiology")
    lung_generation: Optional[LungGeneration] = Field(None, description="Generational lung geometry")
    gi: Optional[GI] = Field(None, description="GI tract physiology")
    pk: Optional[PK] = Field(None, description="Pharmacokinetic parameters")
    inhalation_maneuver: Optional[InhalationManeuver] = Field(None, description="Inhalation maneuver parameters")
    
    # Internal state
    is_loaded: bool = Field(False, exclude=True, description="Whether all components are loaded")
    
    @classmethod
    def from_builtin(
        cls,
        subject_name: str,
        demographic_name: str = "healthy_reference",
        lung_geometry_population: str = "healthy",
        gi_name: str = "default",
        inhalation_profile: str = "pMDI_variable_trapezoid",
        api_name: str = "BD"
    ) -> 'Subject':
        """Create Subject instance with all components loaded from builtin catalog.
        
        Args:
            subject_name: Name for this subject instance
            demographic_name: Demographic profile name
            lung_geometry_population: Lung geometry population type
            gi_name: GI tract model name
            inhalation_profile: Inhalation maneuver profile name
            api_name: API name for loading default PK parameters ('BD', 'GP', 'FF')
            
        Returns:
            Subject instance with all domain entities loaded
        """
        # Load all domain entities from builtin
        demographic = Demographic.from_builtin(demographic_name)
        lung_regional = LungRegional.from_builtin()
        lung_generation = LungGeneration.from_builtin(lung_geometry_population)
        gi = GI.from_builtin(gi_name)
        pk = PK.from_builtin(api_name)  # Use API name to load default PK parameters
        inhalation_maneuver = InhalationManeuver.from_builtin(inhalation_profile)
        
        # Create unified subject
        subject = cls(
            name=subject_name,
            description=f"Subject with {demographic_name} demographics and {lung_geometry_population} lung geometry",
            demographic=demographic,
            lung_regional=lung_regional,
            lung_generation=lung_generation,
            gi=gi,
            pk=pk,
            inhalation_maneuver=inhalation_maneuver
        )
        subject.is_loaded = True
        
        return subject
    
    def get_demographic_info(self) -> Dict[str, Any]:
        """Get demographic information as dictionary."""
        if not self.demographic:
            return {}
        return {
            "age_years": self.demographic.age_years,
            "weight_kg": self.demographic.weight_kg,
            "height_cm": self.demographic.height_cm,
            "sex": self.demographic.sex,
            "bmi_kg_m2": self.demographic.bmi_kg_m2,
            "bsa_m2": self.demographic.bsa_m2,
            "frc_ml": self.demographic.frc_ml
        }
    
    def get_lung_regions(self) -> List[str]:
        """Get list of lung regions from regional model."""
        if not self.lung_regional:
            return []
        return self.lung_regional.regions
    
    def get_gi_parameters_for_api(self, api_name: str) -> Dict[str, List[float]]:
        """Get GI parameters for specific API."""
        if not self.gi:
            return {}
        return self.gi.get_api_parameters(api_name)
    
    def get_inhalation_flow_profile(self) -> Optional[np.ndarray]:
        """Calculate and return inhalation flow profile."""
        if not self.inhalation_maneuver:
            return None
        return self.inhalation_maneuver.calculate_inhalation_maneuver_flow_profile()
    
    def validate_completeness(self) -> bool:
        """Check if all required domain entities are loaded."""
        required_components = [
            self.demographic,
            self.lung_regional,
            self.lung_generation,
            self.gi,
            self.inhalation_maneuver
        ]
        return all(component is not None for component in required_components)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert subject to dictionary representation."""
        return {
            "name": self.name,
            "description": self.description,
            "demographic": self.demographic.model_dump() if self.demographic else None,
            "lung_regional": self.lung_regional.model_dump() if self.lung_regional else None,
            "lung_generation": self.lung_generation.model_dump() if self.lung_generation else None,
            "gi": self.gi.model_dump() if self.gi else None,
            "inhalation_maneuver": self.inhalation_maneuver.model_dump() if self.inhalation_maneuver else None,
            "is_complete": self.validate_completeness()
        }
    
    def apply_variability(self, enable_variability: bool = True, api: Optional[str] = None, ref_lung_array: Optional[np.ndarray] = None) -> Tuple['Subject', 'Subject']:
        """Apply variability to create inter and intra subject instances, then compute final attributes.
        
        This demonstrates the universal structure for applying variability:
        - Inter subject: base_value * inter_factor
        - Intra subject: inter_value * (1 + intra_factor)
        - Final attributes: computed using scaling and flow profile methods after variability
        
        Args:
            enable_variability: Whether to apply variability
            api: API name for dependent parameters
            ref_lung_array: Reference lung array for scaling (optional)
            
        Returns:
            Tuple of (inter_subject, intra_subject) with variability applied and final attributes computed
        """
        # Get variability factors for each domain
        demographic_var = Demographic.get_variability(enable_variability, self.demographic.name if self.demographic else "healthy_reference")
        lung_regional_var = LungRegional.get_variability(enable_variability) if self.lung_regional else None
        lung_generation_var = LungGeneration.get_variability(enable_variability, self.lung_generation.name if self.lung_generation else "healthy") if self.lung_generation else None  
        gi_var = GI.get_variability(enable_variability, self.gi.name if self.gi else "default")
        pk_var = PK.get_variability(enable_variability, self.pk.name if self.pk else "default") if self.pk else None
        inhalation_var = InhalationManeuver.get_variability(enable_variability, self.inhalation_maneuver.name if self.inhalation_maneuver else "pMDI_variable_trapezoid") if self.inhalation_maneuver else None
        
        # Create inter subject by applying inter factors
        inter_subject = self.model_copy(deep=True)
        
        # Apply demographic variability for inter subject
        if inter_subject.demographic and demographic_var:
            # Apply demographic parameter variability (weight, height, FRC, etc.)
            demo_params = ['weight_kg', 'height_cm', 'frc_ml', 'frc_ref_ml', 'tidal_volume_ml', 'respiratory_rate_bpm']
            for param in demo_params:
                if hasattr(inter_subject.demographic, param) and hasattr(demographic_var, param):
                    base_val = getattr(self.demographic, param, 0.0)
                    var_list = getattr(demographic_var, param, [])
                    if isinstance(var_list, list) and len(var_list) >= 2 and base_val:
                        inter_factor = var_list[0]  # Inter-subject factor  
                        setattr(inter_subject.demographic, param, base_val * inter_factor)
        
        # Apply GI variability for inter subject
        if inter_subject.gi and api and gi_var:
            for param_group in ['gi_area', 'gi_tg', 'gi_vol']:
                if hasattr(inter_subject.gi, param_group) and hasattr(gi_var, param_group):
                    base_values = getattr(self.gi, param_group).get(api, [])
                    var_factors = getattr(gi_var, param_group).get(api, [])
                    
                    if base_values and var_factors:
                        inter_values = []
                        for base_val, var_list in zip(base_values, var_factors):
                            if isinstance(var_list, list) and len(var_list) >= 2:
                                inter_factor = var_list[0]  # Inter-subject factor
                                inter_values.append(base_val * inter_factor)
                            else:
                                inter_values.append(base_val)
                        
                        getattr(inter_subject.gi, param_group)[api] = inter_values

        # Apply PK variability for inter subject (limited to clearance and hepatic extraction only)
        if inter_subject.pk and api and pk_var:
            pk_params = ['clearance_L_h', 'hepatic_extraction']  # Only these parameters have variability
            for param in pk_params:
                if hasattr(inter_subject.pk, param) and hasattr(pk_var, param):
                    base_val = getattr(self.pk, param).get(api, 0.0)
                    var_list = getattr(pk_var, param).get(api, [])
                    if isinstance(var_list, list) and len(var_list) >= 2 and base_val:
                        inter_factor = var_list[0]  # Inter-subject factor
                        getattr(inter_subject.pk, param)[api] = base_val * inter_factor

        # Apply inhalation maneuver variability for inter subject
        if inter_subject.inhalation_maneuver and inhalation_var:
            # Note: inhaled_volume_L excluded as it's scaled proportionally to FRC changes
            inhalation_params = ['pifr_Lpm', 'rise_time_s', 'hold_time_s', 'breath_hold_time_s', 'exhalation_flow_Lpm', 'bolus_volume_ml', 'bolus_delay_s', 'et_scale_factor']
            for param in inhalation_params:
                if hasattr(inter_subject.inhalation_maneuver, param) and hasattr(inhalation_var, param):
                    base_val = getattr(self.inhalation_maneuver, param, 0.0)
                    var_list = getattr(inhalation_var, param, [])
                    if isinstance(var_list, list) and len(var_list) >= 2 and base_val:
                        inter_factor = var_list[0]  # Inter-subject factor
                        setattr(inter_subject.inhalation_maneuver, param, base_val * inter_factor)
        
        # Create intra subject by applying both inter and intra factors
        intra_subject = inter_subject.model_copy(deep=True)
        
        # Apply demographic variability for intra subject
        if intra_subject.demographic and demographic_var:
            demo_params = ['weight_kg', 'height_cm', 'frc_ml', 'frc_ref_ml', 'tidal_volume_ml', 'respiratory_rate_bpm']
            for param in demo_params:
                if hasattr(intra_subject.demographic, param) and hasattr(demographic_var, param):
                    inter_val = getattr(inter_subject.demographic, param, 0.0)
                    var_list = getattr(demographic_var, param, [])
                    if isinstance(var_list, list) and len(var_list) >= 2 and inter_val:
                        intra_factor = var_list[1]  # Intra-subject factor
                        setattr(intra_subject.demographic, param, inter_val * (1 + intra_factor))
        
        # Apply GI variability for intra subject
        if intra_subject.gi and api and gi_var:
            for param_group in ['gi_area', 'gi_tg', 'gi_vol']:
                if hasattr(intra_subject.gi, param_group) and hasattr(gi_var, param_group):
                    inter_values = getattr(inter_subject.gi, param_group).get(api, [])
                    var_factors = getattr(gi_var, param_group).get(api, [])
                    
                    if inter_values and var_factors:
                        intra_values = []
                        for inter_val, var_list in zip(inter_values, var_factors):
                            if isinstance(var_list, list) and len(var_list) >= 2:
                                intra_factor = var_list[1]  # Intra-subject factor
                                # Apply cascade: intra = inter * (1 + intra_factor)
                                intra_values.append(inter_val * (1 + intra_factor))
                            else:
                                intra_values.append(inter_val)
                        
                        getattr(intra_subject.gi, param_group)[api] = intra_values

        # Apply PK variability for intra subject (limited to clearance and hepatic extraction only)
        if intra_subject.pk and api and pk_var:
            pk_params = ['clearance_L_h', 'hepatic_extraction']  # Only these parameters have variability
            for param in pk_params:
                if hasattr(intra_subject.pk, param) and hasattr(pk_var, param):
                    inter_val = getattr(inter_subject.pk, param).get(api, 0.0)
                    var_list = getattr(pk_var, param).get(api, [])
                    if isinstance(var_list, list) and len(var_list) >= 2 and inter_val:
                        intra_factor = var_list[1]  # Intra-subject factor
                        getattr(intra_subject.pk, param)[api] = inter_val * (1 + intra_factor)

        # Apply inhalation maneuver variability for intra subject
        if intra_subject.inhalation_maneuver and inhalation_var:
            # Note: inhaled_volume_L excluded as it's scaled proportionally to FRC changes
            inhalation_params = ['pifr_Lpm', 'rise_time_s', 'hold_time_s', 'breath_hold_time_s', 'exhalation_flow_Lpm', 'bolus_volume_ml', 'bolus_delay_s', 'et_scale_factor']
            for param in inhalation_params:
                if hasattr(intra_subject.inhalation_maneuver, param) and hasattr(inhalation_var, param):
                    inter_val = getattr(inter_subject.inhalation_maneuver, param, 0.0)
                    var_list = getattr(inhalation_var, param, [])
                    if isinstance(var_list, list) and len(var_list) >= 2 and inter_val:
                        intra_factor = var_list[1]  # Intra-subject factor
                        setattr(intra_subject.inhalation_maneuver, param, inter_val * (1 + intra_factor))
        
        # Compute final attributes after variability is applied
        
        # Scale inhaled_volume proportionally to FRC changes for inter subject
        if inter_subject.inhalation_maneuver and inter_subject.demographic and self.demographic:
            if inter_subject.demographic.frc_ml and self.demographic.frc_ref_ml:
                frc_ratio = inter_subject.demographic.frc_ml / self.demographic.frc_ref_ml
                inter_subject.inhalation_maneuver.inhaled_volume_L = self.inhalation_maneuver.inhaled_volume_L * frc_ratio
        
        # Scale inhaled_volume proportionally to FRC changes for intra subject
        if intra_subject.inhalation_maneuver and intra_subject.demographic and self.demographic:
            if intra_subject.demographic.frc_ml and self.demographic.frc_ref_ml:
                frc_ratio = intra_subject.demographic.frc_ml / self.demographic.frc_ref_ml
                intra_subject.inhalation_maneuver.inhaled_volume_L = self.inhalation_maneuver.inhaled_volume_L * frc_ratio
        
        # 1. Lung scaling for inter subject
        if inter_subject.lung_regional and inter_subject.demographic and inter_subject.demographic.frc_ml and ref_lung_array is not None:
            inter_subject._final_lung_regional = inter_subject.lung_regional.lung_scaling(ref_lung_array, inter_subject.demographic.frc_ml)
        if inter_subject.lung_generation and inter_subject.demographic and inter_subject.demographic.frc_ml and ref_lung_array is not None:
            inter_subject._final_lung_generation = inter_subject.lung_generation.lung_scaling(ref_lung_array, inter_subject.demographic.frc_ml)
        
        # 2. Flow profile for inter subject
        if inter_subject.inhalation_maneuver:
            inter_subject._final_flow_profile = inter_subject.inhalation_maneuver.calculate_inhalation_maneuver_flow_profile()
        
        # 3. Lung scaling for intra subject  
        if intra_subject.lung_regional and intra_subject.demographic and intra_subject.demographic.frc_ml and ref_lung_array is not None:
            intra_subject._final_lung_regional = intra_subject.lung_regional.lung_scaling(ref_lung_array, intra_subject.demographic.frc_ml)
        if intra_subject.lung_generation and intra_subject.demographic and intra_subject.demographic.frc_ml and ref_lung_array is not None:
            intra_subject._final_lung_generation = intra_subject.lung_generation.lung_scaling(ref_lung_array, intra_subject.demographic.frc_ml)
        
        # 4. Flow profile for intra subject
        if intra_subject.inhalation_maneuver:
            intra_subject._final_flow_profile = intra_subject.inhalation_maneuver.calculate_inhalation_maneuver_flow_profile()
        
        return inter_subject, intra_subject
    
    def get_final_lung_regional(self) -> Optional[np.ndarray]:
        """Get final lung regional parameters after scaling (if computed)."""
        return getattr(self, '_final_lung_regional', None)
    
    def get_final_lung_generation(self) -> Optional[np.ndarray]:
        """Get final lung generation parameters after scaling (if computed)."""
        return getattr(self, '_final_lung_generation', None)
    
    def get_final_flow_profile(self) -> Optional[np.ndarray]:
        """Get final flow profile after variability (if computed)."""
        return getattr(self, '_final_flow_profile', None)
    
    def get_final_attributes(self) -> Dict[str, Any]:
        """Get all final computed attributes after variability and scaling.
        
        Returns:
            Dictionary containing final lung scaling results and flow profile
        """
        return {
            'final_lung_regional': self.get_final_lung_regional(),
            'final_lung_generation': self.get_final_lung_generation(), 
            'final_flow_profile': self.get_final_flow_profile(),
            'has_final_attributes': any([
                self.get_final_lung_regional() is not None,
                self.get_final_lung_generation() is not None,
                self.get_final_flow_profile() is not None
            ])
        }
    
    def get_final_values(self, apply_variability: bool = False, api_name: Optional[str] = None):
        """Get final transformed subject instance for deposition calculations.
        
        This method applies all necessary transformations to get production-ready parameters:
        - Applies variability if requested
        - Scales lung geometry using FRC
        - Scales inhaled volume proportionally to FRC changes
        - Computes final flow profile
        - Stores computed values as attributes on the subject instance
        
        Args:
            apply_variability: Whether to apply inter/intra subject variability
            api_name: API name for API-specific transformations
            
        Returns:
            Subject instance with computed attributes (scaled_lung_geometry, flow_profile, etc.)
        """
        final_subject = self
        
        # Apply variability if requested
        if apply_variability:
            inter_subject, intra_subject = self.apply_variability(enable_variability=True, api=api_name or "BD")
            final_subject = intra_subject  # Use intra subject as final
        
        # Get scaled lung geometry (always apply lung scaling)
        if final_subject.lung_generation and final_subject.demographic:
            ref_lung_geometry = np.array(final_subject.lung_generation.lung_geometry)
            v_lung = final_subject.demographic.frc_ml  # FRC in mL, will be converted to m³ in lung_scaling
            scaled_lung_geometry = final_subject.lung_generation.lung_scaling(ref_lung_geometry, v_lung)
        else:
            scaled_lung_geometry = None
        
        # Get final flow profile and convert units from L/min to m³/s
        if final_subject.inhalation_maneuver:
            flow_profile_L_per_min = final_subject.inhalation_maneuver.calculate_inhalation_maneuver_flow_profile()
            # Convert flow rates from L/min to m³/s: 1 L/min = 0.001 m³ / 60 s = 1.667e-5 m³/s
            # Only convert the flow rate column (column 1), not the time column (column 0)
            flow_profile = flow_profile_L_per_min.copy()
            flow_profile[:, 1] = flow_profile_L_per_min[:, 1] * (0.001 / 60.0)
        else:
            flow_profile = None
        
        # Scale inhaled volume proportionally to FRC changes
        inhaled_volume_L = final_subject.inhalation_maneuver.inhaled_volume_L
        if final_subject.demographic and final_subject.demographic.frc_ref_ml:
            frc_ratio = final_subject.demographic.frc_ml / final_subject.demographic.frc_ref_ml
            inhaled_volume_L = final_subject.inhalation_maneuver.inhaled_volume_L * frc_ratio
        
        # Store computed values as attributes on the subject instance
        final_subject._scaled_lung_geometry = scaled_lung_geometry
        final_subject._flow_profile = flow_profile
        final_subject._scaled_inhaled_volume_L = inhaled_volume_L
        
        return final_subject


class EntityCollection(BaseModel):
    """Collection of related entities with metadata."""
    
    name: str = Field(..., description="Collection name")
    version: str = Field(default="1.0", description="Schema version")
    description: Optional[str] = Field(None, description="Collection description")
    
    # Entity collections
    demographics: Dict[str, Demographic] = Field(default_factory=dict)
    apis: Dict[str, API] = Field(default_factory=dict)
    products: Dict[str, Product] = Field(default_factory=dict)
    maneuvers: Dict[str, InhalationManeuver] = Field(default_factory=dict)
    gi_tracts: Dict[str, GI] = Field(default_factory=dict)
    subjects: Dict[str, 'Subject'] = Field(default_factory=dict)
    
    def get_entity(self, category: str, name: str) -> BaseModel:
        """Get entity by category and name."""
        collections = {
            "demographic": self.demographics,
            "api": self.apis, 
            "product": self.products,
            "maneuver": self.maneuvers,
            "gi": self.gi_tracts,
            "subject": self.subjects
        }
        
        if category not in collections:
            raise ValueError(f"Unknown entity category: {category}")
            
        collection = collections[category]
        if name not in collection:
            raise ValueError(f"Entity '{name}' not found in category '{category}'")
            
        return collection[name]
    
    def list_entities(self, category: str) -> List[str]:
        """List entity names in a category."""
        collections = {
            "demographic": self.demographics,
            "api": self.apis,
            "product": self.products, 
            "maneuver": self.maneuvers,
            "gi": self.gi_tracts,
            "subject": self.subjects
        }
        
        if category not in collections:
            raise ValueError(f"Unknown entity category: {category}")
            
        return list(collections[category].keys())


# Note: EntityCollection.model_rebuild() called when Subject is imported