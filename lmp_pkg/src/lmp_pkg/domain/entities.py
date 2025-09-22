"""Domain entity schemas using Pydantic.

Note: Subject class has been moved to domain/subject.py as a unified class.
This file now contains only API, Product, InhalationManeuver, and other entities.
"""

from __future__ import annotations
from typing import Optional, Dict, Any, List, Tuple, TYPE_CHECKING, Mapping
from dataclasses import dataclass, fields
from pydantic import BaseModel, Field, field_validator, computed_field, PrivateAttr
import math
import numpy as np

from ..config import constants

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
    def from_builtin(cls, subject_name: str) -> 'Demographic':
        """Create Demographic instance with parameters loaded from builtin catalog.
        
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
    def get_variability(cls, enable_variability: bool = True, subject_name: str = "healthy_reference") -> Dict[str, Any]:
        """Get variability parameters from Variability_ builtin catalog as raw dict.
        
        Args:
            enable_variability: Whether to enable variability (if False, sets first element to 0.0)
            subject_name: Subject name ('healthy_reference', 'adult_70kg', etc.)
            
        Returns:
            Demographic instance with variability parameters loaded (parameters hold lists)
        """
        from ..catalog.builtin_loader import BuiltinDataLoader
        
        loader = BuiltinDataLoader()
        # Always load from Variability_ prefixed file
        params = loader.load_variability_file('subject', f"Variability_{subject_name}")

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
                    if len(params[param]) > 2:
                        params[param][2] = 0.0
                    if len(params[param]) > 1 and not params[param][1]:
                        params[param][1] = 'lognormal'
                    if len(params[param]) > 3 and not params[param][3]:
                        params[param][3] = 'lognormal'
        
        # Return raw variability params (lists) for downstream sampling
        return params


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

    @field_validator("apis", mode="before")
    @classmethod
    def _normalize_apis(cls, v):
        """Normalize apis from [[apis]] list-of-tables to dict keyed by API name.
        Accept both list (with name field) and dict forms. Return empty dict if None.
        """
        if v is None:
            return {}
        # If already a dict, assume correct shape {"BD": {...}, ...}
        if isinstance(v, dict):
            return v
        # Convert list of tables to dict using the 'name' field as key
        if isinstance(v, list):
            normalized: Dict[str, Dict[str, float]] = {}
            for item in v:
                if not isinstance(item, dict):
                    continue
                name = item.get("name") or item.get("api") or item.get("id")
                if not name:
                    continue
                entry = {k: val for k, val in item.items() if k != "name"}
                normalized[str(name)] = entry  # coerce key to string
            return normalized
        return v

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
    
    # Transit time (seconds)
    tg: Dict[str, float] = Field(..., description="Transit time by region")
    
    # Volume fraction
    V_frac_g: float = Field(..., description="Volume fraction")
    
    # Number of epithelial layers
    n_epi_layer: Dict[str, int] = Field(..., description="Number of epithelial layers by region")
    
    # Internal state
    is_loaded: bool = Field(False, exclude=True, description="Whether parameters are loaded from builtin")
    
    @classmethod
    def from_builtin(cls) -> 'LungRegional':
        lung_generation = LungGeneration.from_builtin()
        instance = cls.from_lung_generation(lung_generation)
        instance.is_loaded = True
        return instance

    @classmethod
    def from_lung_generation(
        cls,
        lung_generation: 'LungGeneration',
        name: str = "regional",
        description: Optional[str] = "Derived from generation data",
    ) -> 'LungRegional':
        params = lung_generation.compute_regional_parameters()
        instance = cls(name=name, description=description, **params)
        instance.is_loaded = True
        return instance

    @classmethod
    def get_variability(cls, enable_variability: bool = True) -> Dict[str, Any]:
        return {}


class LungGeneration(BaseModel):
    """Generational lung geometry/physiology derived from builtin tables."""

    name: str = Field(..., description="Geometry model name")
    description: Optional[str] = Field(None, description="Model description")
    generations: List[Dict[str, Any]] = Field(..., description="Generation-level data")

    is_loaded: bool = Field(False, exclude=True, description="Whether parameters are loaded from builtin")

    _sorted_generations: List[Dict[str, Any]] = PrivateAttr(default_factory=list)
    _generation_numbers: Optional[np.ndarray] = PrivateAttr(default=None)
    _regions: List[str] = PrivateAttr(default_factory=list)
    _multiplicity: Optional[np.ndarray] = PrivateAttr(default=None)
    _length_cm: Optional[np.ndarray] = PrivateAttr(default=None)
    _diameter_cm: Optional[np.ndarray] = PrivateAttr(default=None)
    _alveoli_volume_ml: Optional[np.ndarray] = PrivateAttr(default=None)
    _extra_surface_area_cm2: Optional[np.ndarray] = PrivateAttr(default=None)
    _d_elf_cm: Optional[np.ndarray] = PrivateAttr(default=None)
    _d_epi_cm: Optional[np.ndarray] = PrivateAttr(default=None)
    _q_g_ml_min: Optional[np.ndarray] = PrivateAttr(default=None)
    _v_tissue_ml: Optional[np.ndarray] = PrivateAttr(default=None)
    _transit_time_s: Optional[np.ndarray] = PrivateAttr(default=None)
    _expansion_fraction: Optional[np.ndarray] = PrivateAttr(default=None)
    _branching_angle_deg: Optional[np.ndarray] = PrivateAttr(default=None)
    _gravity_angle_deg: Optional[np.ndarray] = PrivateAttr(default=None)
    _radius_m: Optional[np.ndarray] = PrivateAttr(default=None)
    _length_m: Optional[np.ndarray] = PrivateAttr(default=None)
    _alveoli_m3: Optional[np.ndarray] = PrivateAttr(default=None)
    _airway_volume_m3: Optional[np.ndarray] = PrivateAttr(default=None)
    _total_volume_m3: Optional[np.ndarray] = PrivateAttr(default=None)
    _airway_area_cm2: Optional[np.ndarray] = PrivateAttr(default=None)
    _total_area_cm2: Optional[np.ndarray] = PrivateAttr(default=None)
    _reference_volume_m3: float = PrivateAttr(default=2999.60e-6)

    def model_post_init(self, __context: Any) -> None:  # type: ignore[override]
        self._recompute_metrics()

    @classmethod
    def from_builtin(cls, population: str = "healthy") -> 'LungGeneration':
        from ..catalog.builtin_loader import BuiltinDataLoader

        loader = BuiltinDataLoader()
        generation_rows = loader.load_lung_geometry(population)
        if not generation_rows:
            raise ValueError(f"Lung geometry parameters for '{population}' not found in builtin catalog")

        instance = cls(
            name=f"geometry_{population}",
            description="Reference generation-level geometry",
            generations=generation_rows,
        )
        instance.is_loaded = True
        return instance

    @classmethod
    def get_variability(cls, enable_variability: bool = True, population: str = "healthy") -> Dict[str, Any]:
        """Placeholder for variability; currently returns empty configuration."""
        return {}

    def _recompute_metrics(self) -> None:
        generations = sorted(self.generations, key=lambda x: x.get('generation', 0))
        self._sorted_generations = generations

        def _col(name: str, default: float = 0.0) -> np.ndarray:
            return np.array([float(row.get(name, default) or 0.0) for row in generations], dtype=float)

        self._generation_numbers = np.array([
            int(row.get('generation', 0)) for row in generations
        ], dtype=int)
        self._regions = [str(row.get('region', '')).strip() for row in generations]

        self._multiplicity = _col('multiplicity')
        self._length_cm = _col('length_cm')
        self._diameter_cm = _col('diameter_cm')
        self._alveoli_volume_ml = _col('alveoli_volume_ml')
        self._extra_surface_area_cm2 = _col('extra_surface_area_cm2')
        self._d_elf_cm = _col('d_elf_cm')
        self._d_epi_cm = _col('d_epi_cm')
        self._q_g_ml_min = _col('q_g_ml_min') 
        self._q_g_ml_s = _col('q_g_ml_min') / 60.0  # convert to mL/s
        self._v_tissue_ml = _col('v_tissue_ml')
        self._transit_time_s = _col('transit_time_s')
        self._expansion_fraction = _col('expansion_fraction')
        self._branching_angle_deg = _col('branching_angle_deg')
        self._gravity_angle_deg = _col('gravity_angle_deg')

        radius_cm = self._diameter_cm / 2.0
        length_cm = self._length_cm
        mult = self._multiplicity

        # conversions
        radius_m = radius_cm / 100.0
        length_m = length_cm / 100.0
        alveoli_m3 = self._alveoli_volume_ml * 1e-6

        airway_volume_m3 = np.pi * (radius_m ** 2) * length_m * mult
        total_volume_m3 = airway_volume_m3 + alveoli_m3

        airway_area_cm2 = 2.0 * np.pi * radius_cm * length_cm * mult
        total_area_cm2 = airway_area_cm2 + self._extra_surface_area_cm2

        self._radius_m = radius_m
        self._length_m = length_m
        self._alveoli_m3 = alveoli_m3
        self._airway_volume_m3 = airway_volume_m3
        self._total_volume_m3 = total_volume_m3
        self._airway_area_cm2 = airway_area_cm2
        self._total_area_cm2 = total_area_cm2

        self._reference_volume_m3 = 2999.60e-6 #float(np.sum(total_volume_m3)) if np.sum(total_volume_m3) > 0 else 2999.60e-6

    def compute_scaled_geometry(self, frc_ml: float) -> np.ndarray:
        if self._multiplicity is None or self._multiplicity.size == 0:
            self._recompute_metrics()

        lung_size_m3 = frc_ml / 1e6
        sf = (lung_size_m3 / self._reference_volume_m3) ** (1.0 / 3.0)

        radius_scaled = self._radius_m * sf
        length_scaled = self._length_m * sf
        alveoli_scaled = self._alveoli_m3 * sf**3
        airway_volume_scaled = self._airway_volume_m3 * sf**3
        total_volume_scaled = airway_volume_scaled + alveoli_scaled

        scaled = np.zeros((len(self._multiplicity), 8), dtype=float)
        scaled[:, 0] = self._expansion_fraction if self._expansion_fraction is not None else 0.0
        scaled[:, 1] = self._multiplicity
        scaled[:, 2] = alveoli_scaled
        scaled[:, 3] = radius_scaled
        scaled[:, 4] = length_scaled
        scaled[:, 5] = total_volume_scaled
        scaled[:, 6] = self._branching_angle_deg if self._branching_angle_deg is not None else 0.0
        scaled[:, 7] = self._gravity_angle_deg if self._gravity_angle_deg is not None else 0.0
        scaled[0, 5] = 50e-6 # Must set this to avoid zero volume in mouth-throat
        return scaled

    def compute_regional_parameters(self) -> Dict[str, Any]:
        if self._multiplicity is None or self._multiplicity.size == 0:
            self._recompute_metrics()

        region_aliases = {
            'ET': 'ET',
            'et': 'ET',
            'BB': 'BB',
            'bb': 'bb',
            'Bb': 'bb',
            'bb_b': 'bb',
            'Al': 'Al',
            'AL': 'Al',
        }

        regions_order = ['ET', 'BB', 'bb', 'Al']
        region_keys = [region_aliases.get(region, region) for region in self._regions]

        A_elf_ref: Dict[str, float] = {r: 0.0 for r in regions_order}
        extra_area_ref: Dict[str, float] = {r: 0.0 for r in regions_order}
        d_elf: Dict[str, float] = {r: 0.0 for r in regions_order}
        d_epi: Dict[str, float] = {r: 0.0 for r in regions_order}
        V_tissue: Dict[str, float] = {r: 0.0 for r in regions_order}
        Q_g: Dict[str, float] = {r: 0.0 for r in regions_order}
        tg: Dict[str, float] = {r: 0.0 for r in regions_order}
        n_epi_layer: Dict[str, int] = {r: 1 for r in regions_order}
        n_epi_values: Dict[str, list[int]] = {r: [] for r in regions_order}
        v_frac_values: list[float] = []

        for idx, row in enumerate(self._sorted_generations):
            region_key = region_keys[idx]
            if region_key not in regions_order:
                continue

            #A_elf_ref[region_key] += float(self._airway_area_cm2[idx])
            A_elf_ref[region_key] += float(self._total_area_cm2[idx])
            extra_area_ref[region_key] += float(self._extra_surface_area_cm2[idx])
            V_tissue[region_key] += float(self._v_tissue_ml[idx])
            Q_g[region_key] += float(self._q_g_ml_s[idx])
            tg[region_key] += float(self._transit_time_s[idx])

            n_val = row.get('n_epi_layer')
            if n_val is not None:
                try:
                    n_epi_values[region_key].append(int(float(n_val)))
                except (TypeError, ValueError):
                    pass

            v_frac = row.get('V_frac_g') or row.get('v_frac_g')
            if v_frac is not None:
                try:
                    v_frac_values.append(float(v_frac))
                except (TypeError, ValueError):
                    pass

        for region_key in regions_order:
            indices = [i for i, r in enumerate(region_keys) if r == region_key]
            if not indices:
                continue

            #airway_weights = self._airway_area_cm2[indices]
            #total_area_weights = self._airway_area_cm2[indices] + self._extra_surface_area_cm2[indices]
            #airway_weights = self._airway_area_cm2[indices]
            airway_weights = self._total_area_cm2[indices]
            #total_area_weights = self._airway_area_cm2[indices] + self._extra_surface_area_cm2[indices]
            d_elf_vals = self._d_elf_cm[indices]
            d_epi_vals = self._d_epi_cm[indices]

            if np.any(airway_weights > 0):
                d_elf[region_key] = float(np.average(d_elf_vals, weights=airway_weights))
                d_epi[region_key] = float(np.average(d_epi_vals, weights=airway_weights))
            else:
                d_elf[region_key] = float(np.mean(d_elf_vals))
                d_epi[region_key] = float(np.mean(d_epi_vals))
            #if np.any(total_area_weights > 0):
            #    d_epi[region_key] = float(np.average(d_epi_vals, weights=total_area_weights))
            #else:
            #    d_epi[region_key] = float(np.mean(d_epi_vals))

            if n_epi_values[region_key]:
                n_epi_layer[region_key] = int(round(np.median(n_epi_values[region_key])))

        params = {
            'regions': regions_order,
            'A_elf_ref': {k: float(v) for k, v in A_elf_ref.items()},
            'extra_area_ref': {k: float(v) for k, v in extra_area_ref.items()},
            'd_elf': d_elf,
            'd_epi': d_epi,
            'V_tissue': {k: float(v) for k, v in V_tissue.items()},
            'Q_g': {k: float(v) for k, v in Q_g.items()},
            'tg': {k: float(v) for k, v in tg.items()},
            'V_frac_g': float(np.mean(v_frac_values)) if v_frac_values else 0.2,
            'n_epi_layer': n_epi_layer,
        }
        return params


class InhalationManeuver(BaseModel):
    """Inhalation maneuver characteristics loaded from builtin catalog."""
    
    # Basic identification
    name: str = Field(..., description="Profile identifier")
    description: Optional[str] = Field(None, description="Profile description")
    maneuver_type: str = Field(..., description="Type of inhalation maneuver")
    
    # Peak inspiratory flow rate (L/min)
    pifr_Lpm: float = Field(..., gt=0, description="Peak inspiratory flow rate in L/min")
    
    # Rise time to peak flow (s)
    rise_time_s: Optional[float] = Field(
        default=None,
        ge=0,
        description="Rise time to peak flow in seconds",
    )
    
    # Inhaled volume (L)
    inhaled_volume_L: float = Field(..., gt=0, description="Inhaled volume in L")
    
    # Hold time at peak flow (s)
    hold_time_s: Optional[float] = Field(
        default=None,
        ge=0,
        description="Hold time at peak flow in seconds",
    )
    
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

    # Optional tabulated flow profile [time_s, flow_Lpm]
    tabulated_flow_profile: Optional[List[Tuple[float, float]]] = Field(
        default=None,
        description="Tabulated inhalation flow profile as [time_s, flow_Lpm] pairs",
    )

    # Internal state
    is_loaded: bool = Field(False, exclude=True, description="Whether parameters are loaded from builtin")
    _baseline_pifr_Lpm: float = PrivateAttr(default=None)
    _tabulated_reference_profile: Optional[np.ndarray] = PrivateAttr(default=None)

    def model_post_init(self, __context: Any) -> None:
        """Cache baseline values for variability-sensitive calculations."""
        if self._baseline_pifr_Lpm is None:
            self._baseline_pifr_Lpm = self.pifr_Lpm

        if self.maneuver_type.lower() == "tabulated" and self.tabulated_flow_profile:
            arr = np.asarray(self.tabulated_flow_profile, dtype=float)
            if arr.ndim != 2 or arr.shape[1] != 2:
                raise ValueError("tabulated_flow_profile must be an array of [time_s, flow_Lpm] pairs")
            # Ensure monotonically increasing time for downstream integrations
            order = np.argsort(arr[:, 0])
            self._tabulated_reference_profile = arr[order]
            constants.set_flow_profile_steps(len(self._tabulated_reference_profile))
        else:
            constants.reset_flow_profile_steps()
    
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
    def get_variability(cls, enable_variability: bool = True, profile_name: str = "pMDI_variable_trapezoid") -> Dict[str, Any]:
        """Load inhalation maneuver variability specifications from the builtin catalog.
        
        Args:
            enable_variability: Whether to enable variability (if False, sets first element to 0.0)
            profile_name: Profile name ('pMDI_variable_trapezoid', etc.)
        
        Returns:
            Dictionary with variability entries keyed by parameter name
        """
        from ..catalog.builtin_loader import BuiltinDataLoader

        loader = BuiltinDataLoader()
        params = loader.load_variability_file("inhalation", f"Variability_{profile_name}")

        if not enable_variability:
            inhalation_params = [
                'pifr_Lpm', 'rise_time_s', 'inhaled_volume_L', 'hold_time_s',
                'breath_hold_time_s', 'exhalation_flow_Lpm', 'bolus_volume_ml', 'bolus_delay_s', 'et_scale_factor'
            ]
            for param in inhalation_params:
                if param in params and isinstance(params[param], list):
                    if len(params[param]) > 0:
                        params[param][0] = 0.0
                    if len(params[param]) > 2:
                        params[param][2] = 0.0
                    if len(params[param]) > 1 and not params[param][1]:
                        params[param][1] = 'lognormal'
                    if len(params[param]) > 3 and not params[param][3]:
                        params[param][3] = 'lognormal'

        return params
    
    def calculate_inhalation_maneuver_flow_profile(self) -> np.ndarray:
        """Calculate inhalation flow profile using the original InhalationManeuver logic.

        Based on original lmp_apps/population/attributes.py InhalationManeuver.inhale_profile()
        Uses constants from config.constants for N_STEPS.

        Returns:
            Flow profile as [time_points, flow_rates] array with shape (N_STEPS, 2)
            Flow rates in L/min
        """
        from ..config.constants import N_STEPS

        maneuver_type = (self.maneuver_type or "").lower()
        steps = max(N_STEPS, 2)

        if maneuver_type == "constant":
            peak_flow_lpm = max(self.pifr_Lpm, 1e-6)
            duration_s = self.inhaled_volume_L / (peak_flow_lpm / 60.0)
            time_points = np.linspace(0.0, duration_s, steps)
            flow_lpm = np.full_like(time_points, peak_flow_lpm, dtype=float)
            return np.column_stack((time_points, flow_lpm))

        if maneuver_type == "tabulated":
            if self._tabulated_reference_profile is None and self.tabulated_flow_profile:
                arr = np.asarray(self.tabulated_flow_profile, dtype=float)
                if arr.ndim != 2 or arr.shape[1] != 2:
                    raise ValueError("tabulated_flow_profile must be an array of [time_s, flow_Lpm] pairs")
                order = np.argsort(arr[:, 0])
                self._tabulated_reference_profile = arr[order]

            if self._tabulated_reference_profile is None:
                raise ValueError("Tabulated flow profile requires tabulated_flow_profile data")

            baseline = self._baseline_pifr_Lpm or float(np.max(self._tabulated_reference_profile[:, 1]))
            if baseline <= 0:
                baseline = float(np.max(self._tabulated_reference_profile[:, 1]) or 1.0)

            scale = self.pifr_Lpm / baseline if baseline else 1.0
            profile = self._tabulated_reference_profile.copy()
            profile[:, 1] = np.maximum(profile[:, 1] * scale, 0.0)
            return profile

        # Default to variable trapezoid behaviour
        rise_time = self.rise_time_s or 0.0
        pifr_ls = self.pifr_Lpm / 60.0

        if pifr_ls <= 0 or rise_time <= 0:
            hold_time = self.inhaled_volume_L / max(pifr_ls, 1e-6)
        else:
            hold_time = (self.inhaled_volume_L - pifr_ls * rise_time) / pifr_ls
        hold_time = max(0.0, hold_time)

        inhaled_duration = hold_time + 2 * rise_time
        inhaled_duration = max(inhaled_duration, 1e-6)
        slope = pifr_ls / rise_time if rise_time > 0 else 0.0

        time_points = np.linspace(0.0, inhaled_duration, steps)
        flowrate_ls = np.zeros_like(time_points)

        if rise_time > 0:
            mask_rise = time_points < rise_time
            flowrate_ls[mask_rise] = time_points[mask_rise] * slope

        mask_hold = (time_points >= rise_time) & (time_points < (rise_time + hold_time))
        flowrate_ls[mask_hold] = pifr_ls

        mask_fall = time_points >= (rise_time + hold_time)
        if rise_time > 0:
            flowrate_ls[mask_fall] = pifr_ls - (time_points[mask_fall] - (rise_time + hold_time)) * slope

        flowrate_ls = np.maximum(flowrate_ls, 0.0)
        flow_profile = np.zeros((len(flowrate_ls), 2))
        flow_profile[:, 0] = time_points
        flow_profile[:, 1] = flowrate_ls * 60.0  # convert back to L/min

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
    def get_variability(cls, enable_variability: bool = True, gi_name: str = "default", seed: Optional[int] = None) -> Dict[str, Any]:
        """Load GI variability specifications from the builtin catalog.
        
        Args:
            enable_variability: Whether to enable variability (if False, factors default to 1.0)
            gi_name: GI tract model name ('default', etc.)
            seed: Random seed for reproducibility
        
        Returns:
            Dictionary describing variability entries for GI parameters
        """
        from ..catalog.builtin_loader import BuiltinDataLoader

        loader = BuiltinDataLoader()
        params = loader.load_variability_file("gi_tract", f"Variability_{gi_name}")

        if not enable_variability:
            if 'num_comp' in params and isinstance(params['num_comp'], list):
                if len(params['num_comp']) > 0:
                    params['num_comp'][0] = 0.0
                if len(params['num_comp']) > 2:
                    params['num_comp'][2] = 0.0
                if len(params['num_comp']) > 1 and not params['num_comp'][1]:
                    params['num_comp'][1] = 'lognormal'
                if len(params['num_comp']) > 3 and not params['num_comp'][3]:
                    params['num_comp'][3] = 'lognormal'

            for param_group in ['gi_area', 'gi_tg', 'gi_vol']:
                if param_group in params and isinstance(params[param_group], dict):
                    for api_name, comp_list in params[param_group].items():
                        if isinstance(comp_list, list):
                            for spec in comp_list:
                                if isinstance(spec, list):
                                    if len(spec) > 0:
                                        spec[0] = 0.0
                                    if len(spec) > 2:
                                        spec[2] = 0.0
                                    if len(spec) > 1 and not spec[1]:
                                        spec[1] = 'lognormal'
                                    if len(spec) > 3 and not spec[3]:
                                        spec[3] = 'lognormal'

        return params
    


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
    def get_variability(cls, enable_variability: bool = True, pk_name: str = "default", seed: Optional[int] = None) -> Dict[str, Any]:
        """Load PK variability specifications from the builtin catalog.
        
        Args:
            enable_variability: Whether to enable variability (if False, factors default to 1.0)
            pk_name: PK model name ('default', etc.)
            seed: Random seed for reproducibility
        
        Returns:
            Dictionary describing variability entries for PK parameters
        """
        from ..catalog.builtin_loader import BuiltinDataLoader

        loader = BuiltinDataLoader()
        name_key = pk_name or "default"
        if name_key.lower().startswith('pk_'):
            name_key = name_key[3:]
        filename = name_key if name_key.lower().startswith('variability_') else f"Variability_{name_key}"

        try:
            params = loader.load_variability_file("pk", filename)
        except FileNotFoundError:
            # Fallback to default variability file when specific mapping is not available
            params = loader.load_variability_file("pk", "Variability_default")

        if not enable_variability:
            for key in ['clearance_L_h', 'hepatic_extraction', 'volume_central_L', 'q_inter_L_h', 'ka_h', 'f_bioavail']:
                if key in params and isinstance(params[key], dict):
                    for api_name, spec in params[key].items():
                        if isinstance(spec, list):
                            if len(spec) > 0:
                                spec[0] = 0.0
                            if len(spec) > 2:
                                spec[2] = 0.0
                            if len(spec) > 1 and not spec[1]:
                                spec[1] = 'lognormal'
                            if len(spec) > 3 and not spec[3]:
                                spec[3] = 'lognormal'

        return params


@dataclass
class VariabilitySettings:
    """Per-domain toggles controlling subject variability sampling."""

    demographic: bool = True
    lung_regional: bool = True
    lung_generation: bool = True
    gi: bool = True
    pk: bool = True
    inhalation: bool = True

    @classmethod
    def resolve(
        cls,
        enable_variability: bool,
        overrides: Optional['VariabilitySettings | Mapping[str, Any]'] = None,
    ) -> 'VariabilitySettings':
        """Create a settings instance honoring global flag and user overrides."""

        if not enable_variability:
            return cls(False, False, False, False, False, False)

        if overrides is None:
            return cls()

        if isinstance(overrides, VariabilitySettings):
            return cls(
                demographic=bool(overrides.demographic),
                lung_regional=bool(overrides.lung_regional),
                lung_generation=bool(overrides.lung_generation),
                gi=bool(overrides.gi),
                pk=bool(overrides.pk),
                inhalation=bool(overrides.inhalation),
            )

        resolved = {field.name: True for field in fields(cls)}
        for key, value in overrides.items():
            if key is None:
                continue
            key_normalized = str(key).lower()
            for field in fields(cls):
                if field.name.lower() == key_normalized:
                    resolved[field.name] = bool(value)
                    break

        return cls(**resolved)

    def as_dict(self) -> Dict[str, bool]:
        """Return a dictionary representation of the toggle state."""
        return {field.name: bool(getattr(self, field.name)) for field in fields(self)}


InhalationProfile = InhalationManeuver

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

    # Results attached to this subject (populated after running models)
    results_deposition: Optional[dict] = Field(None, description="Deposition results wrapper", exclude=True)
    results_pk: Optional[dict] = Field(None, description="PK results wrapper", exclude=True)
    results_pd: Optional[dict] = Field(None, description="PD results wrapper", exclude=True)
    
    # Internal state
    is_loaded: bool = Field(False, exclude=True, description="Whether all components are loaded")

    @staticmethod
    def _get_variability_entry(variability_source, key):
        """Helper to access variability specification from dict-like or attribute-based objects."""
        if variability_source is None:
            return None
        if isinstance(variability_source, dict):
            return variability_source.get(key)
        return getattr(variability_source, key, None)

    @staticmethod
    def _sample_variability_factor(spec: Any, stage: str) -> float:
        """Sample a multiplicative factor from variability specification for inter/intra subject."""
        if not isinstance(spec, (list, tuple)):
            return 1.0

        if stage == 'inter':
            idx = 0
        else:
            idx = 2

        if len(spec) <= idx:
            return 1.0

        sigma = spec[idx] if idx < len(spec) else 0.0
        distribution = spec[idx + 1] if (idx + 1) < len(spec) else 'lognormal'

        try:
            sigma_val = float(sigma)
        except (TypeError, ValueError):
            sigma_val = 0.0

        if sigma_val <= 0.0:
            return 1.0

        dist_name = (distribution or 'lognormal').lower() if isinstance(distribution, str) else 'lognormal'

        if dist_name == 'lognormal':
            factor = float(np.random.lognormal(mean=0.0, sigma=sigma_val))
        elif dist_name == 'normal':
            factor = 1.0 + float(np.random.normal(loc=0.0, scale=sigma_val))
            if factor <= 0.0:
                factor = 1.0
        else:
            factor = 1.0

        return factor

    @staticmethod
    def _apply_factor(base_value, factor):
        """Apply multiplicative factor to a base value, preserving sign and avoiding non-positive results."""
        if base_value is None:
            return base_value
        try:
            new_value = base_value * factor
        except TypeError:
            return base_value
        if isinstance(new_value, (float, int)) and new_value > 0:
            return new_value
        return base_value

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
        lung_generation = LungGeneration.from_builtin(lung_geometry_population)
        lung_regional = LungRegional.from_lung_generation(lung_generation)
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
    
    def apply_variability(
        self,
        enable_variability: bool = True,
        api: Optional[str] = None,
        ref_lung_array: Optional[np.ndarray] = None,
        variability_settings: Optional['VariabilitySettings | Mapping[str, bool]'] = None,
    ) -> Tuple['Subject', 'Subject']:
        """Apply variability to create inter and intra subject instances, then compute final attributes.
        
        This demonstrates the universal structure for applying variability:
        - Inter subject: base_value * inter_factor
        - Intra subject: inter_value * (intra_factor)
        - Final attributes: computed using scaling and flow profile methods after variability
        
        Args:
            enable_variability: Whether to apply variability
            api: API name for dependent parameters
            ref_lung_array: Reference lung array for scaling (optional)
            variability_settings: Optional per-domain toggle mapping to enable or
                disable variability for specific physiology categories.
            
        Returns:
            Tuple of (inter_subject, intra_subject) with variability applied and final attributes computed
        """
        settings = VariabilitySettings.resolve(enable_variability, variability_settings)

        # Get variability factors for each domain
        demographic_var = (
            Demographic.get_variability(True, self.demographic.name if self.demographic else "healthy_reference")
            if self.demographic and settings.demographic
            else None
        )
        lung_regional_var = (
            LungRegional.get_variability(True)
            if self.lung_regional and settings.lung_regional
            else None
        )
        lung_generation_var = (
            LungGeneration.get_variability(True, self.lung_generation.name if self.lung_generation else "healthy")
            if self.lung_generation and settings.lung_generation
            else None
        )
        gi_var = (
            GI.get_variability(True, self.gi.name if self.gi else "default")
            if self.gi and settings.gi
            else None
        )
        pk_var = (
            PK.get_variability(True, self.pk.name if self.pk else "default")
            if self.pk and settings.pk
            else None
        )
        inhalation_var = (
            InhalationManeuver.get_variability(True, self.inhalation_maneuver.name if self.inhalation_maneuver else "pMDI_variable_trapezoid")
            if self.inhalation_maneuver and settings.inhalation
            else None
        )
        
        # Create inter subject by applying inter factors
        inter_subject = self.model_copy(deep=True)
        
        # Apply demographic variability for inter subject
        if inter_subject.demographic and demographic_var:
            demo_params = ['weight_kg', 'height_cm', 'frc_ml', 'frc_ref_ml', 'tidal_volume_ml', 'respiratory_rate_bpm', 'et_scale_factor']
            for param in demo_params:
                base_val = getattr(self.demographic, param, None)
                spec = self._get_variability_entry(demographic_var, param)
                if base_val is not None and spec is not None:
                    factor = self._sample_variability_factor(spec, 'inter')
                    setattr(inter_subject.demographic, param, self._apply_factor(base_val, factor))

        # Apply GI variability for inter subject
        if inter_subject.gi and api and gi_var:
            for param_group in ['gi_area', 'gi_tg', 'gi_vol']:
                base_values = getattr(self.gi, param_group, {}).get(api, []) if self.gi else []
                var_group = self._get_variability_entry(gi_var, param_group)
                var_specs = var_group.get(api, []) if isinstance(var_group, dict) else []

                if base_values and var_specs:
                    inter_values = []
                    for base_val, spec in zip(base_values, var_specs):
                        factor = self._sample_variability_factor(spec, 'inter')
                        inter_values.append(self._apply_factor(base_val, factor))
                    getattr(inter_subject.gi, param_group)[api] = inter_values

        # Apply PK variability for inter subject (limited to clearance and hepatic extraction only)
        if inter_subject.pk and api and pk_var:
            pk_params = ['clearance_L_h', 'hepatic_extraction']
            for param in pk_params:
                base_val = getattr(self.pk, param, None)
                spec_dict = self._get_variability_entry(pk_var, param)
                spec = spec_dict.get(api) if isinstance(spec_dict, dict) else None
                if base_val is not None and spec is not None:
                    factor = self._sample_variability_factor(spec, 'inter')
                    setattr(inter_subject.pk, param, self._apply_factor(base_val, factor))

        # Apply inhalation maneuver variability for inter subject
        if inter_subject.inhalation_maneuver and inhalation_var:
            inhalation_params = ['pifr_Lpm', 'rise_time_s', 'hold_time_s', 'breath_hold_time_s', 'exhalation_flow_Lpm', 'bolus_volume_ml', 'bolus_delay_s', 'et_scale_factor']
            for param in inhalation_params:
                base_val = getattr(self.inhalation_maneuver, param, None)
                spec = self._get_variability_entry(inhalation_var, param)
                if base_val is not None and spec is not None:
                    factor = self._sample_variability_factor(spec, 'inter')
                    setattr(inter_subject.inhalation_maneuver, param, self._apply_factor(base_val, factor))

        # Create intra subject by applying both inter and intra factors
        intra_subject = inter_subject.model_copy(deep=True)

        # Apply demographic variability for intra subject
        if intra_subject.demographic and demographic_var:
            demo_params = ['weight_kg', 'height_cm', 'frc_ml', 'frc_ref_ml', 'tidal_volume_ml', 'respiratory_rate_bpm', 'et_scale_factor']
            for param in demo_params:
                inter_val = getattr(inter_subject.demographic, param, None)
                spec = self._get_variability_entry(demographic_var, param)
                if inter_val is not None and spec is not None:
                    factor = self._sample_variability_factor(spec, 'intra')
                    setattr(intra_subject.demographic, param, self._apply_factor(inter_val, factor))

        # Apply GI variability for intra subject
        if intra_subject.gi and api and gi_var:
            for param_group in ['gi_area', 'gi_tg', 'gi_vol']:
                inter_values = getattr(inter_subject.gi, param_group, {}).get(api, []) if inter_subject.gi else []
                var_group = self._get_variability_entry(gi_var, param_group)
                var_specs = var_group.get(api, []) if isinstance(var_group, dict) else []

                if inter_values and var_specs:
                    intra_values = []
                    for inter_val, spec in zip(inter_values, var_specs):
                        factor = self._sample_variability_factor(spec, 'intra')
                        intra_values.append(self._apply_factor(inter_val, factor))
                    getattr(intra_subject.gi, param_group)[api] = intra_values

        # Apply PK variability for intra subject (limited to clearance and hepatic extraction only)
        if intra_subject.pk and api and pk_var:
            pk_params = ['clearance_L_h', 'hepatic_extraction']
            for param in pk_params:
                inter_val = getattr(inter_subject.pk, param, None)
                spec_dict = self._get_variability_entry(pk_var, param)
                spec = spec_dict.get(api) if isinstance(spec_dict, dict) else None
                if inter_val is not None and spec is not None:
                    factor = self._sample_variability_factor(spec, 'intra')
                    setattr(intra_subject.pk, param, self._apply_factor(inter_val, factor))

        # Apply inhalation maneuver variability for intra subject
        if intra_subject.inhalation_maneuver and inhalation_var:
            inhalation_params = ['pifr_Lpm', 'rise_time_s', 'hold_time_s', 'breath_hold_time_s', 'exhalation_flow_Lpm', 'bolus_volume_ml', 'bolus_delay_s', 'et_scale_factor']
            for param in inhalation_params:
                inter_val = getattr(inter_subject.inhalation_maneuver, param, None)
                spec = self._get_variability_entry(inhalation_var, param)
                if inter_val is not None and spec is not None:
                    factor = self._sample_variability_factor(spec, 'intra')
                    setattr(intra_subject.inhalation_maneuver, param, self._apply_factor(inter_val, factor))
        
        # Compute final attributes after variability is applied

        def _refresh(subject_obj: 'Subject') -> None:
            if subject_obj.lung_generation:
                subject_obj.lung_regional = LungRegional.from_lung_generation(subject_obj.lung_generation)
                if subject_obj.demographic and subject_obj.demographic.frc_ml:
                    subject_obj._final_lung_generation = subject_obj.lung_generation.compute_scaled_geometry(subject_obj.demographic.frc_ml)
                else:
                    subject_obj._final_lung_generation = None
                subject_obj._final_lung_regional = subject_obj.lung_generation.compute_regional_parameters()
            else:
                subject_obj._final_lung_generation = None
                subject_obj._final_lung_regional = None

        # Scale inhaled_volume proportionally to FRC changes for inter subject
        if inter_subject.inhalation_maneuver:
            inter_subject._final_flow_profile = inter_subject.inhalation_maneuver.calculate_inhalation_maneuver_flow_profile()

        _refresh(intra_subject)

        # Flow profile for intra subject
        if intra_subject.inhalation_maneuver:
            intra_subject._final_flow_profile = intra_subject.inhalation_maneuver.calculate_inhalation_maneuver_flow_profile()

        return inter_subject, intra_subject
    
    def get_final_lung_regional(self) -> Optional[Dict[str, Any]]:
        """Get final lung regional aggregates after variability/scaling."""
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
    
    def get_final_values(
        self,
        apply_variability: bool = False,
        api_name: Optional[str] = None,
        variability_settings: Optional['VariabilitySettings | Mapping[str, bool]'] = None,
    ):
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
            variability_settings: Optional per-domain toggle mapping used when
                variability is applied. Ignored if apply_variability is False.
            
        Returns:
            Subject instance with computed attributes (scaled_lung_geometry, flow_profile, etc.)
        """
        final_subject = self
        
        # Apply variability if requested
        if apply_variability:
            inter_subject, intra_subject = self.apply_variability(
                enable_variability=True,
                api=api_name or "BD",
                variability_settings=variability_settings,
            )
            final_subject = intra_subject  # Use intra subject as final
        
        # Get scaled lung geometry (always apply lung scaling)
        if final_subject.lung_generation:
            frc_ml = final_subject.demographic.frc_ml if final_subject.demographic else None
            frc_ml = frc_ml if frc_ml is not None else 3000.0
            scaled_lung_geometry = final_subject.lung_generation.compute_scaled_geometry(frc_ml)
            final_subject.lung_regional = LungRegional.from_lung_generation(final_subject.lung_generation)
            regional_params = final_subject.lung_generation.compute_regional_parameters()
        else:
            scaled_lung_geometry = None
            regional_params = None
        
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
        final_subject._final_lung_generation = scaled_lung_geometry
        final_subject._final_lung_regional = regional_params
        final_subject._flow_profile = flow_profile
        final_subject._final_flow_profile = flow_profile
        final_subject._scaled_inhaled_volume_L = inhaled_volume_L

        return final_subject

    # -----------------------------------------------------------------
    # Results attachment helpers
    # -----------------------------------------------------------------
    def attach_deposition_results(self, deposition_result, api: 'API') -> 'Subject':
        """Attach deposition results to the subject in a unit-aware wrapper."""
        try:
            from ..data_structures import Results_Deposition
            wrapper = Results_Deposition.from_deposition_result(deposition_result, api.molecular_weight)
            self.results_deposition = {
                'wrapper': wrapper,
                'raw': deposition_result
            }
        except Exception as e:
            # Store raw if wrapper creation fails
            self.results_deposition = {
                'error': str(e),
                'raw': deposition_result
            }
        return self

    def attach_pk_results(self, orchestrator_result, api: 'API') -> 'Subject':
        """Attach PK results from PBPK orchestrator to the subject."""
        try:
            from ..data_structures import Results_PK
            wrapper = Results_PK.from_orchestrator(orchestrator_result, api.molecular_weight)
            self.results_pk = {
                'wrapper': wrapper,
                'raw': orchestrator_result
            }
        except Exception as e:
            self.results_pk = {
                'error': str(e),
                'raw': orchestrator_result
            }
        return self


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
