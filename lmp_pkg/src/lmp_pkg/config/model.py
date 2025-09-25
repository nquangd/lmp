"""Configuration data models."""

from __future__ import annotations
from typing import Any, Dict, List, Optional, Union
from pathlib import Path
from pydantic import BaseModel, Field, field_validator

from ..variability.spec import VariabilitySpec


class PopulationVariabilityConfig(BaseModel):
    """Per-domain toggles for subject variability."""

    demographic: Optional[bool] = None
    lung_regional: Optional[bool] = None
    lung_generation: Optional[bool] = None
    gi: Optional[bool] = None
    pk: Optional[bool] = None
    inhalation: Optional[bool] = None

    def as_overrides(self) -> Dict[str, bool]:
        """Return a mapping of explicitly set overrides."""
        return {
            key: value
            for key, value in self.model_dump().items()
            if value is not None
        }


class StudyConfig(BaseModel):
    """Clinical study settings."""

    study_type: str = "bioequivalence"
    design: str = "parallel"
    n_subjects: int = Field(6, gt=0, description="Number of subjects in study")
    population: Optional[str] = Field(None, description="Population identifier for subject selection")
    charcoal_block: bool = Field(False, description="Enable GI charcoal block (zero GI absorption)")



class SolverConfig(BaseModel):
    """ODE solver configuration."""
    
    method: str = "BDF"
    rtol: float = 1e-6
    atol: float = 1e-9
    max_step: float = 1.0
    
    @field_validator("method")
    @classmethod
    def validate_method(cls, v: str) -> str:
        valid_methods = {"RK45", "BDF", "Radau", "DOP853", "LSODA"}
        if v not in valid_methods:
            raise ValueError(f"method must be one of {valid_methods}")
        return v


class DepositionConfig(BaseModel):
    """Deposition model configuration."""
    
    model: str = "null"
    particle_grid: str = "medium"
    params: Dict[str, float] = Field(default_factory=dict)


class PBBMConfig(BaseModel):
    """Lung PBPK model configuration."""
    
    model: str = "null"
    epi_layers: List[int] = Field(default_factory=lambda: [1, 1, 1, 1])
    solver: SolverConfig = Field(default_factory=SolverConfig)
    params: Dict[str, float] = Field(default_factory=dict)
    charcoal_block: bool = False
    suppress_et_absorption: bool = False
    
    @field_validator("epi_layers")
    @classmethod
    def validate_epi_layers(cls, v: List[int]) -> List[int]:
        if not v or any(x <= 0 for x in v):
            raise ValueError("epi_layers must contain positive integers")
        return v


class PKConfig(BaseModel):
    """Systemic PK model configuration."""
    
    model: str = "null"
    params: Dict[str, float] = Field(default_factory=dict)


class AnalysisConfig(BaseModel):
    """Analysis configuration."""
    
    bioequivalence: bool = True
    custom_metrics: List[str] = Field(default_factory=list)


class RunConfig(BaseModel):
    """Run execution configuration."""
    
    stages: List[str] = Field(default_factory=lambda: ["cfd", "deposition", "pbbm", "pk"])
    workflow_name: Optional[str] = Field(default=None, description="Preconfigured workflow name")
    seed: int = 123
    threads: int = 1
    enable_numba: bool = False
    artifact_dir: str = "results"
    n_replicates: int = Field(default=1, ge=1, description="Number of replicates for variability sampling")
    stage_overrides: Dict[str, Any] = Field(default_factory=dict, description="Per-stage override payloads")

    @field_validator("stages")
    @classmethod
    def validate_stages(cls, v: List[str]) -> List[str]:
        valid_stages = {"cfd", "deposition", "pbbm", "pk", "iv_pk", "gi_pk", "analysis"}
        invalid = set(v) - valid_stages
        if invalid:
            raise ValueError(f"Invalid stages: {invalid}")
        return v
        
    @field_validator("threads")
    @classmethod
    def validate_threads(cls, v: int) -> int:
        if v < 1:
            raise ValueError("threads must be positive")
        return v


class EntityRef(BaseModel):
    """Reference to a catalog entity with optional overrides."""
    
    ref: Optional[str] = None
    overrides: Dict[str, Any] = Field(default_factory=dict)
    
    def __init__(self, **data):
        # Handle both ref-only and inline-only cases
        if len(data) == 1 and "ref" not in data and "overrides" not in data:
            # Single string argument -> treat as ref
            key = next(iter(data.keys()))
            if isinstance(data[key], str):
                data = {"ref": data[key]}
        # Handle TOML "inline" alias
        if "inline" in data:
            data["overrides"] = data.pop("inline")
        super().__init__(**data)


class AppConfig(BaseModel):
    """Complete application configuration."""
    
    run: RunConfig = Field(default_factory=RunConfig)
    study: StudyConfig = Field(default_factory=StudyConfig)
    deposition: DepositionConfig = Field(default_factory=DepositionConfig)
    pbbm: PBBMConfig = Field(default_factory=PBBMConfig)
    pk: PKConfig = Field(default_factory=PKConfig)
    analysis: AnalysisConfig = Field(default_factory=AnalysisConfig)
    
    # Entity references
    subject: EntityRef = Field(default_factory=lambda: EntityRef(ref="healthy_reference"))
    api: EntityRef = Field(default_factory=lambda: EntityRef(ref="BD"))
    product: EntityRef = Field(default_factory=lambda: EntityRef(ref="reference_product"))
    maneuver: EntityRef = Field(default_factory=lambda: EntityRef(ref="pMDI_variable_trapezoid"))
    
    # Variability specification
    variability: Optional[VariabilitySpec] = Field(default=None, description="Variability configuration for Inter/Intra sampling")
    population_variability: Optional[PopulationVariabilityConfig] = Field(
        default=None,
        description="Per-domain toggles for subject variability in stage pipeline workflows",
    )
    
    def get_effective_variability(self) -> VariabilitySpec:
        """Get effective variability specification, with defaults if none specified."""
        if self.variability is not None:
            return self.variability
        elif self.run.n_replicates > 1:
            # Use default variability from original code when replicates > 1
            return VariabilitySpec.from_original_format()
        else:
            # No variability when n_replicates = 1
            return VariabilitySpec.from_original_format().disable_all_variability()
    
    def model_dump_toml(self) -> str:
        """Export configuration as TOML string."""
        try:
            import tomli_w
            return tomli_w.dumps(self.model_dump())
        except ImportError:
            raise ImportError("tomli_w required for TOML export")
    
    @classmethod
    def from_toml_file(cls, path: Union[Path, str]) -> "AppConfig":
        """Load configuration from TOML file."""
        try:
            import tomllib
        except ImportError:
            import tomli as tomllib
            
        with open(path, "rb") as f:
            data = tomllib.load(f)
        return cls.model_validate(data)
