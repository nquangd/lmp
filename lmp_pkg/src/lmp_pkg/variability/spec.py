"""Variability specification classes."""

from __future__ import annotations
from typing import Dict, Any, List, Literal, Optional, Union
from pydantic import BaseModel, Field, field_validator
import math


class DistributionSpec(BaseModel):
    """Specification for a single parameter's variability distribution."""
    
    dist: Literal["lognormal", "normal", "normal_absolute", "uniform"] = Field(
        ..., description="Distribution type"
    )
    
    # Lognormal parameters
    sigma_log: Optional[float] = Field(None, ge=0, description="Lognormal sigma parameter")
    gcv: Optional[float] = Field(None, ge=0, description="Geometric coefficient of variation")
    
    # Normal parameters (multiplicative, mean=1)
    sd: Optional[float] = Field(None, ge=0, description="Standard deviation for normal distribution")
    
    # Normal absolute parameters
    mean: Optional[float] = Field(None, description="Mean for absolute normal distribution")
    
    # Uniform parameters
    min: Optional[float] = Field(None, description="Minimum value for uniform distribution")
    max: Optional[float] = Field(None, description="Maximum value for uniform distribution")
    
    # Metadata
    mean_from: Literal["catalog", "config"] = Field("catalog", description="Source of mean value")
    mode: Literal["multiplicative", "absolute"] = Field("multiplicative", description="Application mode")
    
    @field_validator("sigma_log", "gcv")
    @classmethod 
    def validate_positive_params(cls, v: Optional[float]) -> Optional[float]:
        if v is not None and v < 0:
            raise ValueError("Variability parameters must be non-negative")
        return v
    
    def model_post_init(self, __context: Any) -> None:
        """Validate distribution parameter consistency after initialization."""
        if self.dist == "lognormal":
            if self.sigma_log is None and self.gcv is None:
                raise ValueError("Lognormal distribution requires either sigma_log or gcv")
            if self.sigma_log is not None and self.gcv is not None:
                raise ValueError("Specify either sigma_log or gcv, not both")
                
        elif self.dist == "normal":
            if self.sd is None:
                raise ValueError("Normal distribution requires sd parameter")
                
        elif self.dist == "normal_absolute":
            if self.mean is None or self.sd is None:
                raise ValueError("Normal absolute distribution requires mean and sd")
                
        elif self.dist == "uniform":
            if self.min is None or self.max is None:
                raise ValueError("Uniform distribution requires min and max")
            if self.min >= self.max:
                raise ValueError("Uniform distribution min must be less than max")
    
    def get_effective_sigma_log(self) -> float:
        """Get effective sigma_log parameter for lognormal distribution."""
        if self.dist != "lognormal":
            raise ValueError("Only applicable to lognormal distributions")
            
        if self.sigma_log is not None:
            return self.sigma_log
        elif self.gcv is not None:
            return self.convert_gcv_to_sigma_log(self.gcv)
        else:
            raise ValueError("No sigma_log or gcv specified")
    
    @staticmethod
    def convert_gcv_to_sigma_log(gcv: float) -> float:
        """Convert geometric coefficient of variation to lognormal sigma."""
        if gcv <= 0:
            return 0.0
        return math.sqrt(math.log(gcv**2 + 1))


class LayerSpec(BaseModel):
    """Specification for a variability layer (Inter or Intra)."""
    
    inhalation: Dict[str, DistributionSpec] = Field(default_factory=dict)
    pk: Dict[str, Dict[str, DistributionSpec]] = Field(default_factory=dict)  # param -> group -> spec
    physiology: Dict[str, DistributionSpec] = Field(default_factory=dict)
    product: Dict[str, DistributionSpec] = Field(default_factory=dict)  # Future use
    api: Dict[str, DistributionSpec] = Field(default_factory=dict)      # Future use


class VariabilitySpec(BaseModel):
    """Complete variability specification with Inter/Intra layers."""
    
    layers: List[Literal["inter", "intra"]] = Field(default_factory=lambda: ["inter", "intra"])
    inter: LayerSpec = Field(default_factory=LayerSpec)
    intra: LayerSpec = Field(default_factory=LayerSpec)
    
    @classmethod
    def from_original_format(cls) -> "VariabilitySpec":
        """Create VariabilitySpec matching the original code's parameters."""
        
        # Original inhalation parameters (lognormal factors, mean=1)
        inhalation_params = {
            "pifr_Lpm": DistributionSpec(dist="lognormal", gcv=0.3),
            "rise_time_s": DistributionSpec(dist="lognormal", gcv=0.1),
            "hold_time_s": DistributionSpec(dist="normal", sd=0.0),
            "breath_hold_time_s": DistributionSpec(dist="normal", sd=0.0),
            "exhalation_flow_Lpm": DistributionSpec(dist="normal", sd=0.0),
            "bolus_volume_ml": DistributionSpec(dist="normal", sd=0.0),
            "bolus_delay_s": DistributionSpec(dist="normal", sd=0.0)
        }
        
        # ET uses normal distribution with different values for inter/intra
        inhalation_params_inter = dict(inhalation_params)
        inhalation_params_inter["ET"] = DistributionSpec(dist="normal", sd=0.235)
        
        inhalation_params_intra = dict(inhalation_params) 
        inhalation_params_intra["ET"] = DistributionSpec(dist="normal", sd=0.15)
        
        # Original PK parameters (lognormal factors by API group)
        pk_gcv_inter = {
            'CL': {'BD': 0.26, 'GP': 0.5, 'FF': 0.3},
            'Eh': {'BD': 0.114, 'GP': 0.5, 'FF': 0.4}
        }
        
        pk_gcv_intra = {
            'CL': {'BD': 0.05, 'GP': 0.4, 'FF': 0.1},
            'Eh': {'BD': 0.10, 'GP': 0.4, 'FF': 0.2}
        }
        
        # Convert to DistributionSpec format
        pk_inter = {
            param: {
                group: DistributionSpec(dist="lognormal", gcv=gcv)
                for group, gcv in groups.items()
            }
            for param, groups in pk_gcv_inter.items()
        }
        
        pk_intra = {
            param: {
                group: DistributionSpec(dist="lognormal", gcv=gcv) 
                for group, gcv in groups.items()
            }
            for param, groups in pk_gcv_intra.items()
        }
        
        # Physiology parameters (absolute values)
        physiology = {
            "FRC": DistributionSpec(
                dist="normal_absolute",
                mean=3300.0,
                sd=600.0,
                mode="absolute"
            )
        }
        
        # Product variability parameters (for deposition)
        product_params = {
            "emitted_dose_factor": DistributionSpec(
                dist="lognormal",
                gcv=0.05,
                mode="multiplicative"
            ),
            "particle_size_factor": DistributionSpec(
                dist="lognormal", 
                gcv=0.10,
                mode="multiplicative"
            )
        }
        
        # Maneuver variability parameters (for deposition)
        maneuver_params = {
            "flow_rate_factor": DistributionSpec(
                dist="lognormal",
                gcv=0.15,
                mode="multiplicative"
            ),
            "volume_factor": DistributionSpec(
                dist="lognormal",
                gcv=0.08,
                mode="multiplicative" 
            )
        }
        
        return cls(
            layers=["inter", "intra"],
            inter=LayerSpec(
                inhalation=inhalation_params_inter,
                pk=pk_inter,
                physiology=physiology,
                product=product_params,
                api=maneuver_params  # Using api field for maneuver for now
            ),
            intra=LayerSpec(
                inhalation=inhalation_params_intra,
                pk=pk_intra,
                physiology={},  # Intra physiology not varied in original
                product={},     # No intra product variability by default
                api={}          # No intra maneuver variability by default
            )
        )
    
    def disable_all_variability(self) -> "VariabilitySpec":
        """Return a copy with all variability disabled (sigma/sd = 0)."""
        def zero_spec(spec: DistributionSpec) -> DistributionSpec:
            spec_dict = spec.model_dump()
            if spec.dist == "lognormal":
                spec_dict.update({"sigma_log": 0.0, "gcv": None})
            elif spec.dist in ["normal", "normal_absolute"]:
                spec_dict["sd"] = 0.0
            elif spec.dist == "uniform":
                # For uniform, set min=max=mean
                mean_val = spec_dict.get("mean", 1.0)
                spec_dict.update({"min": mean_val, "max": mean_val})
            return DistributionSpec.model_validate(spec_dict)
        
        def zero_layer(layer: LayerSpec) -> LayerSpec:
            return LayerSpec(
                inhalation={k: zero_spec(v) for k, v in layer.inhalation.items()},
                pk={param: {group: zero_spec(spec) for group, spec in groups.items()} 
                    for param, groups in layer.pk.items()},
                physiology={k: zero_spec(v) for k, v in layer.physiology.items()},
                product={k: zero_spec(v) for k, v in layer.product.items()},
                api={k: zero_spec(v) for k, v in layer.api.items()}
            )
        
        return VariabilitySpec(
            layers=self.layers,
            inter=zero_layer(self.inter),
            intra=zero_layer(self.intra)
        )