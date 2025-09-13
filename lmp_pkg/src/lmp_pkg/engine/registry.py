"""Model registry for dynamic discovery and instantiation."""

from __future__ import annotations
from typing import Dict, List, Type, Set, Optional
import importlib
from dataclasses import dataclass

from ..contracts.stage import Stage


@dataclass
class ModelInfo:
    """Information about a registered model."""
    name: str
    family: str  # "deposition", "lung_pbbm", "systemic_pk"
    model_class: Type[Stage]
    provides: Set[str]
    requires: Set[str]


class ModelRegistry:
    """Registry for discovering and instantiating models."""
    
    def __init__(self):
        self._models: Dict[str, ModelInfo] = {}
        self._by_family: Dict[str, List[str]] = {}
        self._register_builtin_models()
    
    def register(self, family: str, name: str, model_class: Type[Stage]) -> None:
        """Register a model.
        
        Args:
            family: Model family ("deposition", "lung_pbbm", "systemic_pk")  
            name: Model name within family
            model_class: Model class implementing Stage protocol
        """
        # Create instance to get provides/requires
        instance = model_class()
        
        model_info = ModelInfo(
            name=name,
            family=family,
            model_class=model_class,
            provides=instance.provides,
            requires=instance.requires
        )
        
        full_name = f"{family}.{name}"
        self._models[full_name] = model_info
        
        if family not in self._by_family:
            self._by_family[family] = []
        if name not in self._by_family[family]:
            self._by_family[family].append(name)
    
    def get_model(self, family: str, name: str) -> Stage:
        """Get model instance.
        
        Args:
            family: Model family
            name: Model name
            
        Returns:
            Model instance
            
        Raises:
            KeyError: If model not found
        """
        full_name = f"{family}.{name}"
        if full_name not in self._models:
            raise KeyError(f"Model not found: {full_name}")
        
        model_info = self._models[full_name]
        return model_info.model_class()
    
    def list_models(self, family: Optional[str] = None) -> Dict[str, List[str]]:
        """List available models.
        
        Args:
            family: Optional family filter
            
        Returns:
            Dictionary mapping families to model names
        """
        if family is not None:
            if family in self._by_family:
                return {family: self._by_family[family].copy()}
            else:
                return {family: []}
        
        return {k: v.copy() for k, v in self._by_family.items()}
    
    def get_model_info(self, family: str, name: str) -> ModelInfo:
        """Get model information.
        
        Args:
            family: Model family
            name: Model name
            
        Returns:
            Model information
            
        Raises:
            KeyError: If model not found
        """
        full_name = f"{family}.{name}"
        if full_name not in self._models:
            raise KeyError(f"Model not found: {full_name}")
        
        return self._models[full_name]
    
    def get_default_model(self, family: str) -> str:
        """Get default model name for a family.
        
        Args:
            family: Model family
            
        Returns:
            Default model name (currently "null" for all families)
        """
        if family in self._by_family and "null" in self._by_family[family]:
            return "null"
        elif family in self._by_family and self._by_family[family]:
            return self._by_family[family][0]  # First available
        else:
            raise KeyError(f"No models available for family: {family}")
    
    def _register_builtin_models(self) -> None:
        """Register built-in null models."""
        try:
            # Import and register deposition models
            from ..models.deposition.null import NullDeposition
            from ..models.deposition.legacy_bins import LegacyDeposition
            self.register("deposition", "null", NullDeposition)
            self.register("deposition", "legacy_bins", LegacyDeposition)
            
            # Import and register lung PBPK models  
            from ..models.individual.lung_pbbm.null import NullPBPK
            from ..models.individual.lung_pbbm.modular_pbbm import ModularLungPBBM
            from ..models.individual.lung_pbbm.chained_model import ChainedPBBMModel
            self.register("lung_pbbm", "null", NullPBPK)
            self.register("lung_pbbm", "modular", ModularLungPBBM)
            self.register("lung_pbbm", "chained_full", lambda: ChainedPBBMModel(["lung", "gi", "pk"]))
            self.register("lung_pbbm", "chained_lung_pk", lambda: ChainedPBBMModel(["lung", "pk"]))
            self.register("lung_pbbm", "chained_gi_pk", lambda: ChainedPBBMModel(["gi", "pk"]))
            self.register("lung_pbbm", "chained_pk_only", lambda: ChainedPBBMModel(["pk"]))
            
            # Import and register systemic PK models
            from ..models.individual.systemic_pk.null import NullPK
            from ..models.individual.systemic_pk.pk_1c import OneCPKModel
            from ..models.individual.systemic_pk.pk_2c import TwoCPKModel
            from ..models.individual.systemic_pk.pk_3c import ThreeCompartmentPK as ThreeCPKModel
            self.register("systemic_pk", "null", NullPK)
            self.register("systemic_pk", "pk_1c", OneCPKModel)
            self.register("systemic_pk", "pk_2c", TwoCPKModel)  
            self.register("systemic_pk", "pk_3c", ThreeCPKModel)
            
            # Import and register efficacy models
            from ..models.efficacy.fev1_model import FEV1EfficacyModel
            self.register("efficacy", "fev1_copd", FEV1EfficacyModel)
            
        except ImportError as e:
            # Log warning but don't fail - models might not be implemented yet
            import warnings
            warnings.warn(f"Could not register built-in models: {e}")


# Global registry instance
_registry = ModelRegistry()

def get_registry() -> ModelRegistry:
    """Get the global model registry."""
    return _registry