"""Base interfaces for the new clean PBBM architecture."""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Set, List, Union
from dataclasses import dataclass
from enum import Enum
import numpy as np


class ModelType(Enum):
    """Types of PBBM model components."""
    LUNG_PBBM = "lung_pbbm"
    GI_ABSORPTION = "gi_absorption"  
    SYSTEMIC_PK = "systemic_pk"
    DEPOSITION = "deposition"
    EFFICACY = "efficacy"


@dataclass
class ModelDependency:
    """Specification for model dependencies."""
    name: str                    # Dependency identifier
    data_type: str              # Type of data expected
    required: bool = True       # Whether dependency is mandatory
    description: str = ""       # Human-readable description


@dataclass
class ModelOutput:
    """Specification for model outputs."""
    name: str                   # Output identifier  
    data_type: str              # Type of data provided
    description: str = ""       # Human-readable description


class PBBMModelComponent(ABC):
    """Base interface for all PBBM model components.
    
    This interface ensures all model components can be:
    1. Composed together in flexible ways
    2. Validated for compatibility
    3. Tested in isolation
    4. Extended by third-party developers
    """
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Unique identifier for this model component."""
        pass
    
    @property
    @abstractmethod
    def model_type(self) -> ModelType:
        """Type category of this model."""
        pass
    
    @property
    @abstractmethod
    def requires(self) -> List[ModelDependency]:
        """Dependencies this model requires from other models."""
        pass
    
    @property
    @abstractmethod
    def provides(self) -> List[ModelOutput]:
        """Outputs this model provides to other models."""
        pass
    
    @abstractmethod
    def get_parameter_requirements(self) -> Dict[str, type]:
        """Required parameters and their expected types.
        
        Returns:
            Dictionary mapping parameter names to their expected types
        """
        pass
    
    @abstractmethod
    def validate_parameters(self, params: Dict[str, Any]) -> bool:
        """Validate that provided parameters meet requirements.
        
        Args:
            params: Parameter dictionary to validate
            
        Returns:
            True if parameters are valid, raises exception otherwise
        """
        pass
    
    @abstractmethod
    def get_state_size(self, **config) -> int:
        """Number of state variables this model contributes.
        
        Args:
            **config: Model-specific configuration
            
        Returns:
            Number of state variables
        """
        pass
    
    @abstractmethod
    def initialize_state(self, **config) -> np.ndarray:
        """Initialize state vector for this model component.
        
        Args:
            **config: Model-specific configuration
            
        Returns:
            Initial state vector for this model's variables
        """
        pass
    
    @abstractmethod
    def compute_derivatives(self, 
                          t: float, 
                          state: np.ndarray,
                          dependencies: Dict[str, Any],
                          **params) -> np.ndarray:
        """Compute derivatives for this model's state variables.
        
        Args:
            t: Current time [s]
            state: Current state vector for this model
            dependencies: Data from other models this model depends on
            **params: Model parameters
            
        Returns:
            Derivative vector for this model's state variables
        """
        pass
    
    @abstractmethod
    def extract_outputs(self, 
                       t: float, 
                       state: np.ndarray,
                       **params) -> Dict[str, Any]:
        """Extract outputs for other models to use.
        
        Args:
            t: Current time [s]
            state: Current state vector for this model
            **params: Model parameters
            
        Returns:
            Dictionary of outputs with keys matching 'provides' specification
        """
        pass
    
    def get_state_names(self) -> List[str]:
        """Get names of state variables (optional, for debugging/plotting).
        
        Returns:
            List of state variable names
        """
        return [f"{self.name}_state_{i}" for i in range(self.get_state_size())]
    
    def supports_numba(self) -> bool:
        """Whether this model supports numba JIT compilation.
        
        Returns:
            True if model can be JIT compiled with numba
        """
        return False


class ModelComposer(ABC):
    """Base interface for model composition strategies."""
    
    @abstractmethod
    def add_model(self, model: PBBMModelComponent, **config) -> None:
        """Add a model component to the composition.
        
        Args:
            model: Model component to add
            **config: Model-specific configuration
        """
        pass
    
    @abstractmethod
    def validate_composition(self) -> bool:
        """Validate that all model dependencies are satisfied.
        
        Returns:
            True if composition is valid, raises exception otherwise
        """
        pass
    
    @abstractmethod
    def build_combined_system(self) -> 'ComposedPBBMModel':
        """Build the final combined model system.
        
        Returns:
            Composed model ready for simulation
        """
        pass
    
    def get_dependency_graph(self) -> Dict[str, List[str]]:
        """Get dependency graph for visualization/debugging.
        
        Returns:
            Dictionary mapping model names to their dependencies
        """
        pass


class ComposedPBBMModel(ABC):
    """Interface for a fully composed PBBM model system."""
    
    @abstractmethod
    def simulate(self, 
                time_span: tuple, 
                initial_conditions: Dict[str, Any] = None,
                **solver_options) -> 'PBBMResults':
        """Run simulation with the composed model.
        
        Args:
            time_span: (start_time, end_time) in seconds
            initial_conditions: Override default initial conditions
            **solver_options: Options for ODE solver
            
        Returns:
            Comprehensive simulation results
        """
        pass
    
    @abstractmethod
    def get_model_components(self) -> List[PBBMModelComponent]:
        """Get list of component models in this composition.
        
        Returns:
            List of model components
        """
        pass
    
    @abstractmethod 
    def get_total_state_size(self) -> int:
        """Get total number of state variables in composed system.
        
        Returns:
            Total state vector size
        """
        pass


class DataStructure(ABC):
    """Base interface for PBBM data structures."""
    
    @abstractmethod
    def validate(self) -> bool:
        """Validate data structure consistency."""
        pass
    
    @abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        pass
    
    @abstractmethod
    def from_dict(self, data: Dict[str, Any]) -> 'DataStructure':
        """Create from dictionary format."""
        pass


class ParameterTransform(ABC):
    """Base interface for parameter transformations."""
    
    @abstractmethod
    def transform(self, 
                 raw_params: Dict[str, Any],
                 context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Transform raw parameters to model-ready format.
        
        Args:
            raw_params: Raw input parameters
            context: Additional context for transformation
            
        Returns:
            Transformed parameters ready for model use
        """
        pass
    
    @abstractmethod
    def inverse_transform(self, 
                         model_params: Dict[str, Any],
                         context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Transform model parameters back to raw format.
        
        Args:
            model_params: Model-ready parameters
            context: Additional context for transformation
            
        Returns:
            Raw parameters in original format
        """
        pass
    
    @abstractmethod
    def get_transform_requirements(self) -> Dict[str, type]:
        """Get required parameters for this transform.
        
        Returns:
            Dictionary mapping parameter names to types
        """
        pass


# Convenience type aliases
ModelRegistry = Dict[str, PBBMModelComponent]
DependencyGraph = Dict[str, List[str]]
StateIndexMapping = Dict[str, slice]