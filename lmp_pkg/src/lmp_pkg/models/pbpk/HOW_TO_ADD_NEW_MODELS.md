# How to Add New Models to the LMP PBBM Architecture

This guide explains how to add new model components to the LMP PBBM architecture. The modular design makes it straightforward to add new lung models, PK models, or entirely new model types.

## 🏗️ Architecture Overview

The new architecture uses interface-based design where all models implement the `PBBMModelComponent` interface:

```python
class PBBMModelComponent(ABC):
    @property
    @abstractmethod
    def requires(self) -> List[ModelDependency]: pass
    
    @property
    @abstractmethod
    def provides(self) -> List[ModelOutput]: pass
    
    @abstractmethod
    def compute_derivatives(self, t, state, dependencies, **params): pass
    
    @abstractmethod
    def extract_outputs(self, t, state, **params): pass
    
    @abstractmethod
    def initialize_state(self, **params): pass
```

## 📋 Step-by-Step Guide

### Step 1: Define Your Model Parameters

Create a dataclass for your model parameters:

```python
from dataclasses import dataclass
import numpy as np

@dataclass
class MyNewModelParams:
    """Parameters for my new model."""
    # Required parameters (no defaults)
    volume_L: float
    clearance_L_per_h: float
    
    # Optional parameters (with defaults)
    bioavailability: float = 1.0
    absorption_rate_per_h: float = 1.0
```

### Step 2: Implement the Model Class

```python
from typing import Dict, Any, List
from ..base import PBBMModelComponent, ModelType, ModelDependency, ModelOutput

class MyNewModel(PBBMModelComponent):
    """My new model implementation."""
    
    def __init__(self, params: MyNewModelParams):
        self.params = params
    
    @property
    def name(self) -> str:
        return "my_new_model"
    
    @property
    def model_type(self) -> ModelType:
        return ModelType.SYSTEMIC_PK  # or LUNG_PBBM, GI_ABSORPTION, etc.
    
    @property
    def requires(self) -> List[ModelDependency]:
        """Define what this model needs from other models."""
        return [
            ModelDependency(
                name="drug_input_rate",  # What you need
                data_type="pmol/s",
                description="Drug input rate into the system",
                required=True  # or False for optional
            )
        ]
    
    @property
    def provides(self) -> List[ModelOutput]:
        """Define what this model provides to other models."""
        return [
            ModelOutput(
                name="plasma_concentration",  # What you provide
                data_type="pmol/mL",
                description="Plasma drug concentration"
            ),
            ModelOutput(
                name="elimination_rate",
                data_type="pmol/s", 
                description="Drug elimination rate"
            )
        ]
    
    def get_state_size(self, **params) -> int:
        """Return the size of this model's state vector."""
        return 1  # Number of compartments/state variables
    
    def initialize_state(self, **params) -> np.ndarray:
        """Initialize the model's state vector."""
        return np.array([0.0])  # Initial conditions
    
    def compute_derivatives(self, t: float, state: np.ndarray, 
                          dependencies: Dict[str, Any], **params) -> np.ndarray:
        """Compute time derivatives of the state vector."""
        # Extract current state
        amount_pmol = state[0]
        
        # Get dependencies from other models
        input_rate = dependencies.get("drug_input_rate", 0.0)
        
        # Calculate elimination
        concentration = amount_pmol / self.params.volume_L
        elimination_rate = concentration * self.params.clearance_L_per_h / 3600.0  # Convert to /s
        
        # Calculate derivative
        damount_dt = input_rate - elimination_rate
        
        return np.array([damount_dt])
    
    def extract_outputs(self, t: float, state: np.ndarray, **params) -> Dict[str, Any]:
        """Extract outputs that other models can use."""
        amount_pmol = state[0]
        concentration = amount_pmol / self.params.volume_L
        elimination_rate = concentration * self.params.clearance_L_per_h / 3600.0
        
        return {
            "plasma_concentration": concentration,
            "elimination_rate": elimination_rate
        }
    
    def validate_parameters(self, params: Dict[str, Any]) -> bool:
        """Validate parameter values (optional but recommended)."""
        if self.params.volume_L <= 0:
            raise ValueError("Volume must be positive")
        if self.params.clearance_L_per_h <= 0:
            raise ValueError("Clearance must be positive")
        return True
    
    def get_parameter_requirements(self) -> List[str]:
        """Return list of required parameter names."""
        return ["volume_L", "clearance_L_per_h"]
```

### Step 3: Add Model to Appropriate Directory

Place your model in the correct subdirectory:

- **Lung models**: `models_new/individual/lung_pbbm/my_new_lung_model.py`
- **PK models**: `models_new/individual/systemic_pk/my_new_pk_model.py`
- **GI models**: `models_new/individual/gi_absorption/my_new_gi_model.py`
- **New category**: `models_new/individual/my_new_category/my_new_model.py`

### Step 4: Update __init__.py Files

Add imports to make your model easily accessible:

```python
# In models_new/individual/systemic_pk/__init__.py
from .my_new_pk_model import MyNewModel, MyNewModelParams

# In models_new/individual/__init__.py  
from .systemic_pk import MyNewModel, MyNewModelParams
```

### Step 5: Create Integration Test

```python
#!/usr/bin/env python3
"""Test integration of MyNewModel with the architecture."""

def test_my_new_model():
    """Test the new model integration."""
    from lmp_pkg.models_new.individual.systemic_pk import MyNewModel, MyNewModelParams
    from lmp_pkg.models_new.composition.sequential import SequentialModelComposer
    
    # Create model
    params = MyNewModelParams(
        volume_L=5.0,
        clearance_L_per_h=35.0
    )
    model = MyNewModel(params)
    
    # Test interface compliance
    assert hasattr(model, 'requires')
    assert hasattr(model, 'provides') 
    assert hasattr(model, 'compute_derivatives')
    assert hasattr(model, 'extract_outputs')
    assert hasattr(model, 'initialize_state')
    
    # Test basic functionality
    state = model.initialize_state()
    assert len(state) == model.get_state_size()
    
    # Test derivatives calculation
    deps = {"drug_input_rate": 100.0}  # pmol/s
    derivatives = model.compute_derivatives(0.0, state, deps)
    assert len(derivatives) == len(state)
    
    # Test output extraction
    outputs = model.extract_outputs(0.0, state)
    assert "plasma_concentration" in outputs
    
    print("✅ MyNewModel integration test passed!")

if __name__ == "__main__":
    test_my_new_model()
```

## 🔗 Model Integration Patterns

### Pattern 1: Simple Independent Model
For models that don't depend on others:

```python
@property
def requires(self) -> List[ModelDependency]:
    return []  # No dependencies
```

### Pattern 2: Consumer Model  
For models that need inputs from others:

```python
@property
def requires(self) -> List[ModelDependency]:
    return [
        ModelDependency(
            name="systemic_absorption_rate", 
            data_type="pmol/s", 
            description="Input from lung", 
            required=True
        ),
        ModelDependency(
            name="gi_absorption_rate", 
            data_type="pmol/s", 
            description="Input from GI", 
            required=False  # Optional
        )
    ]
```

### Pattern 3: Provider Model
For models that mainly provide outputs:

```python
@property  
def provides(self) -> List[ModelOutput]:
    return [
        ModelOutput(
            name="regional_amounts", 
            data_type="pmol", 
            description="Regional drug amounts"
        ),
        ModelOutput(
            name="clearance_rates", 
            data_type="pmol/s", 
            description="Regional clearance rates"
        ),
        ModelOutput(
            name="binding_fractions", 
            data_type="dimensionless", 
            description="Bound drug fractions"
        )
    ]
```

### Pattern 4: Multi-Compartment Model

```python
def get_state_size(self, **params) -> int:
    return 3  # Central + 2 peripheral compartments
    
def initialize_state(self, **params) -> np.ndarray:
    return np.array([0.0, 0.0, 0.0])  # All compartments empty initially

def compute_derivatives(self, t: float, state: np.ndarray, 
                      dependencies: Dict[str, Any], **params) -> np.ndarray:
    central, peripheral1, peripheral2 = state
    
    # Input rate
    input_rate = dependencies.get("drug_input_rate", 0.0)
    
    # Inter-compartment transfers
    k12 = self.params.k12_per_h / 3600.0  # Convert to per second
    k21 = self.params.k21_per_h / 3600.0
    k13 = self.params.k13_per_h / 3600.0  
    k31 = self.params.k31_per_h / 3600.0
    k10 = self.params.k10_per_h / 3600.0  # Elimination
    
    # Mass balance equations
    dcentral_dt = input_rate - k12*central + k21*peripheral1 - k13*central + k31*peripheral2 - k10*central
    dperipheral1_dt = k12*central - k21*peripheral1
    dperipheral2_dt = k13*central - k31*peripheral2
    
    return np.array([dcentral_dt, dperipheral1_dt, dperipheral2_dt])
```

## 🧪 Testing Your Model

### Unit Tests

```python
def test_model_mass_balance():
    """Test that your model conserves mass."""
    model = MyNewModel(MyNewModelParams(volume_L=5.0, clearance_L_per_h=35.0))
    
    state = model.initialize_state()
    deps = {"drug_input_rate": 100.0}
    
    # Run for short time
    dt = 1.0  # 1 second
    derivs = model.compute_derivatives(0.0, state, deps)
    new_state = state + derivs * dt
    
    # Check mass balance
    input_mass = 100.0 * dt  # pmol
    state_change = np.sum(new_state - state)
    
    # Should be approximately equal (allowing for elimination)
    assert abs(state_change) <= input_mass
```

### Integration Tests

```python
def test_model_composition():
    """Test model works in composition."""
    from lmp_pkg.models_new.composition.sequential import SequentialModelComposer
    
    # Create lung model that provides input
    lung_model = SomeLungModel(...)
    
    # Create your model that consumes input
    my_model = MyNewModel(MyNewModelParams(...))
    
    # Compose them
    composer = SequentialModelComposer()
    composer.add_model(lung_model)
    composer.add_model(my_model)
    
    # Should validate successfully
    assert composer.validate_composition()
    
    # Should build and simulate
    combined = composer.build_combined_system()
    results = combined.simulate((0.0, 3600.0))
    
    assert len(results.t) > 0
```

## 📊 Adding to Analysis and Visualization

### Make Your Model Visualizable

```python
def extract_outputs(self, t: float, state: np.ndarray, **params) -> Dict[str, Any]:
    outputs = super().extract_outputs(t, state, **params)
    
    # Add visualization-friendly outputs
    outputs.update({
        "compartment_amounts": state.tolist(),
        "total_amount": np.sum(state),
        "concentration_profile": state / self.params.volumes  # If multi-compartment
    })
    
    return outputs
```

### Add Custom Visualization Methods

```python
# In your model class or as a separate utility
def create_compartment_plot(self, results):
    """Create custom visualization for this model type."""
    import matplotlib.pyplot as plt
    
    fig, ax = plt.subplots()
    
    # Extract your model's data from results
    for model_name in results.models:
        if isinstance(results.models[model_name], MyNewModel):
            outputs = results.get_model_outputs(model_name)
            # Plot your specific outputs
            ax.plot(results.t, [out['total_amount'] for out in outputs])
    
    return fig
```

## 🚀 Advanced Model Features

### Dynamic Model Structure

```python
def get_state_size(self, n_compartments: int = 3) -> int:
    """Allow variable number of compartments."""
    return n_compartments

def compute_derivatives(self, t: float, state: np.ndarray, 
                      dependencies: Dict[str, Any], 
                      n_compartments: int = 3) -> np.ndarray:
    """Handle variable compartment structure."""
    derivatives = np.zeros_like(state)
    
    for i in range(n_compartments):
        # Compute derivatives for compartment i
        pass
    
    return derivatives
```

### Parameter Transforms Integration

```python
from ..transforms.parameter_scaling import ParameterTransform

class MyModelTransform(ParameterTransform):
    """Transform parameters for MyNewModel."""
    
    def transform(self, base_params: Dict[str, Any], 
                  scaling_factors: Dict[str, float]) -> MyNewModelParams:
        
        scaled_volume = base_params['volume_L'] * scaling_factors.get('volume_scale', 1.0)
        scaled_clearance = base_params['clearance_L_per_h'] * scaling_factors.get('clearance_scale', 1.0)
        
        return MyNewModelParams(
            volume_L=scaled_volume,
            clearance_L_per_h=scaled_clearance
        )
```

### Numba Optimization

```python
from numba import jit

@jit(nopython=True)
def _compute_derivatives_numba(state, input_rate, volume, clearance):
    """Optimized derivative calculation."""
    amount = state[0]
    concentration = amount / volume  
    elimination_rate = concentration * clearance / 3600.0
    damount_dt = input_rate - elimination_rate
    return damount_dt

def compute_derivatives(self, t: float, state: np.ndarray, 
                      dependencies: Dict[str, Any], **params) -> np.ndarray:
    input_rate = dependencies.get("drug_input_rate", 0.0)
    
    # Call optimized function
    deriv = _compute_derivatives_numba(
        state, input_rate, self.params.volume_L, self.params.clearance_L_per_h
    )
    
    return np.array([deriv])
```

## ✅ Checklist for New Models

- [ ] Create parameter dataclass with type hints
- [ ] Implement all required abstract methods
- [ ] Define clear `requires` and `provides` interfaces
- [ ] Add validation in `validate_parameters`
- [ ] Write unit tests for the model logic
- [ ] Write integration tests with model composition
- [ ] Add to appropriate `__init__.py` files
- [ ] Test with visualization system
- [ ] Document parameter meanings and units
- [ ] Consider Numba optimization for performance
- [ ] Add to catalog system if needed
- [ ] Update architecture documentation

## 🎯 Common Patterns Summary

1. **State Management**: Use numpy arrays, size determined by `get_state_size()`
2. **Dependencies**: Declare what you need in `requires`, get via `dependencies` dict
3. **Outputs**: Declare what you provide in `provides`, return via `extract_outputs`
4. **Units**: Always specify units in dependency/output descriptions  
5. **Validation**: Check parameters in `validate_parameters`
6. **Testing**: Test both standalone and in composition
7. **Performance**: Consider Numba for computationally intensive models

The architecture is designed to be extensible - follow these patterns and your new models will integrate seamlessly!