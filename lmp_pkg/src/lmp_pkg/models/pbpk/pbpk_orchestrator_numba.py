"""PBPK Orchestrator - Full Numba Optimization for Maximum Performance

This class:
1. Takes FINAL parameters from Subject.get_final_attributes(), API, Product
2. Creates NUMBA-COMPILED model instances for maximum performance
3. Uses streamlined ODE system that minimizes Python overhead
4. NO sampling, NO hardcoded values, NO solver integration
5. Compatible with Stage protocol and Pipeline execution
6. TRUE MODULARITY - can combine any model components without hardcoding
7. FULL NUMBA PATHS - all critical model computations in jitclasses
"""

from __future__ import annotations

import numpy as np
from typing import Dict, List, Any, Tuple, Optional, Union, Callable
from enum import Enum
from abc import ABC, abstractmethod

# Import the organized Numba models
from .lung_pbbm import create_regional_lung_entities, create_generational_lung_entities
from .gi_pbbm import create_gi_model, create_gi_model_default
from .systemic_pk import create_pk_model


class ModelComponentType(Enum):
    """Types of PBPK model components."""
    LUNG = "lung"
    GI = "gi" 
    PK = "pk"


class ModelComponent(ABC):
    """Simple base class for PBPK orchestrator components."""
    
    @property
    @abstractmethod
    def component_type(self) -> ModelComponentType:
        pass
    
    @property
    @abstractmethod
    def n_states(self) -> int:
        pass
    
    @abstractmethod
    def set_state_slice(self, state: np.ndarray, start_idx: int):
        pass
    
    @abstractmethod
    def compute_derivatives(self, fluxes: Any, external_inputs: Dict[str, float]) -> Tuple[np.ndarray, Any]:
        pass
    
    @abstractmethod
    def initialize_state(self, initial_values: Dict[str, float]) -> np.ndarray:
        pass


class LungComponent(ModelComponent):
    """Lung PBBM component using real Numba LungEntity models."""
    
    def __init__(self, subject_params, api_params, model_type: str = "regional", solve_dissolution: bool = True, suppress_et_absorption: bool = False):
        self.subject_params = subject_params
        self.api_params = api_params
        self.model_type = model_type
        self.suppress_et_absorption = suppress_et_absorption
        try:
            self._v_frac_g = float(getattr(subject_params.lung_regional, 'V_frac_g', 0.2))
        except Exception:
            self._v_frac_g = 0.2
        
        if model_type == "regional":
            self.lung_entities = create_regional_lung_entities(subject_params, api_params, {}, solve_dissolution)
            self.regions = ['ET', 'BB', 'bb', 'Al']
        elif model_type == "generational":
            self.lung_entities = create_generational_lung_entities(subject_params, api_params, {})
            self.regions = [f'gen_{i}' for i in range(25)]
        else:
            raise ValueError(f"Unknown lung model type: {model_type}")
        
        self._n_states = sum(entity.get_state_size() for entity in self.lung_entities)
    
    @property
    def component_type(self) -> ModelComponentType:
        return ModelComponentType.LUNG
    
    @property
    def n_states(self) -> int:
        return self._n_states
    
    def set_state_slice(self, state: np.ndarray, start_idx: int):
        current_idx = start_idx
        for entity in self.lung_entities:
            entity_state_size = entity.get_state_size()
            entity_state = state[current_idx:current_idx + entity_state_size]
            entity.set_state(entity_state)
            current_idx += entity_state_size
    
    def compute_derivatives(self, fluxes: Any, external_inputs: Dict[str, float]) -> Tuple[np.ndarray, Any]:
        derivatives = np.zeros(self.n_states)
        total_lung_to_systemic = 0.0
        total_mcc_to_gi = 0.0
        total_mcc_bb_to_BB = 0.0
        
        current_idx = 0
        for i, (entity, region) in enumerate(zip(self.lung_entities, self.regions)):
            deposition_input = external_inputs.get(f'{region}_deposition', 0.0)
            external_mcc_input = external_inputs.get(f'{region}_mcc', 0.0)
            
            entity_derivatives, mcc_output, systemic_absorption = entity.compute_derivatives(
                fluxes['plasma_concentration'], deposition_input, external_mcc_input
            )
            
            if self.suppress_et_absorption and region.upper() == 'ET':
                systemic_absorption = 0.0

            entity_state_size = entity.get_state_size()
            derivatives[current_idx:current_idx + entity_state_size] = entity_derivatives
            current_idx += entity_state_size
            
            total_lung_to_systemic += systemic_absorption
            
            if region == 'Al':
                pass
            elif region == 'bb':
                total_mcc_bb_to_BB += mcc_output
            else:
                total_mcc_to_gi += mcc_output
        
        fluxes['lung_to_systemic'] = total_lung_to_systemic
        fluxes['mcc_to_gi'] = total_mcc_to_gi
        fluxes['mcc_bb_to_BB'] = total_mcc_bb_to_BB
        
        return derivatives, fluxes
    
    def initialize_state(self, initial_values: Dict[str, float]) -> np.ndarray:
        state = np.zeros(self.n_states)
        current_idx = 0
        
        for i, entity in enumerate(self.lung_entities):
            entity_state = entity.initialize_state()
            entity_state_size = len(entity_state)
            region = self.regions[i]
            depo_key = f'{region}_deposition'
            
            if depo_key in initial_values:
                regional_amount = initial_values[depo_key]
                psd_key = f'{region}_psd'
                if psd_key in initial_values and isinstance(initial_values[psd_key], dict):
                    psd_data = initial_values[psd_key]
                    particle_sizes = psd_data.get('sizes', entity.initial_radii)
                    mass_fractions = psd_data.get('fractions', np.ones(len(entity.initial_radii)) / len(entity.initial_radii))
                else:
                    reference_sizes = np.array([0.1643, 0.2063, 0.2591, 0.3255, 0.4088, 0.5135, 0.6449, 0.8100, 
                                              1.0173, 1.2777, 1.6048, 2.0156, 2.5315, 3.1795, 3.9933, 5.0155, 
                                              6.2993, 7.9116, 9.9368]) * 1e-4 / 2.0
                    reference_fractions = np.array([0.185, 0.4513, 0.9911, 1.96, 3.489, 5.594, 8.075, 10.5, 
                                                  12.28, 12.95, 12.28, 10.5, 8.075, 5.594, 3.489, 1.96, 
                                                  0.9911, 0.4513, 0.1842]) / 100.0
                    if len(reference_sizes) == len(entity.initial_radii):
                        particle_sizes = reference_sizes
                        mass_fractions = reference_fractions
                    else:
                        particle_sizes = entity.initial_radii
                        mass_fractions = np.ones(len(entity.initial_radii)) / len(entity.initial_radii)
                
                if not entity.solve_dissolution:
                    entity_state[0] = regional_amount
                    particle_dose_pmol = 0.0
                else:
                    particle_dose_pmol = regional_amount
                    entity_state[0] = 0.0
                
                particle_volumes_cm3 = (4.0/3.0) * np.pi * (particle_sizes**3)
                particle_mass_pmol = particle_volumes_cm3 / entity.molar_volume
                radii_start_idx = 1 + 2*entity.n_epithelium_layers + 2
                particle_start_idx = radii_start_idx + entity.n_dissolution_bins
                
                if entity.solve_dissolution and particle_dose_pmol > 0:
                    for bin_idx in range(entity.n_dissolution_bins):
                        entity_state[radii_start_idx + bin_idx] = particle_sizes[bin_idx]
                        if particle_mass_pmol[bin_idx] > 0 and bin_idx < len(mass_fractions):
                            dose_this_bin = particle_dose_pmol * mass_fractions[bin_idx]
                            num_particles = dose_this_bin / particle_mass_pmol[bin_idx]
                            entity_state[particle_start_idx + bin_idx] = max(0.0, num_particles)
                else:
                    for bin_idx in range(entity.n_dissolution_bins):
                        entity_state[radii_start_idx + bin_idx] = particle_sizes[bin_idx]
                        entity_state[particle_start_idx + bin_idx] = 0.0
            
            state[current_idx:current_idx + entity_state_size] = entity_state
            current_idx += entity_state_size
        
        return state
    
    def extract_results(self, t: np.ndarray, state_slice: np.ndarray) -> Dict[str, Any]:
        results = {}
        current_idx = 0
        
        for i, (entity, region) in enumerate(zip(self.lung_entities, self.regions)):
            entity_state_size = entity.get_state_size()
            entity_state_slice = state_slice[:, current_idx:current_idx + entity_state_size]
            entity_results = entity.extract_results(t, entity_state_slice.T)
            
            for key, value in entity_results.items():
                results[f'{region}_{key}'] = value
            try:
                results[f'{region}_elf_volume_ml'] = float(entity.vol_elf)
                results[f'{region}_epithelium_volume_ml'] = float(entity.vol_epithelium_layer * entity.n_epithelium_layers)
                results[f'{region}_tissue_volume_ml'] = float(entity.vol_tissue)
                # Export fu effective values
                fu_epi_calc = float(entity.fu_epithelium)
                if entity.k_in_epithelium > 0.0 and entity.k_out_epithelium > 0.0:
                    fu_epi_calc = float(entity.fu_epithelium * (1.0 + entity.k_in_epithelium / entity.k_out_epithelium))
                fu_tis_calc = float(entity.fu_tissue)
                if entity.k_in_tissue > 0.0 and entity.k_out_tissue > 0.0:
                    if entity.cell_binding != 0:
                        fu_tis_calc = float(1.0 / max(self._v_frac_g, 1e-9))
                    else:
                        fu_tis_calc = float(entity.fu_tissue * (1.0 + entity.k_in_tissue / entity.k_out_tissue))
                results[f'{region}_fu_epithelium_calc'] = fu_epi_calc
                results[f'{region}_fu_tissue_calc'] = fu_tis_calc
            except Exception:
                pass
            current_idx += entity_state_size
        
        return results
    
    @property
    def state_names(self) -> List[str]:
        names = []
        for entity, region in zip(self.lung_entities, self.regions):
            entity_names = getattr(entity, 'state_names', [f'{region}_state_{i}' for i in range(entity.get_state_size())])
            names.extend(entity_names)
        return names


class GIComponent(ModelComponent):
    def __init__(self, subject_params, api_params, enable_absorption: bool = True):
        self.subject_params = subject_params
        self.api_params = api_params
        self.enable_absorption = enable_absorption
        try:
            self.gi_model = create_gi_model(subject_params, api_params)
        except Exception:
            self.gi_model = create_gi_model_default(api_params)
        self._n_states = self.gi_model.get_state_size()
    
    @property
    def component_type(self) -> ModelComponentType:
        return ModelComponentType.GI
    
    @property
    def n_states(self) -> int:
        return self._n_states
    
    def set_state_slice(self, state: np.ndarray, start_idx: int):
        gi_state = state[start_idx:start_idx + self.n_states]
        self.gi_model.set_state(gi_state)
    
    def compute_derivatives(self, fluxes: Any, external_inputs: Dict[str, float]) -> Tuple[np.ndarray, Any]:
        gi_derivs, gi_absorption, hepatic_clearance = self.gi_model.compute_derivatives(
            fluxes['mcc_to_gi'], fluxes['plasma_concentration']
        )
        if not self.enable_absorption:
            gi_absorption = 0.0
            hepatic_clearance = 0.0
            gi_derivs = np.zeros_like(gi_derivs)
        fluxes['gi_to_systemic'] = gi_absorption
        fluxes['hepatic_clearance'] = hepatic_clearance
        return gi_derivs, fluxes
    
    def initialize_state(self, initial_values: Dict[str, float]) -> np.ndarray:
        state = np.zeros(self.n_states)
        for i in range(self.n_states):
            key = f'gi_{i}'
            if key in initial_values:
                state[i] = initial_values[key]
        return state
    
    def extract_results(self, t: np.ndarray, state_slice: np.ndarray) -> Dict[str, Any]:
        results = {}
        if state_slice.size > 0 and len(state_slice.shape) > 1:
            for i in range(min(self.n_states, state_slice.shape[1])):
                results[f'compartment_{i}'] = state_slice[:, i]
        else:
            for i in range(self._n_states):
                results[f'compartment_{i}'] = np.zeros(len(t))
        return results
    
    @property
    def state_names(self) -> List[str]:
        return [f'gi_compartment_{i}' for i in range(self._n_states)]


class PKComponent(ModelComponent):
    def __init__(self, subject_params, api_params, model_type: str = "3c"):
        self.subject_params = subject_params
        self.api_params = api_params
        self.model_type = model_type
        
        # Build PK from API/Subject
        pk = getattr(subject_params, 'pk', None)
        if pk is not None:
            vd_central = float(getattr(pk, 'volume_central_L', getattr(api_params, 'volume_central_L', 5.0)))
            cl_systemic = float(getattr(pk, 'clearance_L_h', getattr(api_params, 'clearance_L_h', 35.0)))
            vd_peripheral1 = float(getattr(pk, 'volume_peripheral1_L', 15.0))
            vd_peripheral2 = float(getattr(pk, 'volume_peripheral2_L', 50.0))
            cl_distribution1 = float(getattr(pk, 'cl_distribution1_L_h', 100.0))
            cl_distribution2 = float(getattr(pk, 'cl_distribution2_L_h', 20.0))
        else:
            vd_central = float(getattr(api_params, 'volume_central_L', 5.0))
            cl_systemic = float(getattr(api_params, 'clearance_L_h', 35.0))
            k12_h = getattr(api_params, 'k12_h', None)
            k21_h = getattr(api_params, 'k21_h', None)
            k13_h = getattr(api_params, 'k13_h', None)
            k31_h = getattr(api_params, 'k31_h', None)
            if k12_h and k21_h:
                cl_distribution1 = float(k12_h * vd_central)
                vd_peripheral1 = float(cl_distribution1 / k21_h) if k21_h else 15.0
            else:
                cl_distribution1 = 100.0
                vd_peripheral1 = 15.0
            if k13_h and k31_h:
                cl_distribution2 = float(k13_h * vd_central)
                vd_peripheral2 = float(cl_distribution2 / k31_h) if k31_h else 50.0
            else:
                cl_distribution2 = 20.0
                vd_peripheral2 = 50.0

        vd_peripheral = vd_peripheral1
        pk_params = {
            'vd_central_L': vd_central,
            'vd_peripheral_L': vd_peripheral,
            'vd_peripheral1_L': vd_peripheral1,
            'vd_peripheral2_L': vd_peripheral2,
            'cl_systemic_L_h': cl_systemic,
            'cl_distribution_L_h': cl_distribution1,
            'cl_distribution1_L_h': cl_distribution1,
            'cl_distribution2_L_h': cl_distribution2
        }
        self.pk_model = create_pk_model(model_type, pk_params)
        self._n_states = self.pk_model.get_state_size()
        if model_type == "1c":
            self._state_names = ['pk_central']
        elif model_type == "2c":
            self._state_names = ['pk_central', 'pk_peripheral']
        elif model_type == "3c":
            self._state_names = ['pk_central', 'pk_peripheral1', 'pk_peripheral2']
        else:
            self._state_names = [f'pk_state_{i}' for i in range(self._n_states)]
    
    @property
    def component_type(self) -> ModelComponentType:
        return ModelComponentType.PK
    
    @property
    def n_states(self) -> int:
        return self._n_states
    
    def set_state_slice(self, state: np.ndarray, start_idx: int):
        pk_state = state[start_idx:start_idx + self.n_states]
        if self.model_type == "1c":
            self.pk_model.set_state(pk_state[0])
        elif self.model_type == "2c":
            self.pk_model.set_state(pk_state[0], pk_state[1])
        elif self.model_type == "3c":
            self.pk_model.set_state(pk_state[0], pk_state[1], pk_state[2])
    
    def compute_derivatives(self, fluxes: Any, external_inputs: Dict[str, float]) -> Tuple[np.ndarray, Any]:
        total_input = fluxes['lung_to_systemic'] + fluxes['gi_to_systemic'] + external_inputs.get('iv_dose_rate', 0.0)
        derivatives_result = self.pk_model.compute_derivatives(total_input, 0.0)
        if self.model_type == "1c":
            derivatives = np.array([derivatives_result])
        elif self.model_type in ("2c", "3c"):
            derivatives = np.array(derivatives_result)
        fluxes['plasma_concentration'] = self.pk_model.get_plasma_concentration()
        return derivatives, fluxes
    
    def initialize_state(self, initial_values: Dict[str, float]) -> np.ndarray:
        state = np.zeros(self.n_states)
        for i, name in enumerate(self.state_names):
            if name in initial_values:
                state[i] = initial_values[name]
        return state
    
    def extract_results(self, t: np.ndarray, state_slice: np.ndarray) -> Dict[str, Any]:
        results = {}
        if state_slice.size > 0 and len(state_slice.shape) > 1:
            for i, name in enumerate(self.state_names):
                if i < state_slice.shape[1]:
                    clean_name = name.replace('pk_', '')
                    results[clean_name] = state_slice[:, i]
        else:
            for i, name in enumerate(self.state_names):
                clean_name = name.replace('pk_', '')
                results[clean_name] = np.zeros(len(t))
        if len(state_slice) > 0:
            central_amount_pmol = state_slice[:, 0]
            vd_central_L = getattr(self.pk_model, 'vd_central', None)
            if vd_central_L is None:
                vd_central_L = float(getattr(self.api_params, 'volume_central_L', 29.92))
            mw = float(getattr(self.api_params, 'molecular_weight', 430.54))
            conc_pmol_per_L = central_amount_pmol / vd_central_L
            conc_ng_per_ml = conc_pmol_per_L * mw * 1e-6
            results['plasma_concentration_ng_ml'] = conc_ng_per_ml
        else:
            results['plasma_concentration_ng_ml'] = np.zeros_like(t)
        return results
    
    @property
    def state_names(self) -> List[str]:
        return self._state_names


class PBPKOrchestratorNumba:
    """High-performance PBPK orchestrator leveraging numba-compiled models."""
    
    def __init__(self, 
                 subject_params,
                 api_params,
                 components: Optional[List[Union[ModelComponentType, ModelComponent]]] = None,
                 lung_model_type: str = "regional",
                 pk_model_type: str = "3c",
                 solve_dissolution: bool = True,
                 charcoal_block: bool = False,
                 suppress_et_absorption: bool = False):
        self.subject_params = subject_params
        self.api_params = api_params
        self.lung_model_type = lung_model_type
        self.pk_model_type = pk_model_type
        self.solve_dissolution = solve_dissolution
        self.charcoal_block = charcoal_block
        self.suppress_et_absorption = suppress_et_absorption
        
        self.components: List[ModelComponent] = []
        self.state_mapping: Dict[str, int] = {}
        self.state_names: List[str] = []
        
        if components is None:
            components = [ModelComponentType.LUNG, ModelComponentType.GI, ModelComponentType.PK]
        
        current_idx = 0
        for comp in components:
            if isinstance(comp, ModelComponentType):
                component = self._create_component(comp, subject_params, api_params)
            else:
                component = comp
            self.components.append(component)
            for i, name in enumerate(component.state_names):
                self.state_mapping[name] = current_idx + i
                self.state_names.append(name)
            current_idx += component.n_states
        self.n_states = current_idx
    
    def _create_component(self, comp_type: ModelComponentType, 
                         subject_params: Dict[str, Any], 
                         api_params: Dict[str, Any]) -> ModelComponent:
        if comp_type == ModelComponentType.LUNG:
            return LungComponent(subject_params, api_params, self.lung_model_type, self.solve_dissolution, self.suppress_et_absorption)
        elif comp_type == ModelComponentType.GI:
            return GIComponent(subject_params, api_params, enable_absorption=not self.charcoal_block)
        elif comp_type == ModelComponentType.PK:
            return PKComponent(subject_params, api_params, self.pk_model_type)
        else:
            raise ValueError(f"Unknown component type: {comp_type}")
    
    def get_ode_system(self) -> Callable:
        """Lean ODE system minimizing Python overhead while leveraging jitclasses."""
        def ode_system(t: float, y: np.ndarray, **kwargs) -> np.ndarray:
            y = np.maximum(y, 0.0)
            external_inputs = kwargs.get('external_inputs', {})
            derivatives = np.zeros_like(y)
            
            # Determine slices
            current_idx = 0
            lung_comp = next((c for c in self.components if c.component_type == ModelComponentType.LUNG), None)
            gi_comp = next((c for c in self.components if c.component_type == ModelComponentType.GI), None)
            pk_comp = next((c for c in self.components if c.component_type == ModelComponentType.PK), None)
            
            lung_size = lung_comp.n_states if lung_comp else 0
            gi_size = gi_comp.n_states if gi_comp else 0
            pk_size = pk_comp.n_states if pk_comp else 0
            
            lung_start = 0
            gi_start = lung_start + lung_size
            pk_start = gi_start + gi_size
            
            # Update PK state and get plasma concentration
            if pk_comp and pk_size > 0:
                pk_state = y[pk_start:pk_start+pk_size]
                pk_comp.set_state_slice(y, pk_start)
                plasma_conc = pk_comp.pk_model.get_plasma_concentration()
            else:
                plasma_conc = 0.0
            
            # Flux dict
            fluxes = {
                'mcc_to_gi': 0.0,
                'mcc_bb_to_BB': 0.0,
                'lung_to_systemic': 0.0,
                'gi_to_systemic': 0.0,
                'hepatic_clearance': 0.0,
                'plasma_concentration': plasma_conc,
            }
            
            # Lung derivatives
            if lung_comp and lung_size > 0:
                lung_state = y[lung_start:lung_start+lung_size]
                lung_comp.set_state_slice(y, lung_start)
                lung_derivs, fluxes = lung_comp.compute_derivatives(fluxes, external_inputs)
                derivatives[lung_start:lung_start+lung_size] = lung_derivs
            
            # GI derivatives
            if gi_comp and gi_size > 0:
                gi_state = y[gi_start:gi_start+gi_size]
                gi_comp.set_state_slice(y, gi_start)
                gi_derivs, fluxes = gi_comp.compute_derivatives(fluxes, external_inputs)
                derivatives[gi_start:gi_start+gi_size] = gi_derivs
            
            # PK derivatives
            if pk_comp and pk_size > 0:
                pk_derivs, fluxes = pk_comp.compute_derivatives(fluxes, external_inputs)
                derivatives[pk_start:pk_start+pk_size] = pk_derivs
            
            return derivatives
        return ode_system
    
    def get_initial_state(self, initial_values: Optional[Dict[str, float]] = None) -> np.ndarray:
        if initial_values is None:
            initial_values = {}
        y0 = np.zeros(self.n_states)
        current_idx = 0
        for component in self.components:
            comp_state = component.initialize_state(initial_values)
            y0[current_idx:current_idx + component.n_states] = comp_state
            current_idx += component.n_states
        return y0
    
    def extract_results(self, t: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        results = {
            'time': t,
            'state_names': self.state_names
        }
        current_idx = 0
        for component in self.components:
            comp_slice = y[:, current_idx:current_idx + component.n_states]
            comp_results = component.extract_results(t, comp_slice)
            results[component.component_type.value] = comp_results
            current_idx += component.n_states
        return results
    
    def get_model_info(self) -> Dict[str, Any]:
        component_info = []
        for comp in self.components:
            component_info.append({
                'type': comp.component_type.value,
                'n_states': comp.n_states,
                'state_names': comp.state_names
            })
        return {
            'n_components': len(self.components),
            'components': component_info,
            'total_states': self.n_states,
            'state_mapping': self.state_mapping
        }
    
    def solve(self, time_points: np.ndarray, **solver_kwargs) -> Any:
        """Solve with lean ODE using jitclasses; fast and robust."""
        from scipy.integrate import solve_ivp
        ode_func = self.get_ode_system()
        y0 = self.get_initial_state(solver_kwargs.pop('initial_conditions', {}))
        t_span = (time_points[0], time_points[-1])
        solver_options = {
            'method': 'BDF',
            'rtol': 1e-4,
            'atol': 1e-8,
            'max_step': 3600.0,
        }
        solver_options.update(solver_kwargs)
        sol = solve_ivp(
            ode_func, t_span, y0,
            t_eval=time_points,
            **solver_options
        )
        results_data = self.extract_results(sol.t, sol.y.T)

        # Compute flux series at t_eval (parity with non-numba orchestrator)
        try:
            t_eval = sol.t
            y_mat = sol.y.T
            # Identify components/slices
            lung_comp = next((c for c in self.components if isinstance(c, LungComponent)), None)
            gi_comp = next((c for c in self.components if isinstance(c, GIComponent)), None)
            pk_comp = next((c for c in self.components if isinstance(c, PKComponent)), None)

            lung_size = lung_comp.n_states if lung_comp else 0
            gi_size = gi_comp.n_states if gi_comp else 0
            pk_size = pk_comp.n_states if pk_comp else 0

            lung_start = 0
            gi_start = lung_start + lung_size
            pk_start = gi_start + gi_size

            per_region: Dict[str, np.ndarray] = {}
            per_region_mcc: Dict[str, np.ndarray] = {}
            total_lung_to_systemic = np.zeros_like(t_eval)
            gi_to_systemic = np.zeros_like(t_eval)
            mcc_total = np.zeros_like(t_eval)

            if lung_comp:
                for region in lung_comp.regions:
                    per_region[region] = np.zeros_like(t_eval)
                    per_region_mcc[region] = np.zeros_like(t_eval)

            # vd_central for plasma concentration
            vd_central = None
            if pk_comp and hasattr(pk_comp.pk_model, 'vd_central'):
                vd_central = float(getattr(pk_comp.pk_model, 'vd_central'))
            if vd_central is None:
                vd_central = float(getattr(self.api_params, 'volume_central_L', 29.92))

            for ti in range(len(t_eval)):
                y_row = y_mat[ti]
                # Plasma concentration (pmol/L)
                plasma_conc = 0.0
                if pk_size > 0:
                    central_amt = y_row[pk_start]
                    plasma_conc = max(0.0, central_amt) / vd_central if vd_central > 0 else 0.0

                # LUNG
                mcc_to_gi_flow = 0.0
                mcc_bb_to_BB_flow = 0.0
                if lung_comp and lung_size > 0:
                    current_idx = lung_start
                    for entity, region in zip(lung_comp.lung_entities, lung_comp.regions):
                        entity_state_size = entity.get_state_size()
                        entity_state = y_row[current_idx:current_idx + entity_state_size]
                        entity.set_state(entity_state)
                        external_mcc_input = mcc_bb_to_BB_flow if region == 'BB' else 0.0
                        _, mcc_out, sys_abs = entity.compute_derivatives(plasma_conc, 0.0, external_mcc_input)
                        total_lung_to_systemic[ti] += sys_abs
                        per_region[region][ti] = sys_abs
                        per_region_mcc[region][ti] = mcc_out
                        mcc_total[ti] += mcc_out
                        if region == 'bb':
                            mcc_bb_to_BB_flow += mcc_out
                        elif region in ('ET', 'BB'):
                            mcc_to_gi_flow += mcc_out
                        current_idx += entity_state_size

                # GI absorption
                if gi_comp and gi_size > 0:
                    gi_state = y_row[gi_start:gi_start + gi_size]
                    gi_comp.gi_model.set_state(gi_state)
                    _, gi_abs, _ = gi_comp.gi_model.compute_derivatives(mcc_to_gi_flow, plasma_conc)
                    gi_to_systemic[ti] = gi_abs

            results_data['flux'] = {
                'total_lung_to_systemic': total_lung_to_systemic,
                'gi_to_systemic': gi_to_systemic,
                'per_region': per_region,
                'mcc_total': mcc_total,
                'per_region_mcc': per_region_mcc,
            }
        except Exception:
            pass

        sol.results_data = results_data
        sol.state_matrix = sol.y.T
        sol.time_points = sol.t
        sol.model_info = self.get_model_info()
        sol.molecular_weight = self.api_params.molecular_weight
        return sol
