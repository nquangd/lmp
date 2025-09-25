"""PBPK Orchestrator - Real Numba Models with True Modularity

This class:
1. Takes FINAL parameters from Subject.get_final_attributes(), API, Product
2. Creates real Numba model instances for computation
3. Provides ODE system compatible with solver folder
4. NO sampling, NO hardcoded values, NO solver integration
5. Compatible with Stage protocol and Pipeline execution
6. TRUE MODULARITY - can combine any model components without hardcoding
"""

import numpy as np
from typing import Dict, List, Any, Tuple, Optional, Union, Callable
from enum import Enum
from abc import ABC, abstractmethod

# Import the organized Numba models
from .lung_pbbm import create_regional_lung_entities, create_generational_lung_entities
from .gi_pbbm import create_gi_model, create_gi_model_default
from .systemic_pk import create_pk_model

# Import simplified utilities
from .utils import FluxVector


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
        """Type of this component."""
        pass
    
    @property
    @abstractmethod
    def n_states(self) -> int:
        """Number of state variables."""
        pass
    
    @abstractmethod
    def set_state_slice(self, state: np.ndarray, start_idx: int):
        """Set state for this component from global state vector."""
        pass
    
    @abstractmethod
    def compute_derivatives(self, fluxes: FluxVector, external_inputs: Dict[str, float]) -> Tuple[np.ndarray, FluxVector]:
        """
        Compute derivatives for this component.
        
        Args:
            fluxes: Current flux state from other components
            external_inputs: External inputs (e.g., deposition, dosing)
            
        Returns:
            (derivatives_array, updated_fluxes)
        """
        pass
    
    @abstractmethod
    def initialize_state(self, initial_values: Dict[str, float]) -> np.ndarray:
        """Initialize state vector for this component."""
        pass


class LungComponent(ModelComponent):
    """Lung PBBM component using real Numba LungEntity models."""
    
    def __init__(self, subject_params, api_params, model_type: str = "regional", solve_dissolution: bool = True,
                 debug_kinetics: bool = False):
        self.subject_params = subject_params
        self.api_params = api_params
        self.model_type = model_type
        self.debug_kinetics = debug_kinetics
        # Store V_frac_g for tissue binding calc
        try:
            self._v_frac_g = float(getattr(subject_params.lung_regional, 'V_frac_g', 0.2))
        except Exception:
            self._v_frac_g = 0.2
        
        # Create lung entities using factory functions
        if model_type == "regional":
            # Now always use domain objects directly
            self.lung_entities = create_regional_lung_entities(subject_params, api_params, {}, solve_dissolution)
            self.regions = ['ET', 'BB', 'bb', 'Al']
        elif model_type == "generational":
            self.lung_entities = create_generational_lung_entities(subject_params, api_params, {})
            self.regions = [f'gen_{i}' for i in range(25)]  # 25 generations
        else:
            raise ValueError(f"Unknown lung model type: {model_type}")
        
        # Get total state information from all entities
        self._n_states = sum(entity.get_state_size() for entity in self.lung_entities)
        # Initialize tracing buffers
        self._last_sys_absorption_per_region = np.zeros(len(self.lung_entities))
        self._last_mcc_per_region = np.zeros(len(self.lung_entities))
        # Debug counters removed for production
        self._dbg_counts = {r: 0 for r in self.regions}
    
    @property
    def component_type(self) -> ModelComponentType:
        return ModelComponentType.LUNG
    
    @property
    def n_states(self) -> int:
        return self._n_states
    
    def set_state_slice(self, state: np.ndarray, start_idx: int):
        """Set state for all lung entities."""
        current_idx = start_idx
        for entity in self.lung_entities:
            entity_state_size = entity.get_state_size()
            entity_state = state[current_idx:current_idx + entity_state_size]
            entity.set_state(entity_state)
            current_idx += entity_state_size
    
    def compute_derivatives(self, fluxes: FluxVector, external_inputs: Dict[str, float]) -> Tuple[np.ndarray, FluxVector]:
        """Compute lung derivatives using all regional entities."""
        
        derivatives = np.zeros(self.n_states)
        total_lung_to_systemic = 0.0
        total_mcc_to_gi = 0.0
        total_mcc_bb_to_BB = 0.0
        
        current_idx = 0
        for i, (entity, region) in enumerate(zip(self.lung_entities, self.regions)):
            # Get deposition input for this region
            deposition_input = external_inputs.get(f'{region}_deposition', 0.0)
            
            # Get MCC input for this entity (MCC from higher regions)
            external_mcc_input = external_inputs.get(f'{region}_mcc', 0.0)
            
            # Compute derivatives for this entity
            entity_derivatives, mcc_output, systemic_absorption = entity.compute_derivatives(
                fluxes.plasma_concentration, deposition_input, external_mcc_input
            )
            
            # Store derivatives
            entity_state_size = entity.get_state_size()
            derivatives[current_idx:current_idx + entity_state_size] = entity_derivatives
            current_idx += entity_state_size
            
            # Accumulate fluxes (sum from all regions)
            total_lung_to_systemic += systemic_absorption
            # Debug logging removed
            self._dbg_counts[region] = self._dbg_counts.get(region, 0) + 1
            self._last_sys_absorption_per_region[i] = systemic_absorption
            
            # DEBUG: Check systemic absorption values - commented out for clean runs
            # print(f"      DEBUG: {region} systemic_absorption = {systemic_absorption:.6e} pmol/s")
            
            
            # MCC routing based on region
            if region == 'Al':
                # Alveolar region has no MCC output (is_alveolar = True)
                pass
            elif region == 'bb':
                # bb MCC goes to BB
                total_mcc_bb_to_BB += mcc_output
                self._last_mcc_per_region[i] = mcc_output
            else:
                # ET and BB MCC goes to GI
                total_mcc_to_gi += mcc_output
                self._last_mcc_per_region[i] = mcc_output
        
        # Update fluxes
        fluxes.lung_to_systemic = total_lung_to_systemic
        fluxes.mcc_to_gi = total_mcc_to_gi
        fluxes.mcc_bb_to_BB = total_mcc_bb_to_BB
        
        return derivatives, fluxes
    
    def initialize_state(self, initial_values: Dict[str, float]) -> np.ndarray:
        """Initialize state for all lung entities with deposition and particle counts."""
        state = np.zeros(self.n_states)
        current_idx = 0
        
        for i, entity in enumerate(self.lung_entities):
            entity_state = entity.initialize_state()  # Start with zeros
            entity_state_size = len(entity_state)
            
            # CRITICAL FIX: Set initial deposition and particle counts
            region = self.regions[i]
            depo_key = f'{region}_deposition'
            
            # Check for both deposition amount and PSD data
            if depo_key in initial_values:
                regional_amount = initial_values[depo_key]
                # print(f"    DEBUG: Initializing {region} with {regional_amount:.1f} pmol")
                
                # Check if PSD data is available
                psd_key = f'{region}_psd'  # Expected format: {region}_psd with size and fraction arrays
                if psd_key in initial_values and isinstance(initial_values[psd_key], dict):
                    psd_data = initial_values[psd_key]
                    particle_sizes = psd_data.get('sizes', entity.initial_radii)  # cm
                    mass_fractions = psd_data.get('fractions', np.ones(len(entity.initial_radii)) / len(entity.initial_radii))
                    # print(f"      Using actual PSD with {len(particle_sizes)} size bins")
                else:
                    # Use reference PSD from config.py (line 37-38)
                    reference_sizes = np.array([0.1643, 0.2063, 0.2591, 0.3255, 0.4088, 0.5135, 0.6449, 0.8100, 
                                              1.0173, 1.2777, 1.6048, 2.0156, 2.5315, 3.1795, 3.9933, 5.0155, 
                                              6.2993, 7.9116, 9.9368]) * 1e-4 / 2.0  # Convert to cm
                    reference_fractions = np.array([0.185, 0.4513, 0.9911, 1.96, 3.489, 5.594, 8.075, 10.5, 
                                                  12.28, 12.95, 12.28, 10.5, 8.075, 5.594, 3.489, 1.96, 
                                                  0.9911, 0.4513, 0.1842]) / 100.0
                    
                    # Use reference PSD if bins match, otherwise use entity defaults
                    if len(reference_sizes) == len(entity.initial_radii):
                        particle_sizes = reference_sizes
                        mass_fractions = reference_fractions
                        # print(f"      Using reference PSD with {len(particle_sizes)} size bins")
                    else:
                        particle_sizes = entity.initial_radii
                        mass_fractions = np.ones(len(entity.initial_radii)) / len(entity.initial_radii)
                        # print(f"      Using uniform PSD fallback with {len(particle_sizes)} size bins (size mismatch)")
                
                # Check if dissolution is disabled - if so, put dose directly in ELF
                if not entity.solve_dissolution:
                    # DISSOLUTION DISABLED: Put dose directly in ELF for immediate absorption
                    entity_state[0] = regional_amount  # 100% starts in ELF
                    particle_dose_pmol = 0.0  # No particles
                    # print(f"      ELF: {regional_amount:.1f} pmol (dissolution disabled - instant availability)")
                else:
                    # NORMAL MODE: All deposited drug starts as particles - ELF starts at 0
                    # Particles will dissolve over time during simulation
                    particle_dose_pmol = regional_amount  # 100% starts as particles
                    entity_state[0] = 0.0  # ELF starts empty
                    # print(f"      ELF: 0.0 pmol (particles will dissolve over time)")
                
                # Calculate particle mass per bin using molar volume approach (matches reference model)
                particle_volumes_cm3 = (4.0/3.0) * np.pi * (particle_sizes**3)
                # Use entity's molar volume (cm³/pmol) - this is the correct approach
                particle_mass_pmol = particle_volumes_cm3 / entity.molar_volume
                
                # CRITICAL: Set both particle radii AND counts (only if dissolution is enabled)
                # State structure: [ELF] + [epi_shallow] + [epi_deep] + [tissue_shallow] + [tissue_deep] + [radii] + [counts]
                radii_start_idx = 1 + 2*entity.n_epithelium_layers + 2  # After ELF + epithelium + tissue
                particle_start_idx = radii_start_idx + entity.n_dissolution_bins  # After radii
                
                if entity.solve_dissolution and particle_dose_pmol > 0:
                    # Normal mode: Initialize particles with PSD
                    for bin_idx in range(entity.n_dissolution_bins):
                        # Set initial particle radius (use actual particle sizes from PSD)
                        entity_state[radii_start_idx + bin_idx] = particle_sizes[bin_idx]
                        
                        # Set initial particle count using actual mass fractions from PSD
                        if particle_mass_pmol[bin_idx] > 0 and bin_idx < len(mass_fractions):
                            dose_this_bin = particle_dose_pmol * mass_fractions[bin_idx]  # Use actual mass fraction
                            num_particles = dose_this_bin / particle_mass_pmol[bin_idx]
                            entity_state[particle_start_idx + bin_idx] = max(0.0, num_particles)
                else:
                    # Dissolution disabled: Set particle radii but no particle counts (all particles = 0)
                    for bin_idx in range(entity.n_dissolution_bins):
                        entity_state[radii_start_idx + bin_idx] = particle_sizes[bin_idx]
                        entity_state[particle_start_idx + bin_idx] = 0.0  # No particles
                
                total_particles = np.sum(entity_state[particle_start_idx:particle_start_idx + entity.n_dissolution_bins])
                avg_radius_um = np.mean(particle_sizes) * 1e4  # Convert cm to μm
                total_mass_check = np.sum(mass_fractions)
                
                # if entity.solve_dissolution:
                #     print(f"      Particles: {total_particles:.1f} particles = {particle_dose_pmol:.1f} pmol")
                # else:
                #     print(f"      Particles: 0.0 particles = 0.0 pmol (dissolution disabled)")
                # print(f"      PSD: {len(particle_sizes)} bins, avg radius = {avg_radius_um:.2f} μm")
                # print(f"      Mass fractions sum: {total_mass_check:.6f} (should be ~1.0)")
            
            state[current_idx:current_idx + entity_state_size] = entity_state
            current_idx += entity_state_size
            
        return state
    
    def extract_results(self, t: np.ndarray, state_slice: np.ndarray) -> Dict[str, Any]:
        """Extract results from all lung entities."""
        results = {}
        current_idx = 0
        
        for i, (entity, region) in enumerate(zip(self.lung_entities, self.regions)):
            entity_state_size = entity.get_state_size()
            entity_state_slice = state_slice[:, current_idx:current_idx + entity_state_size]
            # Transpose to (states, time) for LungEntity expectations
            entity_results = entity.extract_results(t, entity_state_slice.T)
            
            # Store results with region prefix
            for key, value in entity_results.items():
                results[f'{region}_{key}'] = value

            # Add compartment volumes (mL) from entity attributes (not in jit dict)
            try:
                results[f'{region}_elf_volume_ml'] = float(entity.vol_elf)
                results[f'{region}_epithelium_volume_ml'] = float(entity.vol_epithelium_layer * entity.n_epithelium_layers)
                results[f'{region}_tissue_volume_ml'] = float(entity.vol_tissue)
                # Also export effective unbound fractions used in kinetics for convenience
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
        """Get state names for all lung entities."""
        names = []
        for entity, region in zip(self.lung_entities, self.regions):
            entity_names = getattr(entity, 'state_names', [f'{region}_state_{i}' for i in range(entity.get_state_size())])
            names.extend(entity_names)
        return names


class GIComponent(ModelComponent):
    """GI absorption component using real Numba GIModel."""
    
    def __init__(self, subject_params, api_params):
        self.subject_params = subject_params
        self.api_params = api_params
        
        # Create GI model using factory function
        try:
            self.gi_model = create_gi_model(subject_params, api_params)
        except:
            # Fallback to default model if subject parameters incomplete
            self.gi_model = create_gi_model_default(api_params)
        
        # Get state information from the model
        self._n_states = self.gi_model.get_state_size()
    
    @property
    def component_type(self) -> ModelComponentType:
        return ModelComponentType.GI
    
    @property
    def n_states(self) -> int:
        return self._n_states
    
    def set_state_slice(self, state: np.ndarray, start_idx: int):
        """Set GI state."""
        gi_state = state[start_idx:start_idx + self.n_states]
        self.gi_model.set_state(gi_state)
    
    def compute_derivatives(self, fluxes: FluxVector, external_inputs: Dict[str, float]) -> Tuple[np.ndarray, FluxVector]:
        """Compute GI derivatives using real Numba model."""
        
        # Compute derivatives using real Numba model
        gi_derivs, gi_absorption, hepatic_clearance = self.gi_model.compute_derivatives(
            fluxes.mcc_to_gi, fluxes.plasma_concentration
        )
        
        # Update fluxes
        fluxes.gi_to_systemic = gi_absorption
        fluxes.hepatic_clearance = hepatic_clearance
        
        return gi_derivs, fluxes
    
    def initialize_state(self, initial_values: Dict[str, float]) -> np.ndarray:
        """Initialize GI state."""
        state = np.zeros(self.n_states)
        for i in range(self.n_states):
            key = f'gi_{i}'
            if key in initial_values:
                state[i] = initial_values[key]
        return state
    
    def extract_results(self, t: np.ndarray, state_slice: np.ndarray) -> Dict[str, Any]:
        """Extract GI results."""
        results = {}
        if state_slice.size > 0 and len(state_slice.shape) > 1:
            for i in range(min(self.n_states, state_slice.shape[1])):
                results[f'compartment_{i}'] = state_slice[:, i]
        else:
            # Handle empty or 1D state slice
            for i in range(self.n_states):
                results[f'compartment_{i}'] = np.zeros(len(t))
        return results
    
    @property
    def state_names(self) -> List[str]:
        """Get GI state names."""
        return [f'gi_compartment_{i}' for i in range(self.n_states)]


class PKComponent(ModelComponent):
    """PK component using real Numba PKModel."""
    
    def __init__(self, subject_params, api_params, model_type: str = "3c", pk_source: str = "subject"):
        self.subject_params = subject_params
        self.api_params = api_params
        self.model_type = model_type
        self.pk_source = pk_source  # "subject" (default) or "api"
        
        # Select PK parameter source
        use_subject_pk = (self.pk_source == "subject") and hasattr(subject_params, 'pk') and subject_params.pk

        api_vd_central = getattr(api_params, 'volume_central_L', None)
        api_cl_systemic = getattr(api_params, 'clearance_L_h', None)
        api_vd_peripheral1 = getattr(api_params, 'volume_peripheral1_L', None)
        api_vd_peripheral2 = getattr(api_params, 'volume_peripheral2_L', None)
        api_cl_distribution1 = getattr(api_params, 'cl_distribution1_L_h', None)
        api_cl_distribution2 = getattr(api_params, 'cl_distribution2_L_h', None)
        api_k12 = getattr(api_params, 'k12_h', None)
        api_k21 = getattr(api_params, 'k21_h', None)
        api_k13 = getattr(api_params, 'k13_h', None)
        api_k31 = getattr(api_params, 'k31_h', None)

        if use_subject_pk:
            pk = subject_params.pk
            vd_central = float(api_vd_central if api_vd_central is not None else getattr(pk, 'volume_central_L', 5.0))
            cl_systemic = float(api_cl_systemic if api_cl_systemic is not None else getattr(pk, 'clearance_L_h', 35.0))
            vd_peripheral1 = float(api_vd_peripheral1 if api_vd_peripheral1 is not None else getattr(pk, 'volume_peripheral1_L', 15.0))
            vd_peripheral2 = float(api_vd_peripheral2 if api_vd_peripheral2 is not None else getattr(pk, 'volume_peripheral2_L', 50.0))
            cl_distribution1 = float(api_cl_distribution1 if api_cl_distribution1 is not None else getattr(pk, 'cl_distribution1_L_h', 100.0))
            cl_distribution2 = float(api_cl_distribution2 if api_cl_distribution2 is not None else getattr(pk, 'cl_distribution2_L_h', 20.0))
        else:
            vd_central = float(api_vd_central if api_vd_central is not None else 5.0)
            cl_systemic = float(api_cl_systemic if api_cl_systemic is not None else 35.0)

            if api_k12 and api_k21:
                cl_distribution1 = float(api_k12 * vd_central)
                vd_peripheral1 = float(cl_distribution1 / api_k21) if api_k21 else 15.0
            else:
                cl_distribution1 = float(api_cl_distribution1) if api_cl_distribution1 is not None else 100.0
                vd_peripheral1 = float(api_vd_peripheral1) if api_vd_peripheral1 is not None else 15.0

            if api_k13 and api_k31:
                cl_distribution2 = float(api_k13 * vd_central)
                vd_peripheral2 = float(cl_distribution2 / api_k31) if api_k31 else 50.0
            else:
                cl_distribution2 = float(api_cl_distribution2) if api_cl_distribution2 is not None else 20.0
                vd_peripheral2 = float(api_vd_peripheral2) if api_vd_peripheral2 is not None else 50.0

        # 2C compatibility value
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
        
        # Create PK model using factory function
        self.pk_model = create_pk_model(model_type, pk_params)
        
        # Get state information from the model
        self._n_states = self.pk_model.get_state_size()
        
        # Define state names based on model type
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
        """Set PK state."""
        pk_state = state[start_idx:start_idx + self.n_states]
        if self.model_type == "1c":
            self.pk_model.set_state(pk_state[0])
        elif self.model_type == "2c":
            self.pk_model.set_state(pk_state[0], pk_state[1])
        elif self.model_type == "3c":
            self.pk_model.set_state(pk_state[0], pk_state[1], pk_state[2])
    
    def compute_derivatives(self, fluxes: FluxVector, external_inputs: Dict[str, float]) -> Tuple[np.ndarray, FluxVector]:
        """Compute PK derivatives using real Numba model."""
        
        # Total input to systemic circulation
        total_input = fluxes.lung_to_systemic + fluxes.gi_to_systemic
        
        # Add any direct IV input
        iv_input = external_inputs.get('iv_dose_rate', 0.0)
        total_input += iv_input
        
        # DEBUG: Show total input to PK model
        # print(f"      DEBUG: PK total_input = {total_input:.6e} pmol/s (lung: {fluxes.lung_to_systemic:.6e}, gi: {fluxes.gi_to_systemic:.6e})")
        
        # Compute derivatives using real Numba model
        derivatives_result = self.pk_model.compute_derivatives(total_input, 0.0)
        
        # Handle different model types
        if self.model_type == "1c":
            derivatives = np.array([derivatives_result])
        elif self.model_type == "2c":
            derivatives = np.array(derivatives_result)
        elif self.model_type == "3c":
            derivatives = np.array(derivatives_result)
        
        # Update plasma concentration
        fluxes.plasma_concentration = self.pk_model.get_plasma_concentration()
        
        return derivatives, fluxes
    
    def initialize_state(self, initial_values: Dict[str, float]) -> np.ndarray:
        """Initialize PK state."""
        state = np.zeros(self.n_states)
        for i, name in enumerate(self.state_names):
            if name in initial_values:
                state[i] = initial_values[name]
        return state
    
    def extract_results(self, t: np.ndarray, state_slice: np.ndarray) -> Dict[str, Any]:
        """Extract PK results."""
        results = {}
        
        # Add compartment results
        if state_slice.size > 0 and len(state_slice.shape) > 1:
            for i, name in enumerate(self.state_names):
                if i < state_slice.shape[1]:
                    clean_name = name.replace('pk_', '')
                    results[clean_name] = state_slice[:, i]
        else:
            # Handle empty or 1D state slice
            for i, name in enumerate(self.state_names):
                clean_name = name.replace('pk_', '')
                results[clean_name] = np.zeros(len(t))
        
        # Calculate plasma concentration time series from central compartment
        if len(state_slice) > 0:
            # Central compartment is first state for all PK models
            central_amount_pmol = state_slice[:, 0]  # pmol over time

            # Get volume of central compartment from PK model (jitclasses expose 'vd_central')
            vd_central_L = getattr(self.pk_model, 'vd_central', None)
            if vd_central_L is None:
                # Fallback to API param if available, otherwise BD default
                vd_central_L = float(getattr(self.api_params, 'volume_central_L', 29.92))

            # Calculate concentration: pmol/L -> ng/mL
            # (pmol/L) * (MW μg/μmol) * 1e-6 = ng/mL
            mw = float(getattr(self.api_params, 'molecular_weight', 430.54))

            conc_pmol_per_L = central_amount_pmol / vd_central_L
            conc_ng_per_ml = conc_pmol_per_L * mw * 1e-6  # Convert pmol/L to ng/mL

            results['plasma_concentration_ng_ml'] = conc_ng_per_ml
        else:
            results['plasma_concentration_ng_ml'] = np.zeros_like(t)
        
        return results
    
    @property
    def state_names(self) -> List[str]:
        """Get PK state names."""
        return self._state_names


class PBPKOrchestrator:
    """
    True modular PBPK orchestrator using real Numba models.
    
    Key features:
    - Uses REAL Numba models, not simplified versions
    - TRUE MODULARITY: can combine any components without hardcoding
    - NO hardcoded values - all from input parameters
    - Compatible with pipeline/stage/solver architecture
    """
    
    def __init__(self, 
                 subject_params,
                 api_params,
                 components: Optional[List[Union[ModelComponentType, ModelComponent]]] = None,
                 lung_model_type: str = "regional",
                 pk_model_type: str = "3c",
                 solve_dissolution: bool = True,
                 debug_kinetics: bool = False,
                 pk_source: str = "subject"):
        """
        Initialize with final parameters and flexible component list.
        
        Args:
            subject_params: Final subject parameters (no sampling)
            api_params: Final API parameters
            components: List of components to include (types or instances)
            lung_model_type: Type of lung model ("regional" or "generational")
            pk_model_type: Type of PK model ("1c", "2c", "3c")
        """
        self.subject_params = subject_params
        self.api_params = api_params
        self.lung_model_type = lung_model_type
        self.pk_model_type = pk_model_type
        self.solve_dissolution = solve_dissolution
        self.debug_kinetics = debug_kinetics
        self.pk_source = pk_source
        
        # Build component list
        self.components = []
        self.state_mapping = {}
        self.state_names = []
        
        if components is None:
            # Default: full PBPK - order doesn't matter with simultaneous coupling
            components = [ModelComponentType.LUNG, ModelComponentType.GI, ModelComponentType.PK]
        
        # Create components
        current_idx = 0
        for comp in components:
            if isinstance(comp, ModelComponentType):
                # Create component from type
                component = self._create_component(comp, subject_params, api_params)
            else:
                # Use provided component instance
                component = comp
            
            self.components.append(component)
            
            # Map state indices
            for i, name in enumerate(component.state_names):
                self.state_mapping[name] = current_idx + i
                self.state_names.append(name)
            
            current_idx += component.n_states
        
        self.n_states = current_idx
        # Flux placeholders
        self._t_eval = None
        self._flux_rec = None
        self._rec_idx = 0
    
    def _create_component(self, comp_type: ModelComponentType, 
                         subject_params: Dict[str, Any], 
                         api_params: Dict[str, Any]) -> ModelComponent:
        """Create component instance from type - truly modular."""
        
        if comp_type == ModelComponentType.LUNG:
            return LungComponent(subject_params, api_params, self.lung_model_type, self.solve_dissolution,
                                 debug_kinetics=self.debug_kinetics)
        elif comp_type == ModelComponentType.GI:
            return GIComponent(subject_params, api_params)
        elif comp_type == ModelComponentType.PK:
            return PKComponent(subject_params, api_params, self.pk_model_type, pk_source=self.pk_source)
        else:
            raise ValueError(f"Unknown component type: {comp_type}")
    
    def add_component(self, component: Union[ModelComponentType, ModelComponent]):
        """Add a component dynamically - true modularity."""
        
        if isinstance(component, ModelComponentType):
            component = self._create_component(component, self.subject_params, self.api_params)
        
        # Update state mapping
        start_idx = self.n_states
        self.components.append(component)
        
        for i, name in enumerate(component.state_names):
            self.state_mapping[name] = start_idx + i
            self.state_names.append(name)
        
        self.n_states += component.n_states
    
    def get_ode_system(self) -> Callable:
        """
        Get ODE system function for solver - works with real Numba models.
        
        Returns:
            Function compatible with scipy.integrate.solve_ivp
        """
        
        def ode_system(t: float, y: np.ndarray, **kwargs) -> np.ndarray:
            """ODE system using real Numba models."""
            
            # CRITICAL: Clamp negative values to zero (matches old code line 36)
            y = np.maximum(y, 0.0)
            
            # Update all component states
            current_idx = 0
            for component in self.components:
                component.set_state_slice(y, current_idx)
                current_idx += component.n_states
            
            # Initialize flux vector
            fluxes = FluxVector()
            
            # Get external inputs
            external_inputs = kwargs.get('external_inputs', {})
            
            # SIMULTANEOUS COUPLING: Components must run together, not sequentially
            # Step 1: Update ALL component states from current y vector
            current_idx = 0
            pk_component = None
            for component in self.components:
                component.set_state_slice(y, current_idx)
                current_idx += component.n_states
                
                # Remember PK component for later
                if component.component_type.value == 'pk':
                    pk_component = component
            
            # Step 2: Get current plasma concentration from PK component
            if pk_component:
                fluxes.plasma_concentration = pk_component.pk_model.get_plasma_concentration()
            
            # Step 3: Update component states from current state vector
            current_idx = 0
            for component in self.components:
                component_state = y[current_idx:current_idx + component.n_states]
                
                # DEBUG: Check component type detection (only show first time)
                # if current_idx == 0:  # First component (lung)
                #     print("        DEBUG COMPONENT:", type(component).__name__, "has_set_state:", hasattr(component, 'set_state'), "has_lung_entities:", hasattr(component, 'lung_entities'))
                
                if hasattr(component, 'set_state'):
                    component.set_state(component_state)
                elif hasattr(component, 'lung_entities'):  # Lung component
                    # Update all lung entities with their portion of the state
                    entity_idx = 0
                    for entity in component.lung_entities:
                        entity_state_size = entity.get_state_size()
                        entity_state = component_state[entity_idx:entity_idx + entity_state_size]
                        
                        # DEBUG: Show ET entity state sync only if epithelium should be non-zero
                        #if entity.entity_name == "ET" and entity_state_size > 1 and entity_state[1] > 1e-10:
                        #    print("        STATE SYNC ET: state[1]=", entity_state[1], "old_epi=", entity.amount_epithelium_shallow[0])
                        
                        entity.set_state(entity_state)
                        
                        # DEBUG: Show if state sync worked
                        #if entity.entity_name == "ET" and entity_state_size > 1 and entity_state[1] > 1e-10:
                        #    print("        AFTER SYNC ET: new_epi=", entity.amount_epithelium_shallow[0])
                        
                        entity_idx += entity_state_size
                current_idx += component.n_states
            
            # Step 4: Compute derivatives for all components with consistent flux state
            derivatives = np.zeros_like(y)
            current_idx = 0
            
            # Reset fluxes that will be accumulated
            fluxes.lung_to_systemic = 0.0
            fluxes.gi_to_systemic = 0.0
            fluxes.mcc_to_gi = 0.0
            fluxes.mcc_bb_to_BB = 0.0
            
            lung_comp_for_trace = None
            for component in self.components:
                comp_derivs, fluxes = component.compute_derivatives(fluxes, external_inputs)
                
                derivatives[current_idx:current_idx + component.n_states] = comp_derivs
                current_idx += component.n_states
                if isinstance(component, LungComponent):
                    lung_comp_for_trace = component

            # Omit raw flux recording for performance; flux will be computed at t_eval after solve
            
            return derivatives
        
        return ode_system
    
    def get_initial_state(self, initial_values: Optional[Dict[str, float]] = None) -> np.ndarray:
        """Get initial state vector."""
        
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
        """Extract structured results."""
        
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
        """Get model information."""
        
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
        """
        Solve the PBPK ODE system using the proper solver infrastructure.
        
        Args:
            time_points: Time points for integration (in seconds)
            **solver_kwargs: Additional arguments for the solver
            
        Returns:
            Results object with comprehensive analysis
        """
        from scipy.integrate import solve_ivp
        
        # Prepare flux recorder at t_eval only
        self._t_eval = time_points

        # Get ODE system and initial state
        ode_func = self.get_ode_system()
        y0 = self.get_initial_state(solver_kwargs.pop('initial_conditions', {}))
        
        # Silence debug prints in production
        
        try:
            # Use scipy directly to solve the ODE system
            t_span = (time_points[0], time_points[-1])
            
            # Solver options (tuned for stiffness and finite-difference stability)
            solver_options = {
                # Radau is robust for stiff systems and less prone to FD overflow
                'method': 'Radau',
                'rtol': 1e-6,
                'atol': 1e-8, #1e-10,
                'max_step': 300.0,
                'first_step': 0.01,
            }
            solver_options.update(solver_kwargs)
            
            sol = solve_ivp(
                ode_func, t_span, y0, 
                t_eval=time_points,
                **solver_options
            )
            
            
            # Extract comprehensive results
            results_data = self.extract_results(sol.t, sol.y.T)
            # Compute flux time series at t_eval from solution states
            try:
                t_eval = sol.t
                y_mat = sol.y.T  # shape (N, n_states)
                per_region: Dict[str, np.ndarray] = {}
                per_region_mcc: Dict[str, np.ndarray] = {}
                total_lung_to_systemic = np.zeros_like(t_eval)
                gi_to_systemic = np.zeros_like(t_eval)
                mcc_total = np.zeros_like(t_eval)

                # Identify components and sizes
                lung_comp = next((c for c in self.components if isinstance(c, LungComponent)), None)
                gi_comp = next((c for c in self.components if isinstance(c, GIComponent)), None)
                pk_comp = next((c for c in self.components if isinstance(c, PKComponent)), None)

                lung_start = 0
                lung_size = lung_comp.n_states if lung_comp else 0
                gi_start = lung_start + lung_size
                gi_size = gi_comp.n_states if gi_comp else 0
                pk_start = gi_start + gi_size
                pk_size = pk_comp.n_states if pk_comp else 0

                # Prepare per-region arrays
                if lung_comp:
                    for region in lung_comp.regions:
                        per_region[region] = np.zeros_like(t_eval)
                        per_region_mcc[region] = np.zeros_like(t_eval)

                # Get vd_central for plasma concentration
                vd_central = None
                if pk_comp and hasattr(pk_comp.pk_model, 'vd_central'):
                    vd_central = float(getattr(pk_comp.pk_model, 'vd_central'))
                if vd_central is None:
                    vd_central = float(getattr(self.api_params, 'volume_central_L', 29.92))

                # Iterate time grid and compute fluxes
                for ti in range(len(t_eval)):
                    y_row = y_mat[ti]

                    # Plasma concentration (pmol/L) from central amount
                    plasma_conc = 0.0
                    if pk_size > 0:
                        central_amt = y_row[pk_start]
                        plasma_conc = max(0.0, central_amt) / vd_central if vd_central > 0 else 0.0

                    # LUNG fluxes per region
                    mcc_to_gi_flow = 0.0
                    mcc_bb_to_BB_flow = 0.0
                    if lung_comp and lung_size > 0:
                        current_idx = lung_start
                        for r_idx, (entity, region) in enumerate(zip(lung_comp.lung_entities, lung_comp.regions)):
                            entity_state_size = entity.get_state_size()
                            entity_state = y_row[current_idx:current_idx + entity_state_size]
                            entity.set_state(entity_state)
                            # External MCC routing: BB receives from bb
                            external_mcc_input = 0.0
                            if region == 'BB':
                                external_mcc_input = mcc_bb_to_BB_flow
                            derivs, mcc_output, sys_abs = entity.compute_derivatives(plasma_conc, 0.0, external_mcc_input)
                            total_lung_to_systemic[ti] += sys_abs
                            per_region[region][ti] = sys_abs
                            per_region_mcc[region][ti] = mcc_output
                            mcc_total[ti] += mcc_output
                            if region == 'Al':
                                pass
                            elif region == 'bb':
                                mcc_bb_to_BB_flow += mcc_output
                            else:
                                mcc_to_gi_flow += mcc_output
                            current_idx += entity_state_size

                    # GI absorption
                    if gi_comp and gi_size > 0:
                        gi_state = y_row[gi_start:gi_start + gi_size]
                        gi_comp.gi_model.set_state(gi_state)
                        _, gi_abs, _ = gi_comp.gi_model.compute_derivatives(mcc_to_gi_flow, plasma_conc)
                        gi_to_systemic[ti] = gi_abs

                self._flux_rec = {
                    'total_lung_to_systemic': total_lung_to_systemic,
                    'gi_to_systemic': gi_to_systemic,
                    'per_region': per_region,
                    'mcc_total': mcc_total,
                    'per_region_mcc': per_region_mcc,
                }
                results_data['flux'] = self._flux_rec
            except Exception:
                # If flux post-processing fails, proceed without flux
                pass
            
            # Return raw scipy results (bypass PBBMResults import issues)
            # Add useful attributes for mass balance calculation
            sol.results_data = results_data
            sol.state_matrix = sol.y.T
            sol.time_points = sol.t
            sol.model_info = self.get_model_info()
            sol.molecular_weight = self.api_params.molecular_weight
            
            return sol
            
        except Exception as e:
            # Silence errors already raised
            raise
