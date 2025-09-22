"""
Vectorized Lung Model

This wrapper creates a unified interface for lung PBBM that handles multiple 
regions/generations using vectorized operations, regardless of whether it's 
regional (4 entities) or generational (25 entities).

Key features:
- Single interface for both regional and generational models
- Vectorized operations for efficiency
- Parameters passed as vectors with length n
"""

import numpy as np
from typing import Dict, List, Any, Tuple, Union
from .lung_pbbm import create_regional_lung_entities, create_generational_lung_entities


class VectorizedLungModel:
    """
    Unified vectorized lung model for both regional and generational approaches.
    
    Handles multiple lung entities (regions or generations) using vectorized operations
    where parameters are passed as vectors with length n.
    """
    
    def __init__(self, subject, api, deposition_settings, model_type="regional"):
        """
        Initialize vectorized lung model.
        
        Args:
            subject: Subject domain object (from entities.py)
            api: API domain object (from entities.py) 
            deposition_settings: Deposition settings dict
            model_type: "regional" (4 entities) or "generational" (25 entities)
        """
        self.subject = subject
        self.api = api
        self.model_type = model_type
        
        # Create lung entities
        if model_type == "regional":
            self.entities = create_regional_lung_entities(subject, api, deposition_settings)
            self.regions = ['ET', 'BB', 'bb', 'Al']
        elif model_type == "generational":
            self.entities = create_generational_lung_entities(subject, api, deposition_settings)  
            self.regions = [f'gen_{i}' for i in range(len(self.entities))]
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        self.n_entities = len(self.entities)
        
        # Get unified parameters
        self.entity_state_sizes = np.array([entity.get_state_size() for entity in self.entities])
        self.total_state_size = np.sum(self.entity_state_sizes)
        self.state_start_indices = np.cumsum(np.concatenate([[0], self.entity_state_sizes[:-1]]))
        
        print(f"VectorizedLungModel: {model_type} with {self.n_entities} entities, total_state_size={self.total_state_size}")
    
    def get_state_size(self) -> int:
        """Total state size for all entities."""
        return self.total_state_size
    
    def get_n_entities(self) -> int:
        """Number of entities (regions or generations)."""
        return self.n_entities
    
    def get_entity_names(self) -> List[str]:
        """Get entity names."""
        return self.regions
    
    def set_state(self, state_vector: np.ndarray):
        """
        Set state for all entities from unified state vector.
        
        Args:
            state_vector: Full lung state vector with length total_state_size
        """
        for i, entity in enumerate(self.entities):
            start_idx = self.state_start_indices[i]
            end_idx = start_idx + self.entity_state_sizes[i]
            entity_state = state_vector[start_idx:end_idx]
            
            # Update entity state
            self._update_entity_state(entity, entity_state)
    
    def _update_entity_state(self, entity, entity_state):
        """Update individual entity state from state vector."""
        state_idx = 0
        
        # ELF compartment
        entity.amount_elf = entity_state[state_idx]
        state_idx += 1
        
        # Epithelium layers
        for i in range(entity.n_epithelium_layers):
            entity.amount_epithelium_shallow[i] = entity_state[state_idx]
            state_idx += 1
        
        for i in range(entity.n_epithelium_layers):
            entity.amount_epithelium_deep[i] = entity_state[state_idx]
            state_idx += 1
        
        # Tissue compartments
        entity.amount_tissue_shallow = entity_state[state_idx]
        state_idx += 1
        entity.amount_tissue_deep = entity_state[state_idx]
        state_idx += 1
        
        # Dissolution bins
        for i in range(entity.n_dissolution_bins):
            entity.particle_radii[i] = entity_state[state_idx]
            state_idx += 1
        for i in range(entity.n_dissolution_bins):
            entity.particle_counts[i] = entity_state[state_idx]
            state_idx += 1
    
    def compute_derivatives(self, plasma_concentration: float, 
                          deposition_inputs: np.ndarray, 
                          external_mcc_inputs: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute derivatives for all entities using vectorized approach.
        
        Args:
            plasma_concentration: Current plasma concentration (scalar)
            deposition_inputs: Deposition inputs for each entity (vector length n)
            external_mcc_inputs: External MCC inputs for each entity (vector length n, optional)
        
        Returns:
            tuple: (derivatives_vector, mcc_outputs_vector, systemic_absorption_vector)
        """
        if external_mcc_inputs is None:
            external_mcc_inputs = np.zeros(self.n_entities)
        
        # Ensure inputs are correct length
        assert len(deposition_inputs) == self.n_entities, f"deposition_inputs must have length {self.n_entities}"
        assert len(external_mcc_inputs) == self.n_entities, f"external_mcc_inputs must have length {self.n_entities}"
        
        # Compute derivatives for each entity
        all_derivatives = np.zeros(self.total_state_size)
        mcc_outputs = np.zeros(self.n_entities)
        systemic_absorptions = np.zeros(self.n_entities)
        
        for i, entity in enumerate(self.entities):
            start_idx = self.state_start_indices[i]
            end_idx = start_idx + self.entity_state_sizes[i]
            
            # Compute derivatives for this entity
            entity_derivs, mcc_out, systemic_abs = entity.compute_derivatives(
                plasma_concentration, 
                deposition_inputs[i], 
                external_mcc_inputs[i]
            )
            
            # Store results
            all_derivatives[start_idx:end_idx] = entity_derivs
            mcc_outputs[i] = mcc_out
            systemic_absorptions[i] = systemic_abs
        
        return all_derivatives, mcc_outputs, systemic_absorptions
    
    def get_initial_state(self, initial_values: Dict[str, float] = None) -> np.ndarray:
        """
        Get initial state vector for all entities.
        
        Args:
            initial_values: Initial values dict (optional)
        
        Returns:
            Initial state vector with length total_state_size
        """
        y0 = np.zeros(self.total_state_size)
        
        # Initialize dissolution radii for each entity
        for i, entity in enumerate(self.entities):
            start_idx = self.state_start_indices[i]
            n_epi = entity.n_epithelium_layers
            n_bins = entity.n_dissolution_bins
            
            # Radii start after: ELF(1) + Epi_shallow(n) + Epi_deep(n) + Tissue_shallow(1) + Tissue_deep(1)
            radii_start = start_idx + 1 + 2*n_epi + 2
            
            for j in range(n_bins):
                y0[radii_start + j] = entity.initial_radii[j]
        
        return y0
    
    def get_total_outputs(self, mcc_outputs: np.ndarray, systemic_absorptions: np.ndarray) -> Dict[str, float]:
        """
        Get total outputs from vectorized results.
        
        Args:
            mcc_outputs: MCC outputs from each entity
            systemic_absorptions: Systemic absorptions from each entity
        
        Returns:
            Dictionary with total outputs
        """
        # For regional model, only ET and BB contribute to GI
        if self.model_type == "regional":
            # ET and BB are indices 0 and 1
            total_mcc_to_gi = np.sum(mcc_outputs[:2])  # Only ET and BB
        else:
            # For generational model, upper generations contribute to GI
            # (simplified - would need proper generational routing)
            total_mcc_to_gi = np.sum(mcc_outputs[:10])  # First 10 generations
        
        total_systemic_absorption = np.sum(systemic_absorptions)
        
        return {
            'total_mcc_to_gi': total_mcc_to_gi,
            'total_systemic_absorption': total_systemic_absorption,
            'mcc_by_entity': mcc_outputs,
            'absorption_by_entity': systemic_absorptions
        }


def create_vectorized_lung_model(subject, api, deposition_settings, model_type="regional"):
    """
    Factory function to create vectorized lung model.
    
    Args:
        subject: Subject domain object (from entities.py)
        api: API domain object (from entities.py)
        deposition_settings: Deposition settings dict
        model_type: "regional" or "generational"
    
    Returns:
        VectorizedLungModel instance
    """
    return VectorizedLungModel(subject, api, deposition_settings, model_type)
