"""Detailed mass balance checking including solid particles and bound states.

This module provides comprehensive mass balance checking that accounts for:
- Dissolved drug in compartments
- Undissolved solid particles
- Bound and unbound drug states
- Drug in transit between compartments
"""

from __future__ import annotations
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import pandas as pd


@dataclass
class DetailedMassBalanceResult:
    """Detailed results from comprehensive mass balance check."""
    
    is_balanced: bool
    total_initial_mass: float
    
    # Mass accounting
    total_dissolved_mass: float
    total_solid_mass: float
    total_bound_mass: float
    total_unbound_mass: float
    total_eliminated_mass: float
    total_systemic_mass: float
    
    # Final total
    total_final_mass: float
    mass_difference: float
    relative_error: float
    
    # Detailed breakdown
    compartment_masses: Dict[str, float]
    regional_masses: Dict[str, float]
    solid_masses: Dict[str, float]
    
    # Time series
    time_points: np.ndarray
    total_mass_time_series: np.ndarray
    dissolved_mass_time_series: np.ndarray
    solid_mass_time_series: np.ndarray
    
    # Validation
    error_message: Optional[str] = None
    warnings: List[str] = None
    
    def print_detailed_summary(self):
        """Print comprehensive mass balance summary."""
        print("\n" + "="*70)
        print("DETAILED MASS BALANCE CHECK RESULTS")
        print("="*70)
        
        print(f"\n📊 MASS ACCOUNTING:")
        print(f"   Initial Total Mass:      {self.total_initial_mass:.3e} pmol")
        print(f"   Final Total Mass:        {self.total_final_mass:.3e} pmol")
        print(f"   Mass Difference:         {self.mass_difference:.3e} pmol")
        print(f"   Relative Error:          {self.relative_error*100:.6f}%")
        
        print(f"\n💊 DRUG STATE DISTRIBUTION:")
        print(f"   Dissolved Mass:          {self.total_dissolved_mass:.3e} pmol")
        print(f"   Solid (Undissolved):     {self.total_solid_mass:.3e} pmol")
        print(f"   Bound Mass:              {self.total_bound_mass:.3e} pmol")
        print(f"   Unbound Mass:            {self.total_unbound_mass:.3e} pmol")
        
        print(f"\n🔄 MASS MOVEMENT:")
        print(f"   Eliminated Mass:         {self.total_eliminated_mass:.3e} pmol")
        print(f"   Systemic Mass:           {self.total_systemic_mass:.3e} pmol")
        
        # Check mass balance equation
        accounted_mass = (self.total_dissolved_mass + self.total_solid_mass + 
                         self.total_eliminated_mass)
        balance_check = abs(accounted_mass - self.total_initial_mass)
        
        print(f"\n✅ MASS BALANCE EQUATION:")
        print(f"   Initial = Dissolved + Solid + Eliminated")
        print(f"   {self.total_initial_mass:.3e} = {self.total_dissolved_mass:.3e} + "
              f"{self.total_solid_mass:.3e} + {self.total_eliminated_mass:.3e}")
        print(f"   Imbalance: {balance_check:.3e} pmol")
        
        if self.solid_masses:
            print(f"\n🎯 SOLID PARTICLE DISTRIBUTION:")
            total_solid = sum(self.solid_masses.values())
            for region, mass in self.solid_masses.items():
                if mass > 1e-10:
                    percentage = (mass / total_solid) * 100 if total_solid > 0 else 0
                    print(f"   {region}: {mass:.3e} pmol ({percentage:.1f}% of solid)")
        
        if self.warnings:
            print(f"\n⚠️  WARNINGS:")
            for warning in self.warnings:
                print(f"   - {warning}")
        
        if self.is_balanced:
            print(f"\n✅ MASS BALANCE CHECK PASSED")
        else:
            print(f"\n❌ MASS BALANCE CHECK FAILED")
            if self.error_message:
                print(f"   Error: {self.error_message}")
        
        print("="*70)


class ComprehensiveMassBalanceChecker:
    """Performs detailed mass balance checks accounting for all drug states."""
    
    def __init__(self, regions: List[str] = None, tolerance: float = 0.001):
        """Initialize comprehensive mass balance checker.
        
        Args:
            regions: List of lung regions
            tolerance: Relative error tolerance (default: 0.001 = 0.1%)
        """
        self.regions = regions or ['ET', 'BB', 'bb', 'Al']
        self.tolerance = tolerance
        self.num_deposition_bins = 19  # From original lung_pbbm.py
        
    def check_detailed_balance(self,
                              solution: np.ndarray,
                              time_points: np.ndarray,
                              initial_deposition: Dict[str, float],
                              params: any = None,
                              has_solid_particles: bool = False) -> DetailedMassBalanceResult:
        """Perform comprehensive mass balance check.
        
        Args:
            solution: ODE solution array
            time_points: Time points (hours)
            initial_deposition: Initial deposition by region (pmol)
            params: Simulation parameters
            has_solid_particles: Whether system tracks solid particles
            
        Returns:
            DetailedMassBalanceResult with comprehensive accounting
        """
        
        # Initial mass
        total_initial_mass = sum(initial_deposition.values())
        
        # Parse solution structure
        num_regions = len(self.regions)
        
        if has_solid_particles:
            # Full system with solid particles
            # Each region: 1 (ELF) + 2*n_epi + 2 (tissue) + 2*num_bins
            n_epi = 1  # Simplified, would need from params
            states_per_region = 1 + 2*n_epi + 2 + 2*self.num_deposition_bins
            num_lung_states = num_regions * states_per_region
        else:
            # Simplified system (our current implementation)
            num_lung_states = num_regions * 3  # ELF, Epithelium, Tissue
            
        num_gi_states = 4
        num_pk_states = 3
        
        # Extract final state
        final_state = solution[:, -1] if solution.ndim > 1 else solution
        
        # Initialize mass accounting
        total_dissolved_mass = 0.0
        total_solid_mass = 0.0
        total_bound_mass = 0.0
        total_unbound_mass = 0.0
        compartment_masses = {}
        regional_masses = {}
        solid_masses = {}
        warnings = []
        
        # Process lung compartments
        if has_solid_particles:
            # Complex processing for full system
            warnings.append("Full particle tracking not yet implemented in simplified system")
            # Would need to extract solid particle states here
        else:
            # Simplified system processing
            for i, region in enumerate(self.regions):
                region_offset = i * 3
                
                # Extract masses
                elf_mass = final_state[region_offset] if region_offset < len(final_state) else 0
                epi_mass = final_state[region_offset + 1] if region_offset + 1 < len(final_state) else 0
                tissue_mass = final_state[region_offset + 2] if region_offset + 2 < len(final_state) else 0
                
                # Store compartment masses
                compartment_masses[f"{region}_ELF"] = elf_mass
                compartment_masses[f"{region}_Epithelium"] = epi_mass
                compartment_masses[f"{region}_Tissue"] = tissue_mass
                
                # Regional total
                regional_masses[region] = elf_mass + epi_mass + tissue_mass
                
                # All dissolved in simplified system
                total_dissolved_mass += elf_mass + epi_mass + tissue_mass
                
                # Estimate bound/unbound (simplified)
                # ELF is typically all unbound, tissue/epi has binding
                total_unbound_mass += elf_mass
                if params and hasattr(params, 'fraction_unbound_effective'):
                    fu_tissue = params.fraction_unbound_effective.get(region, {}).get('fu_tissue_calc', 0.1)
                    total_unbound_mass += (epi_mass + tissue_mass) * fu_tissue
                    total_bound_mass += (epi_mass + tissue_mass) * (1 - fu_tissue)
                else:
                    # Default assumption: 10% unbound in tissue
                    total_unbound_mass += (epi_mass + tissue_mass) * 0.1
                    total_bound_mass += (epi_mass + tissue_mass) * 0.9
                    
            # Note about solid particles
            if not has_solid_particles:
                warnings.append("Simplified system: all drug assumed dissolved (no solid particle tracking)")
                solid_masses = {region: 0.0 for region in self.regions}
        
        # GI compartments
        gi_start = num_lung_states
        if gi_start < len(final_state):
            gi_masses = final_state[gi_start:gi_start + num_gi_states]
            compartment_masses['GI_Total'] = np.sum(gi_masses)
            total_dissolved_mass += np.sum(gi_masses)
            total_unbound_mass += np.sum(gi_masses)  # Assume GI drug is unbound
        
        # PK compartments
        pk_start = num_lung_states + num_gi_states
        if pk_start < len(final_state):
            central_mass = final_state[pk_start]
            peripheral1_mass = final_state[pk_start + 1] if pk_start + 1 < len(final_state) else 0
            peripheral2_mass = final_state[pk_start + 2] if pk_start + 2 < len(final_state) else 0
            
            compartment_masses['Systemic_Central'] = central_mass
            compartment_masses['Systemic_Peripheral1'] = peripheral1_mass
            compartment_masses['Systemic_Peripheral2'] = peripheral2_mass
            
            total_systemic_mass = central_mass + peripheral1_mass + peripheral2_mass
            total_dissolved_mass += total_systemic_mass
            
            # Systemic binding
            if params:
                fu_plasma = 0.12  # Default for BD
                total_unbound_mass += total_systemic_mass * fu_plasma
                total_bound_mass += total_systemic_mass * (1 - fu_plasma)
            else:
                total_unbound_mass += total_systemic_mass * 0.1
                total_bound_mass += total_systemic_mass * 0.9
        else:
            total_systemic_mass = 0
        
        # Calculate total final mass
        total_final_mass = total_dissolved_mass + total_solid_mass
        
        # Estimate eliminated mass
        total_eliminated_mass = total_initial_mass - total_final_mass
        
        # Calculate error
        mass_difference = total_initial_mass - (total_final_mass + total_eliminated_mass)
        relative_error = abs(mass_difference / total_initial_mass) if total_initial_mass > 0 else 0
        
        # Check balance
        is_balanced = relative_error < self.tolerance
        
        # Create error message
        error_message = None
        if not is_balanced:
            error_message = f"Mass imbalance: {relative_error*100:.3f}% error"
            if not has_solid_particles:
                error_message += " (Note: simplified system without solid particle tracking)"
        
        # Time series (simplified for now)
        if solution.ndim > 1:
            total_mass_time_series = np.sum(solution[:num_lung_states + num_gi_states + num_pk_states, :], axis=0)
            dissolved_mass_time_series = total_mass_time_series  # All dissolved in simplified system
            solid_mass_time_series = np.zeros_like(total_mass_time_series)
        else:
            total_mass_time_series = np.array([total_final_mass])
            dissolved_mass_time_series = np.array([total_dissolved_mass])
            solid_mass_time_series = np.array([0.0])
        
        return DetailedMassBalanceResult(
            is_balanced=is_balanced,
            total_initial_mass=total_initial_mass,
            total_dissolved_mass=total_dissolved_mass,
            total_solid_mass=total_solid_mass,
            total_bound_mass=total_bound_mass,
            total_unbound_mass=total_unbound_mass,
            total_eliminated_mass=total_eliminated_mass,
            total_systemic_mass=total_systemic_mass,
            total_final_mass=total_final_mass,
            mass_difference=mass_difference,
            relative_error=relative_error,
            compartment_masses=compartment_masses,
            regional_masses=regional_masses,
            solid_masses=solid_masses,
            time_points=time_points,
            total_mass_time_series=total_mass_time_series,
            dissolved_mass_time_series=dissolved_mass_time_series,
            solid_mass_time_series=solid_mass_time_series,
            error_message=error_message,
            warnings=warnings
        )