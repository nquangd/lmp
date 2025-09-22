"""Mass balance checking for lung PBBM simulations.

This module provides comprehensive mass balance checking to ensure
conservation of mass throughout the ODE simulation.
"""

from __future__ import annotations
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import pandas as pd


@dataclass
class MassBalanceResult:
    """Results from mass balance check."""
    
    is_balanced: bool
    total_initial_mass: float
    total_final_mass: float
    mass_eliminated: float
    mass_absorbed: float
    mass_difference: float
    relative_error: float
    
    # Detailed compartment masses
    compartment_masses: Dict[str, float]
    regional_masses: Dict[str, float]
    
    # Time series data
    time_points: np.ndarray
    total_mass_time_series: np.ndarray
    cumulative_elimination: np.ndarray
    cumulative_absorption: np.ndarray
    
    # Error details
    error_message: Optional[str] = None
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert mass balance results to DataFrame."""
        data = {
            'Metric': [
                'Initial Mass (pmol)',
                'Final Mass in System (pmol)',
                'Mass Eliminated (pmol)',
                'Mass Absorbed to Systemic (pmol)',
                'Total Accounted Mass (pmol)',
                'Mass Difference (pmol)',
                'Relative Error (%)',
                'Is Balanced?'
            ],
            'Value': [
                self.total_initial_mass,
                self.total_final_mass,
                self.mass_eliminated,
                self.mass_absorbed,
                self.total_final_mass + self.mass_eliminated,
                self.mass_difference,
                self.relative_error * 100,
                'Yes' if self.is_balanced else 'No'
            ]
        }
        return pd.DataFrame(data)
    
    def print_summary(self):
        """Print a formatted summary of mass balance results."""
        print("\n" + "="*60)
        print("MASS BALANCE CHECK RESULTS")
        print("="*60)
        
        print(f"\nInitial Mass:          {self.total_initial_mass:.2e} pmol")
        print(f"Final Mass in System:  {self.total_final_mass:.2e} pmol")
        print(f"Mass Eliminated:       {self.mass_eliminated:.2e} pmol")
        print(f"Mass Absorbed:         {self.mass_absorbed:.2e} pmol")
        print(f"Total Accounted:       {(self.total_final_mass + self.mass_eliminated):.2e} pmol")
        print(f"\nMass Difference:       {self.mass_difference:.2e} pmol")
        print(f"Relative Error:        {self.relative_error*100:.4f}%")
        
        if self.is_balanced:
            print(f"\n✅ MASS BALANCE CHECK PASSED (error < 0.1%)")
        else:
            print(f"\n❌ MASS BALANCE CHECK FAILED (error > 0.1%)")
            if self.error_message:
                print(f"   Error: {self.error_message}")
        
        # Regional breakdown
        print(f"\n📍 Regional Mass Distribution (final):")
        for region, mass in self.regional_masses.items():
            percentage = (mass / self.total_initial_mass) * 100 if self.total_initial_mass > 0 else 0
            print(f"   {region}: {mass:.2e} pmol ({percentage:.1f}%)")
        
        # Compartment breakdown
        print(f"\n📦 Compartment Mass Distribution (final):")
        for comp, mass in self.compartment_masses.items():
            if mass > 1e-10:  # Only show non-zero compartments
                percentage = (mass / self.total_initial_mass) * 100 if self.total_initial_mass > 0 else 0
                print(f"   {comp}: {mass:.2e} pmol ({percentage:.1f}%)")
        
        print("="*60)


class MassBalanceChecker:
    """Performs mass balance checks on PBBM simulations."""
    
    def __init__(self, regions: List[str] = None, tolerance: float = 0.001):
        """Initialize mass balance checker.
        
        Args:
            regions: List of lung regions (default: ['ET', 'BB', 'bb', 'Al'])
            tolerance: Relative error tolerance for balance check (default: 0.001 = 0.1%)
        """
        self.regions = regions or ['ET', 'BB', 'bb', 'Al']
        self.tolerance = tolerance
        
    def check_balance(self, 
                     solution: np.ndarray,
                     time_points: np.ndarray,
                     initial_deposition: Dict[str, float],
                     elimination_rate: float = None,
                     params: any = None) -> MassBalanceResult:
        """Check mass balance for a completed simulation.
        
        Args:
            solution: ODE solution array [states x time_points]
            time_points: Time points of solution (hours)
            initial_deposition: Initial drug deposition by region (pmol)
            elimination_rate: Elimination rate constant (1/h)
            params: Simulation parameters
            
        Returns:
            MassBalanceResult with detailed balance information
        """
        
        # Calculate initial total mass
        total_initial_mass = sum(initial_deposition.values())
        
        # Parse solution structure
        num_regions = len(self.regions)
        num_lung_states = num_regions * 3  # 3 compartments per region
        num_gi_states = 4
        num_pk_states = 3
        
        # Extract final state
        final_state = solution[:, -1] if solution.ndim > 1 else solution
        
        # Calculate masses in each compartment at final time
        compartment_masses = {}
        regional_masses = {}
        
        # Lung compartments
        for i, region in enumerate(self.regions):
            region_offset = i * 3
            elf_mass = final_state[region_offset] if region_offset < len(final_state) else 0
            epi_mass = final_state[region_offset + 1] if region_offset + 1 < len(final_state) else 0
            tissue_mass = final_state[region_offset + 2] if region_offset + 2 < len(final_state) else 0
            
            compartment_masses[f"{region}_ELF"] = elf_mass
            compartment_masses[f"{region}_Epithelium"] = epi_mass
            compartment_masses[f"{region}_Tissue"] = tissue_mass
            
            regional_masses[region] = elf_mass + epi_mass + tissue_mass
        
        # GI compartments
        gi_start = num_lung_states
        gi_masses = final_state[gi_start:gi_start + num_gi_states] if gi_start < len(final_state) else np.zeros(num_gi_states)
        compartment_masses['GI_Total'] = np.sum(gi_masses)
        
        # PK compartments (systemic)
        pk_start = num_lung_states + num_gi_states
        if pk_start < len(final_state):
            central_mass = final_state[pk_start]
            peripheral1_mass = final_state[pk_start + 1] if pk_start + 1 < len(final_state) else 0
            peripheral2_mass = final_state[pk_start + 2] if pk_start + 2 < len(final_state) else 0
            
            compartment_masses['Systemic_Central'] = central_mass
            compartment_masses['Systemic_Peripheral1'] = peripheral1_mass
            compartment_masses['Systemic_Peripheral2'] = peripheral2_mass
        else:
            central_mass = peripheral1_mass = peripheral2_mass = 0
        
        # Calculate total mass in system
        total_final_mass = np.sum(list(compartment_masses.values()))
        
        # Calculate cumulative elimination over time
        if elimination_rate is not None and solution.ndim > 1:
            # Track central compartment over time
            central_time_series = solution[pk_start, :] if pk_start < len(solution) else np.zeros(len(time_points))
            
            # Calculate elimination flux at each time point
            elimination_flux = elimination_rate * central_time_series  # pmol/h
            
            # Integrate to get cumulative elimination
            dt = np.diff(time_points)
            dt = np.append(dt, dt[-1] if len(dt) > 0 else 1)
            cumulative_elimination = np.cumsum(elimination_flux * dt)
            
            mass_eliminated = cumulative_elimination[-1]
        else:
            cumulative_elimination = np.zeros(len(time_points))
            mass_eliminated = 0
        
        # Calculate mass absorbed to systemic circulation
        mass_absorbed = central_mass + peripheral1_mass + peripheral2_mass
        
        # Calculate total mass time series
        if solution.ndim > 1:
            total_mass_time_series = np.sum(solution[:num_lung_states + num_gi_states + num_pk_states, :], axis=0)
        else:
            total_mass_time_series = np.array([total_final_mass])
        
        # Calculate cumulative absorption (mass that left lung/GI and entered systemic)
        lung_gi_mass_time = np.sum(solution[:num_lung_states + num_gi_states, :], axis=0) if solution.ndim > 1 else np.array([0])
        cumulative_absorption = total_initial_mass - lung_gi_mass_time
        
        # Calculate mass balance
        total_accounted = total_final_mass + mass_eliminated
        mass_difference = total_initial_mass - total_accounted
        relative_error = abs(mass_difference / total_initial_mass) if total_initial_mass > 0 else 0
        
        # Check if balanced
        is_balanced = relative_error < self.tolerance
        
        # Create error message if not balanced
        error_message = None
        if not is_balanced:
            if relative_error > 0.1:  # More than 10% error
                error_message = f"Large mass imbalance detected: {relative_error*100:.2f}% error"
            elif mass_difference > 0:
                error_message = f"Mass loss detected: {mass_difference:.2e} pmol unaccounted"
            else:
                error_message = f"Mass gain detected: {abs(mass_difference):.2e} pmol created"
        
        return MassBalanceResult(
            is_balanced=is_balanced,
            total_initial_mass=total_initial_mass,
            total_final_mass=total_final_mass,
            mass_eliminated=mass_eliminated,
            mass_absorbed=mass_absorbed,
            mass_difference=mass_difference,
            relative_error=relative_error,
            compartment_masses=compartment_masses,
            regional_masses=regional_masses,
            time_points=time_points,
            total_mass_time_series=total_mass_time_series,
            cumulative_elimination=cumulative_elimination,
            cumulative_absorption=cumulative_absorption,
            error_message=error_message
        )
    
    def plot_mass_balance(self, result: MassBalanceResult, save_path: Optional[str] = None):
        """Plot mass balance over time.
        
        Args:
            result: MassBalanceResult from check_balance
            save_path: Optional path to save figure
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("Warning: matplotlib not available for plotting")
            return
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
        
        # Plot 1: Total mass over time
        ax1.plot(result.time_points, result.total_mass_time_series, 'b-', linewidth=2)
        ax1.axhline(y=result.total_initial_mass, color='r', linestyle='--', label='Initial Mass')
        ax1.set_xlabel('Time (hours)')
        ax1.set_ylabel('Total Mass (pmol)')
        ax1.set_title('Total System Mass Over Time')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Cumulative elimination
        ax2.plot(result.time_points, result.cumulative_elimination, 'r-', linewidth=2)
        ax2.set_xlabel('Time (hours)')
        ax2.set_ylabel('Cumulative Elimination (pmol)')
        ax2.set_title('Cumulative Drug Elimination')
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Cumulative absorption
        ax3.plot(result.time_points, result.cumulative_absorption, 'g-', linewidth=2)
        ax3.set_xlabel('Time (hours)')
        ax3.set_ylabel('Cumulative Absorption (pmol)')
        ax3.set_title('Cumulative Systemic Absorption')
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Mass balance error over time
        mass_accounted = result.total_mass_time_series + result.cumulative_elimination
        balance_error = (result.total_initial_mass - mass_accounted) / result.total_initial_mass * 100
        ax4.plot(result.time_points, balance_error, 'k-', linewidth=2)
        ax4.axhline(y=0, color='g', linestyle='-', alpha=0.5)
        ax4.axhline(y=0.1, color='r', linestyle='--', alpha=0.5, label='±0.1% tolerance')
        ax4.axhline(y=-0.1, color='r', linestyle='--', alpha=0.5)
        ax4.set_xlabel('Time (hours)')
        ax4.set_ylabel('Mass Balance Error (%)')
        ax4.set_title('Mass Balance Error Over Time')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.suptitle(f'Mass Balance Analysis - {"PASSED ✓" if result.is_balanced else "FAILED ✗"}', 
                     fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Mass balance plot saved to: {save_path}")
        
        plt.show()
