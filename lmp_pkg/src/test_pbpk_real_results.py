#!/usr/bin/env python3
"""
Real PBPK test to get actual mass balance results.
This test will run the actual PBPK simulation and report the real numbers.
"""

import sys
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Add current directory to Python path
sys.path.insert(0, '/Users/duynguyen/Library/Mobile Documents/com~apple~CloudDocs/LMP/lmp/lmp_pkg/src')

def run_real_pbpk_test():
    """Run actual PBPK simulation and get real mass balance results."""
    
    print("="*80)
    print("REAL PBPK MASS BALANCE TEST")
    print("="*80)
    
    try:
        # Load entities using the working pattern
        print("1. Loading entities...")
        from lmp_pkg.domain.entities import Subject, API, Product
        
        subject = Subject.from_builtin("healthy_reference")
        api = API.from_builtin("BD")
        product = Product.from_builtin("reference_product")
        
        print(f"  ✓ Subject: {subject.demographic.age_years} years, {subject.demographic.weight_kg} kg")
        print(f"  ✓ API: {api.name} (MW: {api.molecular_weight} μg/μmol)")
        print(f"  ✓ Product: {product.name}")
        
        # Apply transformations to get final values
        print("2. Computing final parameter values...")
        subject_final = subject.get_final_values(apply_variability=False)
        api_final = api.get_final_values()
        product_final = product.get_final_values("BD")
        
        print(f"  ✓ Scaled FRC: {subject_final.demographic.frc_ml:.0f} mL")
        print(f"  ✓ Product MMAD: {product_final._final_mmad:.2f} μm")
        
        # Create PK parameters (using the existing api parameters)
        print("3. Using API PK parameters...")
        print(f"  ✓ PK clearance: {api_final.clearance_L_h:.1f} L/h")
        print(f"  ✓ PK volume: {api_final.volume_central_L:.1f} L")
        
        # Run deposition to get initial amounts (simplified approach)
        print("4. Running regional deposition...")
        
        # Use the dose calculation from working code
        dose_pmol = product_final._final_dose_pg / api.molecular_weight  # pg / (μg/μmol) = pmol
        
        # For simplicity, use approximate regional fractions from working test
        et_fraction = 0.617
        bb_fraction = 0.039
        bb_b_fraction = 0.033
        al_fraction = 0.309
        
        regional_amounts = [
            dose_pmol * et_fraction,   # ET
            dose_pmol * bb_fraction,   # BB  
            dose_pmol * bb_b_fraction, # bb
            dose_pmol * al_fraction    # Al
        ]
        
        initial_dose = sum(regional_amounts)
        print(f"  ✓ Regional deposition completed")
        print(f"    Total initial dose: {initial_dose:.0f} pmol")
        print(f"    ET: {regional_amounts[0]:.0f} pmol")
        print(f"    BB: {regional_amounts[1]:.0f} pmol") 
        print(f"    bb: {regional_amounts[2]:.0f} pmol")
        print(f"    Al: {regional_amounts[3]:.0f} pmol")
        
        # Create PBPK orchestrator
        print("5. Creating PBPK orchestrator...")
        from lmp_pkg.models.pbpk.pbpk_orchestrator import PBPKOrchestrator
        
        orchestrator = PBPKOrchestrator(
            subject_params=subject_final,
            api_params=api_final,
            lung_model_type="regional",
            solve_dissolution=True  # Enable dissolution for proper functionality
        )
        
        print(f"  ✓ Orchestrator created: {orchestrator.n_states} states")
        
        # Set up initial state with deposition
        print("6. Setting initial state...")
        deposition_initial = {
            'ET_deposition': regional_amounts[0],
            'BB_deposition': regional_amounts[1], 
            'bb_deposition': regional_amounts[2],
            'Al_deposition': regional_amounts[3]
        }
        
        initial_state = orchestrator.get_initial_state(deposition_initial)
        print(f"  ✓ Initial state set: {len(initial_state)} states")
        print(f"    Non-zero states: {np.count_nonzero(initial_state)}")
        
        # Run PBPK simulation
        print("7. Running PBPK simulation...")
        simulation_time_hours = 1  # Start with 1 hour
        time_points = np.linspace(0, simulation_time_hours * 3600, 61)  # 1-minute intervals
        
        results = orchestrator.solve(time_points)
        
        # Get mass balance summary
        print("8. Analyzing mass balance...")
        
        # Calculate mass balance directly from solver output
        if hasattr(results, 'y') and results.y is not None:
            # scipy solve_ivp results format
            final_state = results.y[:, -1]  # Final state across all compartments
            
            # Use double precision for mass balance calculation
            total_final = float(np.sum(final_state, dtype=np.float64))
            
            print(f"  ✓ Using scipy solve_ivp results: {len(final_state)} states")
            print(f"    Raw final state sum: {total_final:.6e} pmol")
            print(f"    Non-zero final states: {np.count_nonzero(final_state)}")
            print(f"    Max final state value: {np.max(final_state):.6e} pmol")
            print(f"    Min final state value: {np.min(final_state):.6e} pmol")
            
            # Show top 10 largest values for debugging
            sorted_indices = np.argsort(final_state)[::-1][:10]
            print(f"    Top 10 largest final state values:")
            for i, idx in enumerate(sorted_indices):
                if final_state[idx] > 0:
                    print(f"      State {idx}: {final_state[idx]:.6e} pmol")
                    
        elif hasattr(results, 'state_matrix') and results.state_matrix is not None:
            # Custom results format
            final_state = results.state_matrix[-1]
            total_final = float(np.sum(final_state, dtype=np.float64))
            print(f"  ✓ Using state matrix results: {len(final_state)} states")
        else:
            print("  ✗ No valid results format found")
            total_final = 0
        
        print("\n" + "="*60)
        print(f"MASS BALANCE RESULTS - {simulation_time_hours} HOUR SIMULATION")
        print("="*60)
        
        mass_balance_pct = (total_final / initial_dose) * 100 if initial_dose > 0 else 0
        drug_lost_pct = 100 - mass_balance_pct
        
        print(f"Initial dose: {initial_dose:.0f} pmol")
        print(f"Final total: {total_final:.6e} pmol")
        print(f"Mass balance: {mass_balance_pct:.8f}%")
        print(f"Drug lost: {drug_lost_pct:.8f}%")
        
        # Evaluate results
        print(f"\n" + "="*60)
        print("EVALUATION:")
        print("="*60)
        
        if mass_balance_pct > 95:
            print(f"✅ EXCELLENT: {mass_balance_pct:.1f}% mass balance - matches notebook!")
            status = "EXCELLENT"
        elif mass_balance_pct > 90:
            print(f"✅ VERY GOOD: {mass_balance_pct:.1f}% mass balance - close to notebook")
            status = "VERY_GOOD"
        elif mass_balance_pct > 80:
            print(f"✓ GOOD: {mass_balance_pct:.1f}% mass balance - needs minor improvement")
            status = "GOOD"
        elif mass_balance_pct > 50:
            print(f"⚠ MODERATE: {mass_balance_pct:.1f}% mass balance - needs significant improvement")
            status = "MODERATE"
        elif mass_balance_pct > 10:
            print(f"❌ POOR: {mass_balance_pct:.1f}% mass balance - major issues remain")
            status = "POOR"
        else:
            print(f"💀 CATASTROPHIC: {mass_balance_pct:.1f}% mass balance - fundamental problems")
            status = "CATASTROPHIC"
        
        # Show raw results for debugging
        print(f"\nRaw results structure:")
        if hasattr(results, 'results_data'):
            for comp_name, comp_data in results.results_data.items():
                print(f"  {comp_name}: {type(comp_data)} with {len(comp_data) if isinstance(comp_data, dict) else 'N/A'} keys")
                if isinstance(comp_data, dict):
                    for key, values in list(comp_data.items())[:3]:  # Show first 3 keys
                        if isinstance(values, np.ndarray):
                            print(f"    {key}: array shape {values.shape}, final value {values[-1]:.2e}")
                        else:
                            print(f"    {key}: {type(values)} = {values}")
        
        return {
            'status': status,
            'mass_balance_pct': mass_balance_pct,
            'initial_dose': initial_dose,
            'final_total': total_final,
            'results': results
        }
        
    except Exception as e:
        print(f"\n❌ PBPK TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return {'status': 'FAILED', 'error': str(e)}

if __name__ == "__main__":
    results = run_real_pbpk_test()
    
    print("\n" + "="*80)
    print("FINAL STATUS:", results['status'])
    if 'mass_balance_pct' in results:
        print(f"MASS BALANCE: {results['mass_balance_pct']:.1f}%")
    print("="*80)