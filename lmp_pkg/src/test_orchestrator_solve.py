#!/usr/bin/env python3
"""
Minimal test to check if PBPKOrchestrator.solve() method works.
"""

import sys
import numpy as np
from pathlib import Path

# Add current directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

from lmp_pkg.config.hydration import hydrate_entities
from lmp_pkg.models.pbpk.pbpk_orchestrator import PBPKOrchestrator

def test_orchestrator_solve():
    """Test the solve method directly."""
    
    print("="*80)
    print("MINIMAL PBPK ORCHESTRATOR SOLVE TEST")
    print("="*80)
    
    try:
        # Load entities
        print("1. Loading entities...")
        subject, api, product, pk = hydrate_entities(
            subject_name="healthy_reference",
            api_name="BD",
            product_name="reference_product",
            pk_name="default"
        )
        print("  ✓ Entities loaded")
        
        # Create orchestrator
        print("2. Creating PBPK orchestrator...")
        orchestrator = PBPKOrchestrator(
            subject_params=subject,
            api_params=api,
            pk_params=pk,
            model_type="regional",
            solve_dissolution=False
        )
        print("  ✓ Orchestrator created")
        print(f"    Total states: {orchestrator.n_states}")
        
        # Simple test with short time
        print("3. Running 10-minute PBPK simulation...")
        time_points = np.linspace(0, 600, 11)  # 0 to 10 minutes, 1-minute intervals
        
        results = orchestrator.solve(time_points)
        print("  ✓ Solve method completed")
        
        # Get summary
        summary = results.get_summary()
        print("  ✓ Summary generated")
        
        print(f"\nResults:")
        print(f"  Total amount: {summary.total_pmol:.1f} pmol")
        print(f"  Lung: {summary.lung_total_pmol:.1f} pmol")
        print(f"  GI: {summary.gi_total_pmol:.1f} pmol") 
        print(f"  PK: {summary.pk_total_pmol:.1f} pmol")
        
        print(f"\n✓ SUCCESS: PBPKOrchestrator.solve() method works!")
        return True
        
    except Exception as e:
        print(f"✗ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_orchestrator_solve()
    print("\n" + "="*80)
    if success:
        print("CONCLUSION: PBPKOrchestrator solve method is working")
    else:
        print("CONCLUSION: PBPKOrchestrator solve method has issues")
    print("="*80)