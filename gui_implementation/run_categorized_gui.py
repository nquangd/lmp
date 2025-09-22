#!/usr/bin/env python3
"""Launch the LMP GUI with categorized API parameters."""

import sys
from pathlib import Path

def main():
    """Launch the categorized LMP GUI application."""

    # Add the lmp_pkg source to Python path
    script_dir = Path(__file__).parent
    lmp_pkg_src = script_dir.parent / "lmp_pkg" / "src"

    if lmp_pkg_src.exists():
        sys.path.insert(0, str(lmp_pkg_src))
        print(f"Added to Python path: {lmp_pkg_src}")
    else:
        print(f"Warning: Could not find lmp_pkg source at {lmp_pkg_src}")

    # Import and run the categorized GUI
    try:
        from main_window_with_categories import main as gui_main
        print("Starting LMP GUI with Categorized API Parameters...")
        print("\nFeatures:")
        print("• Categorized API parameters with checkboxes")
        print("• Groups: Basic Properties, Pharmacokinetics, Transport, etc.")
        print("• Regional parameter tables (ET, BB, bb, Al)")
        print("• Compartment-specific parameters (Epithelium, Tissue)")
        print("• Matches reference interface layout")
        print()
        return gui_main()
    except ImportError as e:
        print(f"Error importing GUI modules: {e}")
        print("Make sure PySide6 and toml are installed:")
        print("  pip install PySide6 toml")
        return 1
    except Exception as e:
        print(f"Error starting GUI: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())