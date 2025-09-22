#!/usr/bin/env python3
"""Simple launcher script to run the LMP GUI.

This script can be used to launch the GUI application with proper
Python path setup to find the lmp_pkg modules.
"""

import sys
import os
from pathlib import Path

def main():
    """Launch the LMP GUI application."""

    # Add the lmp_pkg source to Python path
    script_dir = Path(__file__).parent
    lmp_pkg_src = script_dir.parent / "lmp_pkg" / "src"

    if lmp_pkg_src.exists():
        sys.path.insert(0, str(lmp_pkg_src))
        print(f"Added to Python path: {lmp_pkg_src}")
    else:
        print(f"Warning: Could not find lmp_pkg source at {lmp_pkg_src}")

    # Import and run the GUI
    try:
        from main_window import main as gui_main
        print("Starting LMP GUI...")
        return gui_main()
    except ImportError as e:
        print(f"Error importing GUI modules: {e}")
        print("Make sure PySide6 is installed: pip install PySide6")
        return 1
    except Exception as e:
        print(f"Error starting GUI: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())