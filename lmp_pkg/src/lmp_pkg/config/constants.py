"""Global constants for LMP modeling."""

from __future__ import annotations

# Lung modeling constants (from original Parameters_Settings.py)
N_GI_COMPS_LUNG_GENS_MAX = 25
N_BINS_MAX = 20
MIN_BIN_SIZE_UM = 0.7
MAX_BIN_SIZE_UM = 15.0
N_SLICES_MAX = 200

# Flow discretisation defaults (can be overridden for tabulated profiles)
DEFAULT_N_Q_MAX = 60
N_Q_MAX = DEFAULT_N_Q_MAX
N_STEPS = N_Q_MAX - 1


def set_flow_profile_steps(n_points: int) -> None:
    """Adjust flow discretisation counts for tabulated profiles.

    Args:
        n_points: Number of points in the tabulated profile (time samples).
    """
    global N_Q_MAX, N_STEPS

    # Legacy engine expects one extra point when generating N_STEPS linspace
    target = max(int(n_points) + 1, 2)
    N_Q_MAX = target
    N_STEPS = N_Q_MAX - 1


def reset_flow_profile_steps() -> None:
    """Reset flow discretisation to default analytic profile settings."""
    global N_Q_MAX, N_STEPS

    N_Q_MAX = DEFAULT_N_Q_MAX
    N_STEPS = N_Q_MAX - 1

# Flow and volume constants
CONST_FLOW_M3_S = 500e-6
V_SLICE_CONST_M3 = 20e-6

# Reference lung volume for scaling (3000 mL)
REFERENCE_LUNG_VOLUME_M3 = 2999.60e-6

# Physical constants
AIR_DENSITY_KG_M3 = 1.225      # Air density at 15°C
AIR_VISCOSITY_PA_S = 1.81e-5   # Air dynamic viscosity at 15°C
BODY_TEMPERATURE_K = 310.15     # Body temperature [K]
GRAVITY_M_S2 = 9.81            # Gravitational acceleration
BOLTZMANN_CONSTANT = 1.38e-23   # Boltzmann constant [J/K]

# Default particle properties
DEFAULT_PARTICLE_DENSITY_KG_M3 = 1200.0  # Typical inhaled drug density
DEFAULT_CUNNINGHAM_CORRECTION = 1.0      # For particles > 1 μm

# Legacy constant aliases for backward compatibility with Numba functions
N_GIcomps_LungGens_max = N_GI_COMPS_LUNG_GENS_MAX
N_bins_max = N_BINS_MAX
min_bin_size = MIN_BIN_SIZE_UM
max_bin_size = MAX_BIN_SIZE_UM
N_slices_max = N_SLICES_MAX
N_steps = N_STEPS
const_Flow = CONST_FLOW_M3_S
V_slice_const = V_SLICE_CONST_M3
