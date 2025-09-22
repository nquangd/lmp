"""Utility functions for physiological modeling.

Note: Subject class has been moved to domain/subject.py as a unified class.
This module now contains only utility functions for physiological calculations.
"""

from __future__ import annotations
import math
from typing import Dict, List, Optional, Literal, Any, TYPE_CHECKING
import numpy as np
from scipy import stats
from dataclasses import dataclass

if TYPE_CHECKING:
    from .subject import Subject

from .entities import API, InhalationManeuver
from .variability import VariabilityEngine, VariabilityParameter
from ..catalog.builtin_loader import BuiltinDataLoader


def calculate_lung_volumes(subject: Subject) -> Dict[str, float]:
    """Calculate lung volumes and capacities from subject parameters.
    
    Uses established physiological scaling relationships based on age, 
    height, weight, and sex.
    
    Args:
        subject: Subject entity with physiological parameters
        
    Returns:
        Dictionary of lung volumes in mL:
        - frc: Functional residual capacity
        - tlc: Total lung capacity  
        - rv: Residual volume
        - tv: Tidal volume
        - ic: Inspiratory capacity
        - erc: Expiratory reserve capacity
    """
    height_cm = subject.height_cm
    age_years = subject.age_years
    sex = subject.sex.upper()
    
    # Use measured FRC if available, otherwise calculate
    if subject.frc_ml is not None:
        frc = subject.frc_ml
    else:
        # FRC prediction equations (Crapo et al., Am Rev Respir Dis 1982)
        if sex == "M":
            frc = 0.0472 * height_cm + 0.009 * age_years - 5.92
        else:  # Female
            frc = 0.0360 * height_cm + 0.0031 * age_years - 3.82
        frc *= 1000  # Convert to mL
    
    # TLC prediction (Crapo et al.)
    if sex == "M":
        tlc = 7.99 * height_cm - 7.08 * age_years - 285
    else:  # Female  
        tlc = 6.60 * height_cm - 5.79 * age_years - 309
    
    # Derived volumes
    rv = frc * 0.4  # Typical RV/FRC ratio
    tv = subject.tidal_volume_ml or 500.0  # Use measured or default
    ic = tlc - frc  # Inspiratory capacity
    erc = frc - rv  # Expiratory reserve capacity
    
    return {
        "frc": frc,
        "tlc": tlc, 
        "rv": rv,
        "tv": tv,
        "ic": ic,
        "erc": erc
    }


def calculate_airway_dimensions() -> Dict[str, np.ndarray]:
    """Calculate airway dimensions for lung regions.
    
    Based on Weibel symmetric model with regional modifications
    for tracheobronchial (TB) and alveolar (AL) regions.
    
    Returns:
        Dictionary containing:
        - diameters: Airway diameters by generation (cm)
        - lengths: Airway lengths by generation (cm) 
        - surface_areas: Surface areas by generation (cm²)
        - volumes: Regional volumes (mL)
    """
    # Weibel model parameters (generations 0-23)
    generations = np.arange(24)
    
    # Diameters (cm) - exponential decrease
    d_trachea = 1.8  # Tracheal diameter
    diameters = d_trachea * (0.79 ** generations)
    
    # Lengths (cm) - exponential decrease  
    l_trachea = 12.0  # Tracheal length
    lengths = l_trachea * (0.79 ** generations)
    
    # Number of airways per generation
    n_airways = 2.0 ** generations
    
    # Surface areas (cm²)
    surface_areas = np.pi * diameters * lengths * n_airways
    
    # Regional volumes (simplified)
    tb_volume = 150.0  # Tracheobronchial volume (mL)
    al_volume = 3000.0  # Alveolar volume (mL) 
    
    return {
        "diameters": diameters,
        "lengths": lengths,
        "surface_areas": surface_areas,
        "volumes": {"TB": tb_volume, "AL": al_volume}
    }


def calculate_deposition_fractions(
    particle_diameter_um: float,
    flow_rate_l_min: float,
    lung_volumes: Dict[str, float]
) -> Dict[str, float]:
    """Calculate regional deposition fractions using empirical correlations.
    
    Uses the ICRP 66 model for particle deposition in different lung regions.
    
    Args:
        particle_diameter_um: Particle diameter in micrometers
        flow_rate_l_min: Inspiratory flow rate in L/min
        lung_volumes: Lung volume dictionary from calculate_lung_volumes
        
    Returns:
        Dictionary of deposition fractions:
        - ET: Extrathoracic (nose/mouth, throat)
        - TB: Tracheobronchial 
        - AL: Alveolar
        - total: Total deposition
    """
    dp = particle_diameter_um
    Q = flow_rate_l_min / 60.0  # Convert to L/s
    
    # Dimensionless parameters
    frc_l = lung_volumes["frc"] / 1000.0
    tv_l = lung_volumes["tv"] / 1000.0
    
    # Impaction parameter
    d_ae = dp * math.sqrt(1.0)  # Assume unit density
    Stk = d_ae**2 * Q / (18.0 * 1.81e-5 * frc_l)  # Stokes number
    
    # Diffusion parameter  
    D = 2.4e-7 * (1 + 0.197 * dp**(-0.5))  # Diffusion coefficient
    diff_param = D * tv_l / (Q * frc_l**2)
    
    # Regional deposition fractions (ICRP 66 correlations)
    # Extrathoracic
    if dp < 1.0:
        f_ET = 0.5 * (1 - math.exp(-0.5 * Stk))
    else:
        f_ET = 1.0 - 0.5 * (1 + math.exp(-0.5 * Stk))
    
    # Tracheobronchial  
    f_TB = 0.00352 / dp + 0.234 * math.exp(-0.5 * (math.log(dp) + 2.5)**2)
    f_TB = min(f_TB, 0.99)
    
    # Alveolar (remaining after ET and TB)
    f_total = f_ET + f_TB + diff_param
    f_total = min(f_total, 0.99)
    f_AL = f_total - f_ET - f_TB
    f_AL = max(f_AL, 0.0)  # Ensure non-negative
    
    return {
        "ET": f_ET,
        "TB": f_TB, 
        "AL": f_AL,
        "total": f_total
    }


def calculate_clearance_rates(api_properties: Dict[str, float]) -> Dict[str, float]:
    """Calculate physiological clearance rates for drug absorption.
    
    Args:
        api_properties: API properties including permeability, clearance
        
    Returns:
        Dictionary of clearance rates (1/h):
        - mucociliary: Mucociliary clearance from TB region
        - absorption_tb: Absorption from TB region to systemic
        - absorption_al: Absorption from AL region to systemic  
        - dissolution: Drug dissolution rate
    """
    # Mucociliary clearance (typical values)
    k_muco = 2.0  # 1/h, relatively fast clearance
    
    # Absorption rates depend on permeability
    permeability = api_properties.get("permeability_cm_s", 1e-6)
    
    # Scale absorption by permeability and surface area
    k_abs_tb = permeability * 3600 * 0.1  # Convert cm/s to 1/h, scaled
    k_abs_al = permeability * 3600 * 1.0  # Higher surface area in alveoli
    
    # Dissolution rate (depends on solubility)
    solubility = api_properties.get("solubility_mg_ml", 1.0)
    k_dissolution = min(10.0, solubility)  # Fast for soluble drugs
    
    return {
        "mucociliary": k_muco,
        "absorption_tb": k_abs_tb,
        "absorption_al": k_abs_al,
        "dissolution": k_dissolution
    }


def calculate_flow_profile(maneuver: InhalationManeuver, time_points: np.ndarray) -> np.ndarray:
    """Calculate inspiratory flow rate profile over time.
    
    Args:
        maneuver: Inhalation profile parameters
        time_points: Time points for flow calculation (seconds)
        
    Returns:
        Flow rates in L/min at each time point
    """
    t_inhal = maneuver.inhalation_time_s
    Q_peak = maneuver.peak_inspiratory_flow_l_min
    t_accel = maneuver.flow_acceleration_s or (t_inhal * 0.3)
    t_decel = maneuver.flow_deceleration_s or (t_inhal * 0.7)
    
    flows = np.zeros_like(time_points)
    
    for i, t in enumerate(time_points):
        if t < 0 or t > t_inhal:
            flows[i] = 0.0
        elif t <= t_accel:
            # Acceleration phase - quadratic ramp up
            flows[i] = Q_peak * (t / t_accel)**2
        elif t <= (t_inhal - t_decel):
            # Constant phase
            flows[i] = Q_peak
        else:
            # Deceleration phase - quadratic ramp down
            t_rel = t - (t_inhal - t_decel)
            flows[i] = Q_peak * (1 - (t_rel / t_decel)**2)
    
    return flows


def calculate_inhalation_maneuver_flow_profile(
    pifr_lpm: float,
    rise_time_s: float,
    inhaled_volume_l: float
) -> np.ndarray:
    """Calculate inhalation flow profile using the original InhalationManeuver logic.
    
    Based on original lmp_apps/population/attributes.py InhalationManeuver.inhale_profile()
    Uses constants from config.constants for N_STEPS.
    
    Args:
        pifr_lpm: Peak inspiratory flow rate (L/min)
        rise_time_s: Rise time to peak flow (s)
        inhaled_volume_l: Inhaled volume (L)
        
    Returns:
        Flow profile as [time_points, flow_rates] array with shape (N_STEPS, 2)
        Flow rates in L/min
    """
    from ..config.constants import N_STEPS
    
    # Convert to L/s for calculations
    pifr_ls = pifr_lpm / 60.0
    
    # Calculate hold time
    if pifr_ls <= 0 or rise_time_s <= 0:
        hold_time = inhaled_volume_l / 1e-6 if pifr_ls <= 0 else inhaled_volume_l / pifr_ls
    else:
        hold_time = (inhaled_volume_l - 2 * (0.5 * pifr_ls * rise_time_s)) / pifr_ls
    
    if hold_time < 0:
        hold_time = 0
        
    # Total inhalation duration
    inhaled_duration = hold_time + 2 * rise_time_s
    if inhaled_duration <= 0:
        inhaled_duration = 1e-6  # Avoid division by zero
        
    # Calculate slope for ramp phases
    slope = pifr_ls / rise_time_s if rise_time_s > 0 else 0
    
    # Generate time points using N_STEPS from constants
    time_points = np.linspace(0, inhaled_duration, N_STEPS)
    flowrate = np.zeros_like(time_points)
    
    # Ramp up phase
    mask1 = time_points < rise_time_s
    flowrate[mask1] = time_points[mask1] * slope
    
    # Hold phase  
    mask2 = (time_points >= rise_time_s) & (time_points < (rise_time_s + hold_time))
    flowrate[mask2] = pifr_ls
    
    # Ramp down phase
    mask3 = time_points >= (rise_time_s + hold_time)
    flowrate[mask3] = pifr_ls - (time_points[mask3] - (rise_time_s + hold_time)) * slope
    
    # Ensure no negative flow
    flowrate = np.maximum(0, flowrate)
    
    # Format as [time, flow_rate] array and convert back to L/min
    flow_profile = np.zeros((len(flowrate), 2))
    flow_profile[:, 0] = time_points
    flow_profile[:, 1] = flowrate * 60.0  # Back to L/min
    
    return flow_profile


def validate_physiological_consistency(
    subject: Subject,
    lung_volumes: Dict[str, float]
) -> List[str]:
    """Validate physiological parameter consistency.
    
    Args:
        subject: Subject parameters
        lung_volumes: Calculated lung volumes
        
    Returns:
        List of validation warnings/errors
    """
    warnings = []
    
    # BMI checks
    bmi = subject.bmi_kg_m2
    if bmi < 16:
        warnings.append(f"BMI {bmi:.1f} is severely underweight")
    elif bmi > 40:
        warnings.append(f"BMI {bmi:.1f} is severely obese")
    
    # Lung volume consistency
    frc = lung_volumes["frc"]
    tlc = lung_volumes["tlc"]
    rv = lung_volumes["rv"]
    
    if frc >= tlc:
        warnings.append("FRC should be less than TLC")
    
    if rv >= frc:
        warnings.append("RV should be less than FRC")
    
    # Age-related checks
    if subject.age_years < 5:
        warnings.append("Pediatric subjects <5 years may need specialized models")
    elif subject.age_years > 90:
        warnings.append("Elderly subjects >90 years may need specialized models")
    
    # Respiratory rate consistency
    if subject.respiratory_rate_bpm:
        rr = subject.respiratory_rate_bpm
        if subject.age_years < 18 and rr < 15:
            warnings.append("Low respiratory rate for pediatric subject")
        elif subject.age_years >= 18 and rr > 25:
            warnings.append("High respiratory rate for adult subject")
    
    return warnings


def predict_lung_volumes_from_demographics(
    age_years: float,
    height_cm: float, 
    sex: str,
    ethnicity: Optional[str] = None
) -> Dict[str, float]:
    """Predict lung volumes from demographic data using population equations.
    
    This function provides population-based predictions that can be modified
    by the variability system for individual subjects.
    
    Args:
        age_years: Age in years
        height_cm: Height in centimeters
        sex: Biological sex ('M' or 'F')
        ethnicity: Optional ethnicity for population-specific equations (reserved for future use)
        
    Returns:
        Dictionary of predicted lung volumes [mL]
    """
    # Note: ethnicity parameter reserved for future population-specific equations
    sex = sex.upper()
    
    # GLI 2012 equations (Global Lung Function Initiative)
    # Simplified versions - full implementation would use spline coefficients
    
    if sex == 'M':
        # Male equations
        frc_predicted = (-0.1933 + 0.00064 * age_years + 0.000269 * height_cm**1.5) * 1000
        tlc_predicted = (-1.2082 + 0.0098 * age_years + 0.000346 * height_cm**1.5) * 1000
        rv_predicted = (-0.1933 + 0.00077 * age_years + 0.000201 * height_cm**1.5) * 1000
    else:
        # Female equations  
        frc_predicted = (-0.2178 + 0.00053 * age_years + 0.000241 * height_cm**1.5) * 1000
        tlc_predicted = (-1.3018 + 0.0081 * age_years + 0.000311 * height_cm**1.5) * 1000
        rv_predicted = (-0.2178 + 0.00064 * age_years + 0.000181 * height_cm**1.5) * 1000
    
    # Derived capacities
    ic_predicted = tlc_predicted - frc_predicted  # Inspiratory capacity
    erc_predicted = frc_predicted - rv_predicted  # Expiratory reserve capacity
    vc_predicted = ic_predicted + erc_predicted   # Vital capacity
    
    # Tidal volume (less variable, simpler prediction)
    tv_predicted = 7.0 * (height_cm / 100)**2 * 1000  # Rough approximation
    
    return {
        'frc_ml': max(1000, frc_predicted),
        'tlc_ml': max(3000, tlc_predicted), 
        'rv_ml': max(800, rv_predicted),
        'ic_ml': max(2000, ic_predicted),
        'erc_ml': max(800, erc_predicted),
        'vc_ml': max(3000, vc_predicted),
        'tv_ml': max(300, min(800, tv_predicted))
    }


def calculate_metabolic_scaling(
    weight_kg: float,
    height_cm: float,
    age_years: float,
    sex: str
) -> Dict[str, float]:
    """Calculate metabolic scaling factors for physiological processes.
    
    Used by models to scale clearance rates, metabolic rates, etc.
    Can be modified by variability system.
    
    Args:
        weight_kg: Body weight [kg]
        height_cm: Height [cm]
        age_years: Age [years]
        sex: Biological sex ('M' or 'F')
        
    Returns:
        Dictionary of scaling factors (dimensionless, relative to reference adult)
    """
    # Reference adult (70kg, 175cm, 35 years, male)
    ref_weight = 70.0
    ref_height = 175.0
    # ref_age = 35.0  # Reserved for future age-specific scaling
    
    # Allometric scaling for clearance (typically weight^0.75)
    clearance_scale = (weight_kg / ref_weight) ** 0.75
    
    # Surface area scaling for absorption
    bsa = 0.007184 * (weight_kg ** 0.425) * (height_cm ** 0.725)
    ref_bsa = 0.007184 * (ref_weight ** 0.425) * (ref_height ** 0.725)
    surface_area_scale = bsa / ref_bsa
    
    # Age-related scaling (simplified)
    if age_years < 18:
        # Pediatric adjustments
        age_scale = 0.7 + 0.3 * (age_years / 18)
    elif age_years > 65:
        # Elderly adjustments  
        age_scale = 1.0 - 0.01 * (age_years - 65)
    else:
        age_scale = 1.0
    
    # Sex-related differences (simplified)
    sex_scale = 0.85 if sex.upper() == 'F' else 1.0
    
    return {
        'clearance_scale': clearance_scale * age_scale * sex_scale,
        'surface_area_scale': surface_area_scale,
        'volume_scale': (weight_kg / ref_weight),
        'age_scale': age_scale,
        'sex_scale': sex_scale
    }


def estimate_respiratory_parameters(
    subject: Subject,
    activity_level: str = 'resting'
) -> Dict[str, float]:
    """Estimate respiratory parameters for different activity levels.
    
    Args:
        subject: Subject entity with demographic data
        activity_level: Activity level ('resting', 'light', 'moderate', 'heavy')
        
    Returns:
        Dictionary of respiratory parameters
    """
    base_rr = 12 + (40 - subject.age_years) * 0.1  # Age effect on RR
    base_rr = max(8, min(25, base_rr))  # Physiological bounds
    
    # Activity level multipliers
    activity_factors = {
        'resting': {'rr': 1.0, 'tv': 1.0, 'flow': 1.0},
        'light': {'rr': 1.3, 'tv': 1.2, 'flow': 1.5},  
        'moderate': {'rr': 1.8, 'tv': 1.5, 'flow': 2.2},
        'heavy': {'rr': 2.5, 'tv': 1.8, 'flow': 3.0}
    }
    
    factors = activity_factors.get(activity_level, activity_factors['resting'])
    
    # Estimate parameters
    respiratory_rate = base_rr * factors['rr']
    tidal_volume = (subject.tidal_volume_ml or 500) * factors['tv']
    minute_ventilation = respiratory_rate * tidal_volume  # mL/min
    peak_flow = minute_ventilation * factors['flow'] * 0.1  # L/min (rough conversion)
    
    return {
        'respiratory_rate_bpm': respiratory_rate,
        'tidal_volume_ml': tidal_volume,
        'minute_ventilation_ml_min': minute_ventilation,
        'estimated_peak_flow_l_min': peak_flow
    }


# ==============================================================================
# NOTE: Subject transformation classes have been moved to domain/subject.py
# This file now contains only utility physiology functions
# ==============================================================================


