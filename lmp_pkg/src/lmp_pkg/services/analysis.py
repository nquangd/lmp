"""PK and bioequivalence analysis services."""

from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple
from scipy.integrate import trapz


def calculate_pk_metrics(
    time_points: np.ndarray,
    concentrations: np.ndarray,
    dose: Optional[float] = None
) -> Dict[str, float]:
    """Calculate standard PK metrics from concentration-time data.
    
    Args:
        time_points: Time points [h]
        concentrations: Plasma concentrations [ng/mL or pmol/mL]
        dose: Administered dose (optional, for dose-normalized metrics)
        
    Returns:
        Dictionary with PK metrics
    """
    # Ensure arrays are numpy arrays
    t = np.asarray(time_points)
    c = np.asarray(concentrations)
    
    # Remove NaN/negative values
    valid_idx = ~np.isnan(c) & (c >= 0)
    t_valid = t[valid_idx]
    c_valid = c[valid_idx]
    
    if len(c_valid) == 0:
        return {
            'auc_0_inf': np.nan,
            'auc_0_t': np.nan,
            'cmax': np.nan,
            'tmax': np.nan,
            'c_last': np.nan,
            't_last': np.nan,
            'half_life': np.nan,
            'clearance': np.nan,
            'volume_dist': np.nan
        }
    
    # Cmax and Tmax
    cmax_idx = np.argmax(c_valid)
    cmax = c_valid[cmax_idx]
    tmax = t_valid[cmax_idx]
    
    # AUC from 0 to last time point (trapezoidal rule)
    if len(t_valid) > 1:
        auc_0_t = trapz(c_valid, t_valid)
    else:
        auc_0_t = 0.0
    
    # Last measurable concentration and time
    c_last = c_valid[-1]
    t_last = t_valid[-1]
    
    # Terminal elimination rate constant and half-life
    # Use last 3-5 points for terminal slope if available
    n_terminal = min(5, max(3, len(c_valid) // 3))
    if len(c_valid) >= 3 and c_last > 0:
        # Log-linear regression on terminal points
        terminal_idx = slice(-n_terminal, None)
        t_terminal = t_valid[terminal_idx]
        c_terminal = c_valid[terminal_idx]
        
        # Only use points where concentration is decreasing
        if len(c_terminal) >= 3 and np.all(c_terminal > 0):
            log_c_terminal = np.log(c_terminal)
            
            # Linear regression: log(C) = log(C0) - ke*t
            try:
                slope, intercept = np.polyfit(t_terminal, log_c_terminal, 1)
                ke = -slope  # Elimination rate constant [1/h]
                
                if ke > 0:
                    half_life = 0.693 / ke  # Half-life [h]
                    # Extrapolate AUC to infinity
                    auc_tail = c_last / ke
                    auc_0_inf = auc_0_t + auc_tail
                else:
                    half_life = np.nan
                    auc_0_inf = np.nan
            except:
                half_life = np.nan
                auc_0_inf = np.nan
        else:
            half_life = np.nan
            auc_0_inf = np.nan
    else:
        half_life = np.nan
        auc_0_inf = np.nan
    
    # Derived PK parameters
    clearance = np.nan
    volume_dist = np.nan
    
    if dose is not None and not np.isnan(auc_0_inf) and auc_0_inf > 0:
        clearance = dose / auc_0_inf  # Clearance [L/h or mL/h]
        
        if not np.isnan(half_life) and half_life > 0:
            ke = 0.693 / half_life
            volume_dist = clearance / ke  # Volume of distribution [L or mL]
    
    return {
        'auc_0_inf': auc_0_inf,
        'auc_0_t': auc_0_t,
        'cmax': cmax,
        'tmax': tmax,
        'c_last': c_last,
        't_last': t_last,
        'half_life': half_life,
        'clearance': clearance,
        'volume_dist': volume_dist
    }


def calculate_bioequivalence_metrics(
    test_data: Dict[str, np.ndarray],
    reference_data: Dict[str, np.ndarray],
    metrics: list[str] = ['auc_0_inf', 'cmax'],
    confidence_level: float = 0.90
) -> Dict[str, Dict[str, float]]:
    """Calculate bioequivalence metrics comparing test vs reference.
    
    Args:
        test_data: Test formulation PK metrics
        reference_data: Reference formulation PK metrics
        metrics: Metrics to compare
        confidence_level: Confidence level for intervals (0.90 for 90% CI)
        
    Returns:
        Dictionary with bioequivalence statistics for each metric
    """
    results = {}
    
    for metric in metrics:
        if metric not in test_data or metric not in reference_data:
            results[metric] = {
                'geometric_mean_ratio': np.nan,
                'ci_lower': np.nan,
                'ci_upper': np.nan,
                'within_be_limits': False
            }
            continue
        
        test_values = np.asarray(test_data[metric])
        ref_values = np.asarray(reference_data[metric])
        
        # Remove invalid values
        test_valid = test_values[~np.isnan(test_values) & (test_values > 0)]
        ref_valid = ref_values[~np.isnan(ref_values) & (ref_values > 0)]
        
        if len(test_valid) == 0 or len(ref_valid) == 0:
            results[metric] = {
                'geometric_mean_ratio': np.nan,
                'ci_lower': np.nan,
                'ci_upper': np.nan,
                'within_be_limits': False
            }
            continue
        
        # Log-transform data for geometric mean calculations
        log_test = np.log(test_valid)
        log_ref = np.log(ref_valid)
        
        # Calculate geometric mean ratio
        log_diff = log_test - log_ref
        log_gmr = np.mean(log_diff)
        gmr = np.exp(log_gmr)
        
        # Calculate confidence interval
        n = len(log_diff)
        if n > 1:
            se_diff = np.std(log_diff, ddof=1) / np.sqrt(n)
            
            # t-distribution critical value
            alpha = 1 - confidence_level
            from scipy.stats import t
            t_crit = t.ppf(1 - alpha/2, df=n-1)
            
            # CI on log scale
            log_ci_lower = log_gmr - t_crit * se_diff
            log_ci_upper = log_gmr + t_crit * se_diff
            
            # Back-transform to original scale
            ci_lower = np.exp(log_ci_lower)
            ci_upper = np.exp(log_ci_upper)
        else:
            ci_lower = np.nan
            ci_upper = np.nan
        
        # Check if within bioequivalence limits (80-125%)
        be_lower_limit = 0.80
        be_upper_limit = 1.25
        
        within_be_limits = (
            not np.isnan(ci_lower) and not np.isnan(ci_upper) and
            ci_lower >= be_lower_limit and ci_upper <= be_upper_limit
        )
        
        results[metric] = {
            'geometric_mean_ratio': gmr,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'within_be_limits': within_be_limits
        }
    
    return results


def summarize_pk_results(
    pk_results: Dict[str, Any],
    time_points: Optional[np.ndarray] = None
) -> pd.DataFrame:
    """Summarize PK results into a tidy DataFrame format.
    
    Args:
        pk_results: Results from PK simulation
        time_points: Optional time points for time-series data
        
    Returns:
        DataFrame with summarized PK metrics
    """
    summary_data = []
    
    # Add PK metrics if available
    if 'pk_metrics' in pk_results:
        for metric, value in pk_results['pk_metrics'].items():
            summary_data.append({
                'parameter': metric,
                'value': value,
                'type': 'pk_metric'
            })
    
    # Add concentration data summary if available
    if 'concentrations' in pk_results:
        conc_data = pk_results['concentrations']
        for key, values in conc_data.items():
            if isinstance(values, np.ndarray):
                summary_data.append({
                    'parameter': f'{key}_mean',
                    'value': np.nanmean(values),
                    'type': 'concentration'
                })
                summary_data.append({
                    'parameter': f'{key}_max',
                    'value': np.nanmax(values),
                    'type': 'concentration'
                })
    
    return pd.DataFrame(summary_data)


def analyze_dose_response(
    doses: np.ndarray,
    responses: np.ndarray,
    metric: str = 'auc_0_inf'
) -> Dict[str, Any]:
    """Analyze dose-response relationship.
    
    Args:
        doses: Administered doses
        responses: Response metric values
        metric: Name of the response metric
        
    Returns:
        Dictionary with dose-response analysis results
    """
    # Remove invalid data
    valid_idx = ~np.isnan(doses) & ~np.isnan(responses) & (doses > 0) & (responses > 0)
    doses_valid = doses[valid_idx]
    responses_valid = responses[valid_idx]
    
    if len(doses_valid) < 2:
        return {
            'metric': metric,
            'dose_proportional': False,
            'slope': np.nan,
            'intercept': np.nan,
            'r_squared': np.nan
        }
    
    # Log-log regression to test dose proportionality
    log_doses = np.log(doses_valid)
    log_responses = np.log(responses_valid)
    
    # Linear regression: log(response) = log(intercept) + slope * log(dose)
    slope, log_intercept = np.polyfit(log_doses, log_responses, 1)
    
    # Calculate R-squared
    log_responses_pred = log_intercept + slope * log_doses
    ss_res = np.sum((log_responses - log_responses_pred) ** 2)
    ss_tot = np.sum((log_responses - np.mean(log_responses)) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    
    # Test for dose proportionality (slope should be close to 1)
    # Using a simple criterion: slope between 0.8 and 1.2
    dose_proportional = 0.8 <= slope <= 1.2
    
    return {
        'metric': metric,
        'dose_proportional': dose_proportional,
        'slope': slope,
        'intercept': np.exp(log_intercept),
        'r_squared': r_squared
    }