"""Virtual Bioequivalence (VBE) analysis adapted for new PBBM architecture.

This module migrates and adapts the bioequivalence analysis capabilities
to work with the new modular architecture and data structures.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Callable, Tuple, Union
from scipy import stats
from scipy.stats import t
from multiprocessing import Pool
from dataclasses import dataclass
import warnings

from ..models.base import PBBMModelComponent
from ..data_structures import PKResultsData, RegionalAmountData, ComprehensivePBBMResults
from .vbe_helpers import MetricSpec, compute_metric


@dataclass
class VBEStudyConfig:
    """Configuration for Virtual Bioequivalence studies."""
    n_trials: int = 200
    trial_size: int = 204
    inner_bootstrap: int = 100
    alpha_pk: float = 0.1      # Alpha for PK metrics
    alpha_pd: float = 0.1      # Alpha for PD metrics  
    alpha_metrics: float = 0.05  # Alpha for general metrics
    seed: int = 1000


@dataclass
class SimulationResults:
    """Container for simulation results from new architecture models."""
    time_s: np.ndarray
    pk_results: PKResultsData
    regional_results: Optional[Dict[str, RegionalAmountData]] = None
    lung_deposition: Optional[Dict[str, float]] = None
    subject_id: str = "default"
    product_id: str = "default"
    api_composition: Optional[Dict[str, float]] = None


class NewArchitectureBioequivalenceAssessor:
    """Virtual Bioequivalence assessor adapted for the new architecture.
    
    This class adapts the original bioequivalence capabilities to work with
    the new modular architecture, including:
    - Integration with new data structures
    - Support for composed model systems  
    - Flexible simulation result handling
    - Regional and systemic bioequivalence assessment
    """
    
    def __init__(self, study_config: VBEStudyConfig, residual_var: float = 0.013):
        """Initialize bioequivalence assessor.
        
        Args:
            study_config: Study configuration parameters
            residual_var: Residual variance for parametric methods
        """
        self.config = study_config
        self.residual_var = residual_var
        self.non_parametric_pd_flag = False
        self.non_parametric_pk_flag = True  # Default to non-parametric
        
    @staticmethod
    def bootstrap_ci(data: np.ndarray, func: Callable, n_bootstrap: int = 100, 
                     alpha: float = 0.1) -> Tuple[float, float, float]:
        """Calculate bootstrap confidence interval for a statistic.
        
        Args:
            data: Input data
            func: Function to calculate statistic
            n_bootstrap: Number of bootstrap samples
            alpha: Alpha level for CI
            
        Returns:
            Tuple of (point_estimate, ci_lower, ci_upper)
        """
        n = len(data)
        bootstraps = []
        
        for _ in range(n_bootstrap):
            sample = np.random.choice(data, size=n, replace=True)
            bootstraps.append(func(sample))
        
        bootstraps = np.array(bootstraps)
        lower = np.percentile(bootstraps, 100 * alpha / 2)
        upper = np.percentile(bootstraps, 100 * (1 - alpha / 2))
        
        return func(data), lower, upper
    
    @staticmethod
    def geometric_mean(x: np.ndarray) -> float:
        """Calculate geometric mean of positive values."""
        x = np.array(x)
        x = x[x > 0]
        if len(x) == 0:
            return np.nan
        return np.exp(np.mean(np.log(x)))
    
    @staticmethod
    def ratio_and_ci(test: np.ndarray, ref: np.ndarray, func: Callable,
                     n_bootstrap: int = 100, alpha: float = 0.1, 
                     non_parametric: bool = True) -> Tuple[float, float, float]:
        """Calculate ratio and confidence interval for test vs reference.
        
        Args:
            test: Test group values
            ref: Reference group values
            func: Function for summary statistic (mean, geometric_mean)
            n_bootstrap: Number of bootstrap samples
            alpha: Alpha level for CI
            non_parametric: Use non-parametric bootstrap vs t-test
            
        Returns:
            Tuple of (ratio, ci_lower, ci_upper)
        """
        if len(test) == 0 or len(ref) == 0 or np.all(ref == 0):
            return np.nan, np.nan, np.nan
        
        if non_parametric:
            # Non-parametric bootstrap approach
            logdiff = np.log(test) - np.log(ref)
            gmr = np.exp(np.mean(logdiff))
            n = len(logdiff)
            
            boot_gmr = []
            for _ in range(n_bootstrap):
                idx_boot = np.random.choice(np.arange(n), size=n, replace=True)
                boot_ldiff = logdiff[idx_boot]
                boot_gmr.append(np.exp(np.mean(boot_ldiff)))
            
            boot_gmr = np.array(boot_gmr)
            delta = boot_gmr - gmr
            lo = np.nanpercentile(delta, 100 * alpha / 2)
            hi = np.nanpercentile(delta, 100 * (1 - alpha / 2))
            
            lower_adj = gmr - hi
            upper_adj = gmr - lo
            
            return gmr, lower_adj, upper_adj
        
        else:
            # Parametric t-test approach
            logdiff = np.log(test) - np.log(ref)
            n = len(logdiff)
            mean_ld = np.mean(logdiff)
            std_ld = np.std(logdiff, ddof=1)
            se = std_ld / np.sqrt(n)
            
            t_crit = t.ppf(1 - alpha / 2, n - 1)
            ci_lo = mean_ld - t_crit * se
            ci_hi = mean_ld + t_crit * se
            
            gmr = np.exp(mean_ld)
            lower = np.exp(ci_lo)
            upper = np.exp(ci_hi)
            
            return gmr, lower, upper
    
    def extract_pk_metrics_from_results(self, sim_results: SimulationResults) -> Dict[str, float]:
        """Extract PK metrics from simulation results.
        
        Args:
            sim_results: Simulation results from new architecture
            
        Returns:
            Dictionary with PK metrics (AUC, Cmax, etc.)
        """
        pk_data = sim_results.pk_results
        time_h = sim_results.time_s / 3600.0  # Convert to hours
        
        # Calculate AUC using trapezoidal rule
        if hasattr(pk_data, 'plasma_concentration') and len(pk_data.plasma_concentration) > 1:
            conc = pk_data.plasma_concentration
            auc = np.trapz(conc, time_h)
            cmax = np.max(conc)
            tmax = time_h[np.argmax(conc)]
        else:
            auc = 0.0
            cmax = 0.0
            tmax = 0.0
        
        # Calculate additional metrics if available
        metrics = {
            'AUC_pmol_h_per_mL': auc,
            'Cmax_pmol_per_mL': cmax,
            'Tmax_h': tmax,
            'final_concentration': conc[-1] if len(conc) > 0 else 0.0
        }
        
        # Add compartment amounts if available
        if hasattr(pk_data, 'central_amounts'):
            metrics['central_amount_final'] = pk_data.central_amounts[-1] if len(pk_data.central_amounts) > 0 else 0.0
        
        if hasattr(pk_data, 'total_systemic_amounts'):
            total_amounts = pk_data.total_systemic_amounts
            if len(total_amounts) > 0:
                metrics['total_systemic_final'] = total_amounts[-1]
                metrics['total_systemic_auc'] = np.trapz(total_amounts, time_h)
        
        return metrics
    
    def extract_regional_metrics_from_results(self, sim_results: SimulationResults) -> Dict[str, Dict[str, float]]:
        """Extract regional lung metrics from simulation results.
        
        Args:
            sim_results: Simulation results from new architecture
            
        Returns:
            Dictionary with regional metrics by region
        """
        regional_metrics = {}
        
        if sim_results.regional_results:
            time_h = sim_results.time_s / 3600.0
            
            for region_name, regional_data in sim_results.regional_results.items():
                # Calculate regional AUC and final amounts
                if hasattr(regional_data, 'total_amounts') and len(regional_data.total_amounts) > 0:
                    amounts = regional_data.total_amounts
                    regional_auc = np.trapz(amounts, time_h)
                    final_amount = amounts[-1]
                    max_amount = np.max(amounts)
                else:
                    regional_auc = 0.0
                    final_amount = 0.0
                    max_amount = 0.0
                
                regional_metrics[region_name] = {
                    'AUC_pmol_h': regional_auc,
                    'final_amount_pmol': final_amount,
                    'max_amount_pmol': max_amount,
                    'retention_fraction': final_amount / max_amount if max_amount > 0 else 0.0
                }
        
        return regional_metrics
    
    def create_be_dataset_from_simulations(self, 
                                         test_simulations: List[SimulationResults],
                                         ref_simulations: List[SimulationResults]) -> pd.DataFrame:
        """Create bioequivalence dataset from simulation results.
        
        Args:
            test_simulations: Test product simulation results
            ref_simulations: Reference product simulation results
            
        Returns:
            DataFrame formatted for bioequivalence analysis
        """
        be_data = []
        
        # Process test simulations
        for i, sim_result in enumerate(test_simulations):
            subject_id = sim_result.subject_id or f"Subject_{i+1}"
            
            # Extract PK metrics
            pk_metrics = self.extract_pk_metrics_from_results(sim_result)
            
            # Extract regional metrics
            regional_metrics = self.extract_regional_metrics_from_results(sim_result)
            
            # Create base record
            base_record = {
                'Subject': subject_id,
                'Product': 'Test',
                'Systemic_AUC': pk_metrics['AUC_pmol_h_per_mL'],
                'Systemic_Cmax': pk_metrics['Cmax_pmol_per_mL'],
                'Systemic_Tmax': pk_metrics['Tmax_h']
            }
            
            # Add API composition if available
            if sim_result.api_composition:
                for api, dose in sim_result.api_composition.items():
                    base_record['API'] = api  # Simplified - assumes single API
                    break
            else:
                base_record['API'] = 'Unknown'
            
            # Add regional metrics
            for region, metrics in regional_metrics.items():
                for metric, value in metrics.items():
                    base_record[f'{region}_{metric}'] = value
            
            be_data.append(base_record)
        
        # Process reference simulations
        for i, sim_result in enumerate(ref_simulations):
            subject_id = sim_result.subject_id or f"Subject_{i+1}"
            
            pk_metrics = self.extract_pk_metrics_from_results(sim_result)
            regional_metrics = self.extract_regional_metrics_from_results(sim_result)
            
            base_record = {
                'Subject': subject_id,
                'Product': 'Reference',
                'Systemic_AUC': pk_metrics['AUC_pmol_h_per_mL'],
                'Systemic_Cmax': pk_metrics['Cmax_pmol_per_mL'],
                'Systemic_Tmax': pk_metrics['Tmax_h']
            }
            
            if sim_result.api_composition:
                for api, dose in sim_result.api_composition.items():
                    base_record['API'] = api
                    break
            else:
                base_record['API'] = 'Unknown'
            
            for region, metrics in regional_metrics.items():
                for metric, value in metrics.items():
                    base_record[f'{region}_{metric}'] = value
            
            be_data.append(base_record)
        
        return pd.DataFrame(be_data)
    
    def run_systemic_bioequivalence_analysis(self, 
                                           test_simulations: List[SimulationResults],
                                           ref_simulations: List[SimulationResults],
                                           pool_cores: int = 4) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Run systemic bioequivalence analysis on simulation results.
        
        Args:
            test_simulations: Test product simulation results
            ref_simulations: Reference product simulation results
            pool_cores: Number of CPU cores for parallel processing
            
        Returns:
            Tuple of (summary_df, trial_df)
        """
        # Create BE dataset
        be_df = self.create_be_dataset_from_simulations(test_simulations, ref_simulations)
        
        # Prepare arguments for parallel processing
        args_list = [
            (be_df.copy(), self.config.trial_size, self.config.inner_bootstrap, 
             self.config.alpha_pk, self.non_parametric_pk_flag, self.config.seed + i)
            for i in range(self.config.n_trials)
        ]
        
        # Run parallel analysis 
        # TODO: Implement parallel processing for systemic bioequivalence
        results = []
        for args in args_list:
            # For now, run sequentially
            # results.append(self._systemic_bioequiv_single_trial(args))
            results.append([])  # Placeholder
        
        # Flatten and index trials
        trial_rows = []
        for trial_idx, rows in enumerate(results):
            for row in rows:
                r = row.copy()
                r['Trial'] = trial_idx
                trial_rows.append(r)
        
        trial_df = pd.DataFrame(trial_rows)
        
        # Summary aggregation using simple averaging for now
        summary = []
        for key, g in trial_df.groupby(['API', 'Parameter']) if not trial_df.empty else []:
            vals = g['Mean_Ratio'].values if 'Mean_Ratio' in g.columns else []
            geo_vals = g['GeoMean_Ratio'].values if 'GeoMean_Ratio' in g.columns else []
            
            # Simple averaging for now
            mean = np.mean(vals) if len(vals) > 0 else np.nan
            mean_low = np.percentile(vals, 5) if len(vals) > 0 else np.nan
            mean_up = np.percentile(vals, 95) if len(vals) > 0 else np.nan
            
            geo = np.mean(geo_vals) if len(geo_vals) > 0 else np.nan
            geo_low = np.percentile(geo_vals, 5) if len(geo_vals) > 0 else np.nan
            geo_up = np.percentile(geo_vals, 95) if len(geo_vals) > 0 else np.nan
            
            summary.append({
                'API': key[0], 'Parameter': key[1],
                'Mean_Ratio': mean, 'Mean_CI_Lower': mean_low, 'Mean_CI_Upper': mean_up,
                'GeoMean_Ratio': geo, 'GeoMean_CI_Lower': geo_low, 'GeoMean_CI_Upper': geo_up
            })
        
        summary_df = pd.DataFrame(summary)
        return summary_df, trial_df
    
    def run_regional_bioequivalence_analysis(self, 
                                           test_simulations: List[SimulationResults],
                                           ref_simulations: List[SimulationResults]) -> Dict[str, Dict[str, Any]]:
        """Run regional bioequivalence analysis.
        
        Args:
            test_simulations: Test product simulation results  
            ref_simulations: Reference product simulation results
            
        Returns:
            Dictionary with regional BE results
        """
        # Extract regional data
        regional_be_results = {}
        
        # Get all unique regions
        all_regions = set()
        for sim in test_simulations + ref_simulations:
            if sim.regional_results:
                all_regions.update(sim.regional_results.keys())
        
        # Analyze each region
        for region in all_regions:
            # Extract regional metrics for test and reference
            test_metrics = []
            ref_metrics = []
            
            for sim in test_simulations:
                if sim.regional_results and region in sim.regional_results:
                    regional_data = sim.regional_results[region]
                    if hasattr(regional_data, 'total_amounts') and len(regional_data.total_amounts) > 0:
                        final_amount = regional_data.total_amounts[-1]
                        test_metrics.append(final_amount)
            
            for sim in ref_simulations:
                if sim.regional_results and region in sim.regional_results:
                    regional_data = sim.regional_results[region]
                    if hasattr(regional_data, 'total_amounts') and len(regional_data.total_amounts) > 0:
                        final_amount = regional_data.total_amounts[-1]
                        ref_metrics.append(final_amount)
            
            # Calculate bioequivalence metrics
            if len(test_metrics) > 0 and len(ref_metrics) > 0:
                test_array = np.array(test_metrics)
                ref_array = np.array(ref_metrics)
                
                # Geometric mean ratio and CI
                gmr, gmr_low, gmr_up = self.ratio_and_ci(
                    test_array, ref_array, self.geometric_mean,
                    self.config.inner_bootstrap, self.config.alpha_pd,
                    self.non_parametric_pd_flag
                )
                
                # Arithmetic mean ratio and CI
                amr, amr_low, amr_up = self.ratio_and_ci(
                    test_array, ref_array, np.mean,
                    self.config.inner_bootstrap, self.config.alpha_pd,
                    self.non_parametric_pd_flag
                )
                
                regional_be_results[region] = {
                    'geometric_mean_ratio': gmr,
                    'geometric_ci': (gmr_low, gmr_up),
                    'arithmetic_mean_ratio': amr,
                    'arithmetic_ci': (amr_low, amr_up),
                    'n_test': len(test_metrics),
                    'n_ref': len(ref_metrics)
                }
            else:
                regional_be_results[region] = {
                    'geometric_mean_ratio': np.nan,
                    'geometric_ci': (np.nan, np.nan),
                    'arithmetic_mean_ratio': np.nan,
                    'arithmetic_ci': (np.nan, np.nan),
                    'n_test': len(test_metrics),
                    'n_ref': len(ref_metrics)
                }
        
        return regional_be_results
    
    def assess_bioequivalence(self, summary_df: pd.DataFrame, 
                             be_limits: Tuple[float, float] = (0.80, 1.25)) -> Dict[str, Any]:
        """Assess bioequivalence based on confidence interval inclusion in BE limits.
        
        Args:
            summary_df: Summary DataFrame from bioequivalence analysis
            be_limits: Bioequivalence acceptance limits (lower, upper)
            
        Returns:
            Dictionary with bioequivalence assessment results
        """
        be_results = {}
        
        for _, row in summary_df.iterrows():
            api = row['API']
            param = row['Parameter']
            
            # Check geometric mean ratio BE
            gmr_be = (row['GeoMean_CI_Lower'] >= be_limits[0] and 
                     row['GeoMean_CI_Upper'] <= be_limits[1])
            
            # Check arithmetic mean ratio BE
            mean_be = (row['Mean_CI_Lower'] >= be_limits[0] and 
                      row['Mean_CI_Upper'] <= be_limits[1])
            
            key = f"{api}_{param}"
            be_results[key] = {
                'geometric_mean_ratio': row['GeoMean_Ratio'],
                'geometric_ci': (row['GeoMean_CI_Lower'], row['GeoMean_CI_Upper']),
                'geometric_bioequivalent': gmr_be,
                'arithmetic_mean_ratio': row['Mean_Ratio'],
                'arithmetic_ci': (row['Mean_CI_Lower'], row['Mean_CI_Upper']),
                'arithmetic_bioequivalent': mean_be
            }
        
        # Overall BE assessment
        all_gmr_be = all(result['geometric_bioequivalent'] for result in be_results.values())
        all_mean_be = all(result['arithmetic_bioequivalent'] for result in be_results.values())
        
        return {
            'individual_results': be_results,
            'overall_geometric_bioequivalent': all_gmr_be,
            'overall_arithmetic_bioequivalent': all_mean_be,
            'bioequivalence_limits': be_limits
        }


def run_new_architecture_vbe_study(
    test_simulations: List[SimulationResults],
    ref_simulations: List[SimulationResults],
    study_config: Optional[VBEStudyConfig] = None,
    pool_cores: int = 4
) -> Dict[str, Any]:
    """Run a complete virtual bioequivalence study with new architecture.
    
    Args:
        test_simulations: Test product simulation results
        ref_simulations: Reference product simulation results
        study_config: Study configuration
        pool_cores: Number of CPU cores for parallel processing
        
    Returns:
        Complete VBE study results
    """
    if study_config is None:
        study_config = VBEStudyConfig()
    
    # Initialize assessor
    assessor = NewArchitectureBioequivalenceAssessor(study_config)
    
    # Run systemic bioequivalence analysis
    systemic_summary_df, systemic_trial_df = assessor.run_systemic_bioequivalence_analysis(
        test_simulations, ref_simulations, pool_cores
    )
    
    # Run regional bioequivalence analysis
    regional_results = assessor.run_regional_bioequivalence_analysis(
        test_simulations, ref_simulations
    )
    
    # Assess systemic bioequivalence
    systemic_be_assessment = assessor.assess_bioequivalence(systemic_summary_df)
    
    return {
        'study_config': study_config,
        'systemic_summary_results': systemic_summary_df,
        'systemic_trial_results': systemic_trial_df,
        'systemic_bioequivalence_assessment': systemic_be_assessment,
        'regional_results': regional_results,
        'n_test_simulations': len(test_simulations),
        'n_ref_simulations': len(ref_simulations),
        'n_trials_completed': len(systemic_trial_df['Trial'].unique()) if not systemic_trial_df.empty else 0
    }
