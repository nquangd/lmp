"""Visualization tools for new PBBM architecture."""

from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List, Tuple, Union
from dataclasses import dataclass
from pathlib import Path

try:
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    from matplotlib.figure import Figure
    from matplotlib.axes import Axes
    import seaborn as sns
    HAS_PLOTTING = True
except ImportError:
    plt = None
    Figure = None
    Axes = None
    HAS_PLOTTING = False
    

# Optional sequential results import (not always present in all builds)
try:
    from ..models.composition.sequential import SequentialPBBMResults  # type: ignore
    HAS_SEQUENTIAL = True
except Exception:
    SequentialPBBMResults = None  # type: ignore
    HAS_SEQUENTIAL = False

from .bioequivalence import SimulationResults


@dataclass
class PlotConfig:
    """Configuration for plot styling and output."""
    figsize: Tuple[float, float] = (12, 8)
    dpi: int = 300
    style: str = 'seaborn-v0_8'
    colors: List[str] = None
    save_format: str = 'png'
    save_path: Optional[str] = None
    show_plot: bool = True
    
    def __post_init__(self):
        if self.colors is None:
            self.colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', 
                          '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']


class PBBMVisualizer:
    """Visualization tools for PBBM simulation results."""
    
    def __init__(self, config: PlotConfig = None):
        """Initialize visualizer with configuration."""
        if not HAS_PLOTTING:
            raise ImportError("Matplotlib and seaborn are required for visualization")
            
        self.config = config or PlotConfig()
        
        # Set plotting style
        if self.config.style:
            plt.style.use(self.config.style)
    
    def plot_pk_profiles(self, 
                        results: Union[SequentialPBBMResults, SimulationResults],
                        compartments: List[str] = None,
                        log_scale: bool = False,
                        show_individual_models: bool = False) -> Figure:
        """Plot PK concentration profiles over time.
        
        Args:
            results: Simulation results
            compartments: Which compartments to plot (default: plasma)
            log_scale: Use log scale for y-axis
            show_individual_models: Show breakdown by individual models
            
        Returns:
            Matplotlib figure
        """
        if compartments is None:
            compartments = ['plasma_concentration']
            
        fig, axes = plt.subplots(figsize=self.config.figsize)
        if not isinstance(axes, list):
            axes = [axes]
        
        # Extract time series data
        if HAS_SEQUENTIAL and isinstance(results, SequentialPBBMResults):
            time_points = results.t
            self._plot_sequential_pk(results, axes[0], compartments, log_scale)
        elif isinstance(results, SimulationResults):
            time_points = results.time_s / 3600.0  # Convert to hours
            self._plot_simulation_pk(results, axes[0], compartments, log_scale)
        elif hasattr(results, 't') and hasattr(results, 'get_model_outputs'):
            # Handle mock objects for testing
            time_points = results.t
            self._plot_sequential_pk(results, axes[0], compartments, log_scale)
        else:
            raise ValueError(f"Unsupported results type: {type(results)}")
        
        # Styling
        axes[0].set_xlabel('Time (h)')
        axes[0].set_ylabel('Concentration (pmol/mL)')
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()
        
        if log_scale:
            axes[0].set_yscale('log')
            axes[0].set_ylabel('Concentration (pmol/mL, log scale)')
        
        plt.tight_layout()
        
        if self.config.save_path:
            self._save_figure(fig, 'pk_profiles')
        
        return fig
    
    def _plot_sequential_pk(self, results, ax: Axes, 
                           compartments: List[str], log_scale: bool):
        """Plot PK data from sequential results."""
        if not HAS_SEQUENTIAL:
            raise RuntimeError("SequentialPBBMResults plotting requested but composition module not available")
        time_points = results.t
        
        # Find PK models in the results
        pk_models = [name for name in results.models.keys() 
                    if any(output.name == 'plasma_concentration' 
                          for output in results.models[name].provides)]
        
        for i, model_name in enumerate(pk_models):
            try:
                outputs = results.get_model_outputs(model_name, time_points)
                concentrations = [out.get('plasma_concentration', 0.0) for out in outputs]
                
                ax.plot(time_points, concentrations, 
                       color=self.config.colors[i % len(self.config.colors)],
                       linewidth=2, label=f'{model_name.replace("_", " ").title()}')
                       
            except Exception as e:
                print(f"Warning: Could not plot {model_name}: {e}")
    
    def _plot_simulation_pk(self, results: SimulationResults, ax: Axes,
                           compartments: List[str], log_scale: bool):
        """Plot PK data from simulation results."""
        # Extract plasma concentration from SimulationResults
        time_hours = results.time_s / 3600.0
        plasma_conc = results.pk_results.plasma_concentration
        
        ax.plot(time_hours, plasma_conc, linewidth=2, 
               label='Plasma Concentration')
    
    def plot_lung_regional_distribution(self,
                                       results: Union[SequentialPBBMResults, SimulationResults],
                                       time_point: float = None,
                                       show_binding: bool = True) -> Figure:
        """Plot lung regional drug distribution.
        
        Args:
            results: Simulation results
            time_point: Specific time to show (default: final time)
            show_binding: Show bound vs unbound fractions
            
        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()
        
        # Extract lung model data
        lung_data = self._extract_lung_data(results, time_point)
        
        if not lung_data:
            # Create placeholder plot
            for ax in axes:
                ax.text(0.5, 0.5, 'No lung regional data available', 
                       ha='center', va='center', transform=ax.transAxes)
            return fig
        
        # Plot 1: Regional amounts
        regions = list(lung_data.keys())
        amounts = [lung_data[region].get('total_amount', 0) for region in regions]
        
        bars = axes[0].bar(regions, amounts, color=self.config.colors[0])
        axes[0].set_title('Regional Drug Amount')
        axes[0].set_ylabel('Amount (pmol)')
        axes[0].tick_params(axis='x', rotation=45)
        
        # Plot 2: Binding fractions (if available)
        if show_binding and any('bound_fraction' in data for data in lung_data.values()):
            bound_fractions = [lung_data[region].get('bound_fraction', 0) for region in regions]
            axes[1].bar(regions, bound_fractions, color=self.config.colors[1])
            axes[1].set_title('Bound Fraction by Region')
            axes[1].set_ylabel('Bound Fraction')
            axes[1].set_ylim(0, 1)
            axes[1].tick_params(axis='x', rotation=45)
        
        # Plot 3: Dissolution state (if available)
        if any('dissolved_amount' in data for data in lung_data.values()):
            dissolved = [lung_data[region].get('dissolved_amount', 0) for region in regions]
            undissolved = [lung_data[region].get('undissolved_amount', 0) for region in regions]
            
            width = 0.35
            x = np.arange(len(regions))
            axes[2].bar(x - width/2, dissolved, width, label='Dissolved', color=self.config.colors[2])
            axes[2].bar(x + width/2, undissolved, width, label='Undissolved', color=self.config.colors[3])
            axes[2].set_title('Dissolution State')
            axes[2].set_ylabel('Amount (pmol)')
            axes[2].set_xticks(x)
            axes[2].set_xticklabels(regions, rotation=45)
            axes[2].legend()
        
        # Plot 4: Clearance rates (if available)
        if any('clearance_rate' in data for data in lung_data.values()):
            clearance_rates = [lung_data[region].get('clearance_rate', 0) for region in regions]
            axes[3].bar(regions, clearance_rates, color=self.config.colors[4])
            axes[3].set_title('Regional Clearance Rates')
            axes[3].set_ylabel('Clearance Rate (pmol/s)')
            axes[3].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        if self.config.save_path:
            self._save_figure(fig, 'lung_regional_distribution')
        
        return fig
    
    def _extract_lung_data(self, results: Union[SequentialPBBMResults, SimulationResults],
                          time_point: float = None) -> Dict[str, Dict[str, float]]:
        """Extract lung regional data from results."""
        lung_data = {}
        
        if isinstance(results, SimulationResults):
            # Extract data from regional_results
            if results.regional_results:
                for region_name, regional_data in results.regional_results.items():
                    # Get final values or at specific time point
                    lung_data[region_name] = {
                        'total_amount': regional_data.total_amounts[-1],
                        'epithelium_amount': regional_data.epithelium_amounts[-1],
                        'tissue_amount': regional_data.tissue_amounts[-1],
                        'elf_amount': regional_data.elf_amounts[-1] if regional_data.elf_amounts is not None else 0.0
                    }
            return lung_data
        
        elif isinstance(results, SequentialPBBMResults):
            # Find lung models
            lung_models = [name for name in results.models.keys() 
                          if 'lung' in name.lower()]
            
            if not lung_models:
                return lung_data
            
            model_name = lung_models[0]  # Use first lung model
            if time_point is None:
                time_point = results.t[-1]
            
            try:
                outputs = results.get_model_outputs(model_name, [time_point])
                if outputs:
                    output = outputs[0]
                    # Extract regional data from model outputs
                    for key, value in output.items():
                        if 'regional' in key or any(region in key.lower() 
                                                   for region in ['central', 'peripheral', 'alveolar']):
                            region_name = key.split('_')[0] if '_' in key else key
                            if region_name not in lung_data:
                                lung_data[region_name] = {}
                            lung_data[region_name][key] = value
            except Exception as e:
                print(f"Warning: Could not extract lung data: {e}")
        
        return lung_data
    
    def plot_bioequivalence_comparison(self,
                                     test_results: Union[SequentialPBBMResults, SimulationResults],
                                     ref_results: Union[SequentialPBBMResults, SimulationResults],
                                     be_metrics: Dict[str, Dict[str, float]] = None) -> Figure:
        """Plot bioequivalence comparison between test and reference.
        
        Args:
            test_results: Test formulation results
            ref_results: Reference formulation results
            be_metrics: Pre-calculated BE metrics (optional)
            
        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Plot 1: Concentration-time profiles
        self._plot_be_profiles(axes[0, 0], test_results, ref_results)
        
        # Plot 2: AUC comparison
        self._plot_be_metrics(axes[0, 1], test_results, ref_results, 'auc_0_inf', be_metrics)
        
        # Plot 3: Cmax comparison  
        self._plot_be_metrics(axes[1, 0], test_results, ref_results, 'cmax', be_metrics)
        
        # Plot 4: BE acceptance limits
        if be_metrics:
            self._plot_be_limits(axes[1, 1], be_metrics)
        
        plt.tight_layout()
        
        if self.config.save_path:
            self._save_figure(fig, 'bioequivalence_comparison')
        
        return fig
    
    def _plot_be_profiles(self, ax: Axes,
                         test_results: Union[SequentialPBBMResults, SimulationResults],
                         ref_results: Union[SequentialPBBMResults, SimulationResults]):
        """Plot concentration-time profiles for BE comparison."""
        # Extract plasma concentrations
        test_time, test_conc = self._extract_plasma_data(test_results)
        ref_time, ref_conc = self._extract_plasma_data(ref_results)
        
        ax.plot(test_time, test_conc, 'b-', linewidth=2, label='Test')
        ax.plot(ref_time, ref_conc, 'r-', linewidth=2, label='Reference')
        
        ax.set_xlabel('Time (h)')
        ax.set_ylabel('Plasma Concentration (pmol/mL)')
        ax.set_title('Concentration-Time Profiles')
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    def _plot_be_metrics(self, ax: Axes,
                        test_results: Union[SequentialPBBMResults, SimulationResults],
                        ref_results: Union[SequentialPBBMResults, SimulationResults],
                        metric: str,
                        be_metrics: Dict[str, Dict[str, float]] = None):
        """Plot BE metric comparison."""
        # Calculate metrics if not provided
        if be_metrics is None or metric not in be_metrics:
            from .bioequivalence import NewArchitectureBioequivalenceAssessor, VBEStudyConfig
            config = VBEStudyConfig()  # Use default config
            assessor = NewArchitectureBioequivalenceAssessor(config)
            
            # For visualization purposes, create simplified BE comparison
            test_pk = self._calculate_pk_metrics(test_results)
            ref_pk = self._calculate_pk_metrics(ref_results)
            
            # Create simplified BE comparison
            if metric in test_pk and metric in ref_pk and test_pk[metric] > 0 and ref_pk[metric] > 0:
                gmr = test_pk[metric] / ref_pk[metric]
                # Simple mock confidence interval for visualization
                ci_lower = gmr * 0.85
                ci_upper = gmr * 1.15
                be_metrics = {
                    'geometric_mean_ratio': gmr,
                    'ci_lower': ci_lower,
                    'ci_upper': ci_upper,
                    'within_be_limits': 0.8 <= ci_lower and ci_upper <= 1.25
                }
            else:
                be_metrics = {
                    'geometric_mean_ratio': 1.0,
                    'ci_lower': 0.9,
                    'ci_upper': 1.1,
                    'within_be_limits': True
                }
        else:
            be_metrics = be_metrics[metric]
        
        # Create comparison plot
        gmr = be_metrics.get('geometric_mean_ratio', 1.0)
        ci_lower = be_metrics.get('ci_lower', gmr)
        ci_upper = be_metrics.get('ci_upper', gmr)
        
        ax.errorbar([1], [gmr], yerr=[[gmr - ci_lower], [ci_upper - gmr]], 
                   fmt='o', capsize=10, capthick=2, color=self.config.colors[0])
        
        # Add acceptance limits
        ax.axhline(y=0.8, color='red', linestyle='--', alpha=0.7, label='BE Limits')
        ax.axhline(y=1.25, color='red', linestyle='--', alpha=0.7)
        ax.axhspan(0.8, 1.25, alpha=0.1, color='green', label='Acceptance Range')
        
        ax.set_xlim(0.5, 1.5)
        ax.set_ylim(0.6, 1.5)
        ax.set_ylabel('Geometric Mean Ratio')
        ax.set_title(f'{metric.upper()} Bioequivalence')
        ax.set_xticks([1])
        ax.set_xticklabels(['Test/Reference'])
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_be_limits(self, ax: Axes, be_metrics: Dict[str, Dict[str, float]]):
        """Plot BE limits summary."""
        metrics = list(be_metrics.keys())
        gmrs = [be_metrics[m].get('geometric_mean_ratio', 1.0) for m in metrics]
        ci_lowers = [be_metrics[m].get('ci_lower', gmr) for m, gmr in zip(metrics, gmrs)]
        ci_uppers = [be_metrics[m].get('ci_upper', gmr) for m, gmr in zip(metrics, gmrs)]
        within_limits = [be_metrics[m].get('within_be_limits', False) for m in metrics]
        
        y_pos = np.arange(len(metrics))
        colors = ['green' if w else 'red' for w in within_limits]
        
        ax.barh(y_pos, gmrs, xerr=[np.array(gmrs) - np.array(ci_lowers),
                                  np.array(ci_uppers) - np.array(gmrs)],
                capsize=5, color=colors, alpha=0.7)
        
        ax.axvline(x=0.8, color='red', linestyle='--', alpha=0.7)
        ax.axvline(x=1.25, color='red', linestyle='--', alpha=0.7)
        ax.axvspan(0.8, 1.25, alpha=0.1, color='green')
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels([m.upper() for m in metrics])
        ax.set_xlabel('Geometric Mean Ratio')
        ax.set_title('Bioequivalence Summary')
        ax.set_xlim(0.6, 1.5)
        ax.grid(True, alpha=0.3)
    
    def _extract_plasma_data(self, results: Union[SequentialPBBMResults, SimulationResults]
                           ) -> Tuple[np.ndarray, np.ndarray]:
        """Extract plasma concentration data from results."""
        if isinstance(results, SequentialPBBMResults):
            time_points = results.t
            # Find PK model with plasma concentration
            for model_name in results.models.keys():
                try:
                    outputs = results.get_model_outputs(model_name, time_points)
                    concentrations = [out.get('plasma_concentration', 0.0) for out in outputs]
                    if any(c > 0 for c in concentrations):
                        return time_points, np.array(concentrations)
                except:
                    continue
            return np.array([]), np.array([])
        
        elif isinstance(results, SimulationResults):
            time_hours = results.time_s / 3600.0  # Convert to hours
            concentrations = results.pk_results.plasma_concentration
            return time_hours, concentrations
        elif hasattr(results, 't') and hasattr(results, 'get_model_outputs'):
            # Handle mock objects for testing
            time_points = results.t
            # Find PK model with plasma concentration
            for model_name in results.models.keys():
                try:
                    outputs = results.get_model_outputs(model_name, time_points)
                    concentrations = [out.get('plasma_concentration', 0.0) for out in outputs]
                    if any(c > 0 for c in concentrations):
                        return time_points, np.array(concentrations)
                except:
                    continue
            return np.array([]), np.array([])
        
        return np.array([]), np.array([])
    
    def _calculate_pk_metrics(self, results: Union[SequentialPBBMResults, SimulationResults]
                            ) -> Dict[str, float]:
        """Calculate PK metrics from results."""
        if isinstance(results, SimulationResults):
            # Extract existing PK metrics from SimulationResults
            pk_results = results.pk_results
            return {
                'auc_0_inf': pk_results.auc_pmol_h_per_ml if pk_results.auc_pmol_h_per_ml else 0.0,
                'cmax': pk_results.cmax_pmol_per_ml if pk_results.cmax_pmol_per_ml else 0.0,
                'tmax': pk_results.tmax_h if pk_results.tmax_h else 0.0,
                'half_life': 0.0  # Not available in current PKResultsData
            }
        
        # For SequentialPBBMResults and mock objects, calculate from concentration data
        time_points, concentrations = self._extract_plasma_data(results)
        
        if len(concentrations) == 0:
            return {}
        
        # Import analysis functions
        from ...services.analysis import calculate_pk_metrics
        return calculate_pk_metrics(time_points, concentrations)
    
    def plot_sensitivity_analysis(self,
                                 parameter_names: List[str],
                                 sensitivity_indices: Dict[str, np.ndarray],
                                 output_metrics: List[str] = None) -> Figure:
        """Plot sensitivity analysis results.
        
        Args:
            parameter_names: Names of varied parameters
            sensitivity_indices: Sensitivity indices for each output
            output_metrics: Names of output metrics
            
        Returns:
            Matplotlib figure
        """
        if output_metrics is None:
            output_metrics = list(sensitivity_indices.keys())
        
        n_metrics = len(output_metrics)
        fig, axes = plt.subplots(1, n_metrics, figsize=(5*n_metrics, 6))
        
        if n_metrics == 1:
            axes = [axes]
        
        for i, metric in enumerate(output_metrics):
            if metric not in sensitivity_indices:
                continue
                
            indices = sensitivity_indices[metric]
            
            # Create horizontal bar plot
            y_pos = np.arange(len(parameter_names))
            bars = axes[i].barh(y_pos, indices, color=self.config.colors[i])
            
            # Color bars by magnitude
            for bar, index in zip(bars, indices):
                if abs(index) > 0.1:  # High sensitivity
                    bar.set_color('red')
                elif abs(index) > 0.05:  # Medium sensitivity
                    bar.set_color('orange')
                else:  # Low sensitivity
                    bar.set_color('lightblue')
            
            axes[i].set_yticks(y_pos)
            axes[i].set_yticklabels(parameter_names)
            axes[i].set_xlabel('Sensitivity Index')
            axes[i].set_title(f'{metric} Sensitivity')
            axes[i].grid(True, alpha=0.3)
            
            # Add vertical line at zero
            axes[i].axvline(x=0, color='black', linestyle='-', alpha=0.3)
        
        plt.tight_layout()
        
        if self.config.save_path:
            self._save_figure(fig, 'sensitivity_analysis')
        
        return fig

    def plot_time_series(self, time_h: np.ndarray, values: Dict[str, np.ndarray], title: str = 'Time Series', ylabel: str = 'Value', log_scale: bool = False) -> Figure:
        """Simple line plot for parameters vs time (e.g., fluxes, amounts)."""
        fig, ax = plt.subplots(figsize=self.config.figsize)
        for i, (name, series) in enumerate(values.items()):
            ax.plot(time_h, series, label=name, color=self.config.colors[i % len(self.config.colors)])
        ax.set_title(title)
        ax.set_xlabel('Time (h)')
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        if log_scale:
            ax.set_yscale('log')
        ax.legend()
        plt.tight_layout()
        if self.config.save_path:
            self._save_figure(fig, title.replace(' ', '_').lower())
        return fig

    def vbe_trials_to_df(self, trials: List[Dict[str, Any]]) -> pd.DataFrame:
        """Convert a list of vbe_multi_from_results dicts into a tidy DataFrame.

        Each element in trials is expected to be a dict: metric_id -> vbe_result,
        where vbe_result has keys 'gmr', 'ci_lower', 'ci_upper'.
        """
        rows: List[Dict[str, Any]] = []
        for i, trial in enumerate(trials):
            trial_id = i + 1
            for metric_id, res in trial.items():
                rows.append({
                    'Trial': trial_id,
                    'Metric': metric_id,
                    'GMR': res.get('gmr'),
                    'CI_Lower': res.get('ci_lower'),
                    'CI_Upper': res.get('ci_upper'),
                })
        return pd.DataFrame(rows)

    def plot_be_trials(self,
                        trials_df: pd.DataFrame,
                        metrics: Optional[List[str]] = None,
                        ref_interval: Tuple[float, float] = (0.8, 1.25),
                        xlim: Tuple[float, float] = (0.6, 1.4),
                        figsize_per_subplot: Tuple[float, float] = (4, 8),
                        title: str = 'BE Trials') -> Figure:
        """Plot GMR and CI for each bootstrap trial per metric.

        Expects DataFrame columns: ['Trial','Metric','GMR','CI_Lower','CI_Upper']
        """
        df = trials_df.copy()
        if metrics is None:
            metrics = sorted(df['Metric'].unique())
        ncols = max(1, len(metrics))
        fig, axes = plt.subplots(1, ncols, figsize=(figsize_per_subplot[0] * ncols, figsize_per_subplot[1]), sharey=True)
        if ncols == 1:
            axes = [axes]

        for idx, metric in enumerate(metrics):
            ax = axes[idx]
            sub = df[df['Metric'] == metric].sort_values('Trial').copy()
            if sub.empty:
                ax.set_visible(False)
                continue
            sub['y'] = sub['Trial'].rank(method='first') - 1
            x = sub['GMR']
            xerr = np.vstack([x - sub['CI_Lower'], sub['CI_Upper'] - x])
            ax.errorbar(x=x, y=sub['y'], xerr=xerr, fmt='o', color=self.config.colors[idx % len(self.config.colors)],
                        ecolor=self.config.colors[idx % len(self.config.colors)], capsize=2, markersize=5, linestyle='none')
            ax.set_title(str(metric), fontsize=12)
            ax.set_xlabel('Geometric Mean Ratio')
            ax.set_xlim(*xlim)
            ax.axvline(ref_interval[0], color='red', ls='dashed')
            ax.axvline(ref_interval[1], color='red', ls='dashed')
            ax.set_ylabel('Trials' if idx == 0 else '')
            ax.invert_yaxis()
            ax.grid(axis='x', linestyle=':', alpha=0.5)

        plt.suptitle(title, y=1.02)
        plt.tight_layout()
        return fig
    def plot_time_series(self, time_h: np.ndarray, values: Dict[str, np.ndarray], title: str = 'Time Series', ylabel: str = 'Value', log_scale: bool = False) -> Figure:
        """Simple line plot for parameters vs time (e.g., fluxes, amounts)."""
        fig, ax = plt.subplots(figsize=self.config.figsize)
        for i, (name, series) in enumerate(values.items()):
            ax.plot(time_h, series, label=name, color=self.config.colors[i % len(self.config.colors)])
        ax.set_title(title)
        ax.set_xlabel('Time (h)')
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        if log_scale:
            ax.set_yscale('log')
        ax.legend()
        plt.tight_layout()
        if self.config.save_path:
            self._save_figure(fig, title.replace(' ', '_').lower())
        return fig
    
    def plot_parameter_estimation_results(self,
                                        parameters: Dict[str, float],
                                        confidence_intervals: Dict[str, Tuple[float, float]],
                                        fit_quality: Dict[str, float] = None) -> Figure:
        """Plot parameter estimation results.
        
        Args:
            parameters: Estimated parameter values
            confidence_intervals: 95% confidence intervals for parameters
            fit_quality: Goodness of fit metrics
            
        Returns:
            Matplotlib figure
        """
        param_names = list(parameters.keys())
        param_values = list(parameters.values())
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Plot 1: Parameter estimates with confidence intervals
        y_pos = np.arange(len(param_names))
        
        # Calculate error bars
        lower_errors = []
        upper_errors = []
        for name in param_names:
            if name in confidence_intervals:
                ci_lower, ci_upper = confidence_intervals[name]
                lower_errors.append(parameters[name] - ci_lower)
                upper_errors.append(ci_upper - parameters[name])
            else:
                lower_errors.append(0)
                upper_errors.append(0)
        
        axes[0].barh(y_pos, param_values, 
                    xerr=[lower_errors, upper_errors],
                    capsize=5, color=self.config.colors[0], alpha=0.7)
        
        axes[0].set_yticks(y_pos)
        axes[0].set_yticklabels(param_names)
        axes[0].set_xlabel('Parameter Value')
        axes[0].set_title('Parameter Estimates with 95% CI')
        axes[0].grid(True, alpha=0.3)
        
        # Plot 2: Fit quality metrics (if provided)
        if fit_quality:
            metrics = list(fit_quality.keys())
            values = list(fit_quality.values())
            
            bars = axes[1].bar(metrics, values, color=self.config.colors[1])
            axes[1].set_ylabel('Value')
            axes[1].set_title('Goodness of Fit Metrics')
            axes[1].tick_params(axis='x', rotation=45)
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                height = bar.get_height()
                axes[1].text(bar.get_x() + bar.get_width()/2., height,
                           f'{value:.3f}', ha='center', va='bottom')
        else:
            axes[1].text(0.5, 0.5, 'No fit quality metrics provided',
                        ha='center', va='center', transform=axes[1].transAxes)
        
        plt.tight_layout()
        
        if self.config.save_path:
            self._save_figure(fig, 'parameter_estimation')
        
        return fig
    
    def _save_figure(self, fig: Figure, filename: str):
        """Save figure to specified path."""
        if not self.config.save_path:
            return
        
        save_path = Path(self.config.save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        full_path = save_path / f"{filename}.{self.config.save_format}"
        fig.savefig(full_path, dpi=self.config.dpi, bbox_inches='tight')
        print(f"📊 Plot saved: {full_path}")
    
    def create_dashboard(self,
                        results: Union[SequentialPBBMResults, SimulationResults],
                        include_lung_distribution: bool = True,
                        include_pk_metrics: bool = True) -> Figure:
        """Create comprehensive dashboard with multiple plots.
        
        Args:
            results: Simulation results
            include_lung_distribution: Include lung regional plots
            include_pk_metrics: Include PK metrics summary
            
        Returns:
            Matplotlib figure
        """
        # Determine subplot layout
        n_plots = 2  # Always include PK profile and summary
        if include_lung_distribution:
            n_plots += 1
        if include_pk_metrics:
            n_plots += 1
        
        fig = plt.figure(figsize=(16, 4*n_plots))
        
        plot_idx = 1
        
        # PK concentration profile
        ax1 = fig.add_subplot(n_plots, 1, plot_idx)
        time_points, concentrations = self._extract_plasma_data(results)
        if len(concentrations) > 0:
            ax1.plot(time_points, concentrations, 'b-', linewidth=2)
            ax1.set_xlabel('Time (h)')
            ax1.set_ylabel('Plasma Concentration (pmol/mL)')
            ax1.set_title('Plasma Concentration Profile')
            ax1.grid(True, alpha=0.3)
        plot_idx += 1
        
        # Lung distribution (if requested)
        if include_lung_distribution:
            ax2 = fig.add_subplot(n_plots, 1, plot_idx)
            lung_data = self._extract_lung_data(results)
            if lung_data:
                regions = list(lung_data.keys())
                amounts = [lung_data[region].get('total_amount', 0) for region in regions]
                ax2.bar(regions, amounts, color=self.config.colors[1])
                ax2.set_ylabel('Amount (pmol)')
                ax2.set_title('Lung Regional Distribution')
                ax2.tick_params(axis='x', rotation=45)
            plot_idx += 1
        
        # PK metrics (if requested)
        if include_pk_metrics:
            ax3 = fig.add_subplot(n_plots, 1, plot_idx)
            pk_metrics = self._calculate_pk_metrics(results)
            if pk_metrics:
                metrics = ['auc_0_inf', 'cmax', 'tmax', 'half_life']
                available_metrics = [m for m in metrics if m in pk_metrics and not np.isnan(pk_metrics[m])]
                values = [pk_metrics[m] for m in available_metrics]
                
                bars = ax3.bar(available_metrics, values, color=self.config.colors[2])
                ax3.set_ylabel('Value')
                ax3.set_title('PK Metrics Summary')
                ax3.tick_params(axis='x', rotation=45)
                
                # Add value labels
                for bar, value in zip(bars, values):
                    height = bar.get_height()
                    ax3.text(bar.get_x() + bar.get_width()/2., height,
                           f'{value:.2g}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        if self.config.save_path:
            self._save_figure(fig, 'dashboard')
        
        return fig


def create_quick_visualizer(save_path: str = None) -> PBBMVisualizer:
    """Create a visualizer with default configuration.
    
    Args:
        save_path: Optional path to save plots
        
    Returns:
        Configured PBBMVisualizer instance
    """
    config = PlotConfig(save_path=save_path)
    return PBBMVisualizer(config)


# Convenience functions for quick plotting
def plot_pk_comparison(test_results: Union[SequentialPBBMResults, SimulationResults],
                      ref_results: Union[SequentialPBBMResults, SimulationResults] = None,
                      save_path: str = None) -> Figure:
    """Quick function to plot PK comparison.
    
    Args:
        test_results: Test results to plot
        ref_results: Reference results for comparison (optional)
        save_path: Path to save plot
        
    Returns:
        Matplotlib figure
    """
    if not HAS_PLOTTING:
        raise ImportError("Matplotlib is required for plotting")
    
    visualizer = create_quick_visualizer(save_path)
    
    if ref_results is not None:
        return visualizer.plot_bioequivalence_comparison(test_results, ref_results)
    else:
        return visualizer.plot_pk_profiles(test_results)


def plot_lung_distribution(results: Union[SequentialPBBMResults, SimulationResults],
                          save_path: str = None) -> Figure:
    """Quick function to plot lung distribution.
    
    Args:
        results: Simulation results
        save_path: Path to save plot
        
    Returns:
        Matplotlib figure
    """
    if not HAS_PLOTTING:
        raise ImportError("Matplotlib is required for plotting")
    
    visualizer = create_quick_visualizer(save_path)
    return visualizer.plot_lung_regional_distribution(results)


def create_results_dashboard(results: Union[SequentialPBBMResults, SimulationResults],
                           save_path: str = None) -> Figure:
    """Quick function to create comprehensive dashboard.
    
    Args:
        results: Simulation results
        save_path: Path to save plot
        
    Returns:
        Matplotlib figure
    """
    if not HAS_PLOTTING:
        raise ImportError("Matplotlib is required for plotting")
    
    visualizer = create_quick_visualizer(save_path)
    return visualizer.create_dashboard(results)
