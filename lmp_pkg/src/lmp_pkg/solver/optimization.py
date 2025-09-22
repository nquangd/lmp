"""Optimization solvers for parameter fitting and model calibration."""

from __future__ import annotations
import copy
import numpy as np
from dataclasses import dataclass
from typing import Dict, Any, Callable, Optional, Union, Tuple, Sequence, List
from scipy.optimize import minimize, differential_evolution, curve_fit
import warnings

from .. import app_api
from ..contracts.types import RunResult
from ..config.model import AppConfig
from .base import OptimizationSolverBase


class OptimizationSolver(OptimizationSolverBase):
    """General-purpose optimization solver for parameter estimation.
    
    Supports various optimization algorithms for:
    - Parameter fitting to experimental data
    - Model calibration and validation
    - Sensitivity analysis support
    """
    
    def __init__(self, method: str = "L-BFGS-B", **default_options):
        super().__init__("scipy_optimizer", method)
        self.default_options.update(default_options)
    
    def optimize(
        self,
        objective: Callable,
        x0: np.ndarray,
        bounds: Optional[list] = None,
        **options
    ) -> Any:
        """Solve optimization problem using scipy.optimize.
        
        Args:
            objective: Objective function to minimize f(x) -> float
            x0: Initial parameter guess
            bounds: Parameter bounds [(min, max), ...] or None
            **options: Optimizer options
            
        Returns:
            scipy optimization result object
        """
        # Merge options
        opt_options = self.default_options.copy()
        opt_options.update(options)
        
        # Extract method
        method = opt_options.pop('method', self.method)
        
        try:
            if method in ['differential_evolution', 'DE']:
                # Filter allowed options for differential_evolution
                de_allowed = {
                    'strategy', 'maxiter', 'popsize', 'tol', 'mutation', 'recombination',
                    'seed', 'callback', 'disp', 'polish', 'init', 'atol', 'updating',
                    'workers', 'constraints'
                }
                de_kwargs = {k: v for k, v in opt_options.items() if k in de_allowed}
                if bounds is None:
                    raise ValueError("Differential evolution requires bounds")
                result = differential_evolution(objective, bounds, **de_kwargs)
            else:
                # Local optimization via minimize; pass method-specific options via 'options'
                tol = opt_options.pop('tol', None)
                minimize_allowed = {'maxiter', 'ftol', 'gtol', 'maxfun', 'eps', 'disp'}
                minimize_options = {k: v for k, v in opt_options.items() if k in minimize_allowed}
                result = minimize(
                    objective,
                    x0,
                    method=method,
                    bounds=bounds,
                    tol=tol,
                    options=minimize_options if minimize_options else None,
                )
            
            # Store convergence info
            self.convergence_info = {
                'success': result.success,
                'message': result.message,
                'nfev': result.nfev,
                'njev': getattr(result, 'njev', None),
                'fun': result.fun,
                'method': method
            }
            
            self.last_result = result
            return result
            
        except Exception as e:
            self.convergence_info = {
                'success': False,
                'message': f"Optimization failed: {str(e)}",
                'error': e
            }
            raise

    # Provide a generic solve() to satisfy abstract base class and delegate to optimize
    def solve(
        self,
        objective: Callable,
        x0: np.ndarray,
        bounds: Optional[list] = None,
        **options
    ) -> Any:
        return self.optimize(objective, x0, bounds=bounds, **options)


def optimize_parameters(
    objective_func: Callable,
    initial_params: np.ndarray,
    bounds: Optional[list] = None,
    method: str = "L-BFGS-B",
    **options
) -> Any:
    """Convenience function for parameter optimization.
    
    Args:
        objective_func: Function to minimize
        initial_params: Initial parameter values
        bounds: Parameter bounds
        method: Optimization method
        **options: Additional optimizer options
        
    Returns:
        Optimization result
    """
    solver = OptimizationSolver(method=method, **options)
    return solver.optimize(objective_func, initial_params, bounds)


# ---------------------------------------------------------------------------
#  High-level parameter fitting via the LMP pipeline
# ---------------------------------------------------------------------------


@dataclass
class ParameterDefinition:
    """Specification of an optimisable configuration parameter."""

    name: str
    path: str
    bounds: Tuple[float, float]
    transform: Optional[Callable[[float], float]] = None
    inverse_transform: Optional[Callable[[float], float]] = None

    def to_config_value(self, x: float) -> float:
        return self.inverse_transform(x) if self.inverse_transform else x

    def to_optim_value(self, x: float) -> float:
        return self.transform(x) if self.transform else x


def _vector_to_overrides(
    params: Sequence[ParameterDefinition],
    vector: np.ndarray,
) -> Dict[str, Any]:
    overrides: Dict[str, Any] = {}
    for p, value in zip(params, vector):
        overrides[p.path] = p.to_config_value(value)
    return overrides


def _extract_pk_profile(result: RunResult) -> Tuple[np.ndarray, np.ndarray]:
    if result.pbbk and result.pbbk.comprehensive is not None:
        comp = result.pbbk.comprehensive
        return comp.time_s, comp.pk_data.plasma_concentration_ng_per_ml
    if result.pk is not None:
        return result.pk.t, result.pk.conc_plasma
    raise ValueError("Simulation result does not contain PK output")


class ParameterFitter:
    """Optimise configuration parameters against observed PK data."""

    def __init__(
        self,
        base_config: AppConfig,
        parameters: Sequence[ParameterDefinition],
        observed_time_s: np.ndarray,
        observed_concentration: np.ndarray,
        metric: Optional[Callable[[np.ndarray, np.ndarray], float]] = None,
    ) -> None:
        self.base_config = base_config
        self.parameters = list(parameters)
        self.observed_time = np.asarray(observed_time_s)
        self.observed_concentration = np.asarray(observed_concentration)
        self.metric = metric or self._default_metric

    def _simulate(self, vector: np.ndarray) -> RunResult:
        overrides = _vector_to_overrides(self.parameters, vector)
        return app_api.run_single_simulation(
            copy.deepcopy(self.base_config),
            parameter_overrides=overrides,
        )

    def _default_metric(self, predicted: np.ndarray, observed: np.ndarray) -> float:
        residuals = observed - predicted
        return float(np.sum(residuals ** 2))

    def evaluate(self, vector: np.ndarray) -> float:
        result = self._simulate(vector)
        time, conc = _extract_pk_profile(result)
        interpolated = np.interp(self.observed_time, time, conc)
        return self.metric(interpolated, self.observed_concentration)

    def fit(
        self,
        initial_guess: np.ndarray,
        method: str = "L-BFGS-B",
        bounds: Optional[List[Tuple[float, float]]] = None,
        **options,
    ) -> Any:
        if bounds is None:
            bounds = [p.bounds for p in self.parameters]
        solver = OptimizationSolver(method=method, **options)
        return solver.optimize(self.evaluate, initial_guess, bounds=bounds)


def fit_model_to_data(
    model_func: Callable,
    time_points: np.ndarray,
    observed_data: np.ndarray,
    initial_params: np.ndarray,
    param_bounds: Optional[list] = None,
    weights: Optional[np.ndarray] = None,
    method: str = "curve_fit"
) -> Dict[str, Any]:
    """Fit model parameters to experimental data.
    
    Args:
        model_func: Model function y = f(t, *params)
        time_points: Time points for observations
        observed_data: Observed data values
        initial_params: Initial parameter guess
        param_bounds: Parameter bounds for curve_fit
        weights: Optional data weights (1/sigma)
        method: Fitting method ("curve_fit" or "minimize")
        
    Returns:
        Dictionary with fitted parameters, covariance, and fit statistics
    """
    if method == "curve_fit":
        try:
            # Use scipy curve_fit
            bounds_tuple = None
            if param_bounds is not None:
                lower_bounds = [b[0] for b in param_bounds]
                upper_bounds = [b[1] for b in param_bounds]
                bounds_tuple = (lower_bounds, upper_bounds)
            
            popt, pcov = curve_fit(
                model_func,
                time_points,
                observed_data,
                p0=initial_params,
                bounds=bounds_tuple,
                sigma=1/weights if weights is not None else None,
                absolute_sigma=True
            )
            
            # Calculate fit statistics
            predicted = model_func(time_points, *popt)
            residuals = observed_data - predicted
            ss_res = np.sum(residuals**2)
            ss_tot = np.sum((observed_data - np.mean(observed_data))**2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
            
            # Parameter uncertainties
            param_errors = np.sqrt(np.diag(pcov)) if pcov is not None else None
            
            return {
                'fitted_params': popt,
                'param_covariance': pcov,
                'param_errors': param_errors,
                'predicted_values': predicted,
                'residuals': residuals,
                'r_squared': r_squared,
                'aic': len(observed_data) * np.log(ss_res / len(observed_data)) + 2 * len(popt),
                'success': True
            }
            
        except Exception as e:
            warnings.warn(f"curve_fit failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'fitted_params': initial_params,
                'param_covariance': None
            }
    
    elif method == "minimize":
        # Use minimize with least squares objective
        def objective(params):
            try:
                predicted = model_func(time_points, *params)
                residuals = observed_data - predicted
                if weights is not None:
                    residuals *= weights
                return np.sum(residuals**2)
            except:
                return 1e10  # Large penalty for invalid parameters
        
        # Convert bounds format
        scipy_bounds = None
        if param_bounds is not None:
            scipy_bounds = [(b[0], b[1]) for b in param_bounds]
        
        result = optimize_parameters(
            objective,
            initial_params,
            bounds=scipy_bounds,
            method="L-BFGS-B"
        )
        
        if result.success:
            fitted_params = result.x
            predicted = model_func(time_points, *fitted_params)
            residuals = observed_data - predicted
            ss_res = np.sum(residuals**2)
            ss_tot = np.sum((observed_data - np.mean(observed_data))**2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
            
            return {
                'fitted_params': fitted_params,
                'param_covariance': None,  # Not available with minimize
                'param_errors': None,
                'predicted_values': predicted,
                'residuals': residuals,
                'r_squared': r_squared,
                'aic': len(observed_data) * np.log(ss_res / len(observed_data)) + 2 * len(fitted_params),
                'success': True,
                'optimization_result': result
            }
        else:
            return {
                'success': False,
                'error': result.message,
                'fitted_params': initial_params,
                'optimization_result': result
            }
    
    else:
        raise ValueError(f"Unknown fitting method: {method}")


def calculate_parameter_sensitivity(
    model_func: Callable,
    params: np.ndarray,
    time_points: np.ndarray,
    param_names: Optional[list] = None,
    perturbation: float = 0.01
) -> Dict[str, np.ndarray]:
    """Calculate parameter sensitivity using finite differences.
    
    Args:
        model_func: Model function f(t, *params)
        params: Nominal parameter values
        time_points: Time points for evaluation
        param_names: Parameter names (optional)
        perturbation: Relative perturbation for sensitivity calculation
        
    Returns:
        Dictionary with sensitivity matrices and indices
    """
    n_params = len(params)
    n_times = len(time_points)
    
    if param_names is None:
        param_names = [f"param_{i}" for i in range(n_params)]
    
    # Nominal model output
    y_nominal = model_func(time_points, *params)
    
    # Sensitivity matrix: S[i,j] = dy_i/dp_j
    sensitivity_matrix = np.zeros((n_times, n_params))
    
    for j, (param_val, param_name) in enumerate(zip(params, param_names)):
        # Perturb parameter
        delta_p = abs(param_val) * perturbation if param_val != 0 else perturbation
        params_perturbed = params.copy()
        params_perturbed[j] += delta_p
        
        try:
            # Calculate perturbed output
            y_perturbed = model_func(time_points, *params_perturbed)
            
            # Finite difference sensitivity
            sensitivity_matrix[:, j] = (y_perturbed - y_nominal) / delta_p
            
        except Exception as e:
            warnings.warn(f"Sensitivity calculation failed for {param_name}: {e}")
            sensitivity_matrix[:, j] = 0.0
    
    # Calculate sensitivity indices
    normalized_sensitivity = np.zeros_like(sensitivity_matrix)
    for j, param_val in enumerate(params):
        if param_val != 0:
            normalized_sensitivity[:, j] = sensitivity_matrix[:, j] * param_val / y_nominal
    
    return {
        'sensitivity_matrix': sensitivity_matrix,
        'normalized_sensitivity': normalized_sensitivity,
        'parameter_names': param_names,
        'time_points': time_points,
        'nominal_output': y_nominal
    }
