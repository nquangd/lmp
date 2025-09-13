"""ODE solver implementations for PBBM and PK models."""

from __future__ import annotations
import numpy as np
from typing import Dict, Any, Callable, Optional, Tuple
from scipy.integrate import solve_ivp
import warnings

from .base import ODESolverBase


class ODESolver(ODESolverBase):
    """Main ODE solver for all model types.
    
    Provides a unified interface for solving ODE systems from:
    - PBBM models (lung absorption)  
    - PK models (systemic kinetics)
    - Chained models (combined systems)
    """
    
    def __init__(self, method: str = "BDF", **default_options):
        super().__init__("scipy_ode_solver", method)
        self.default_options.update(default_options)
    
    def solve_ode(
        self, 
        ode_func: Callable,
        y0: np.ndarray,
        t_span: Tuple[float, float],
        t_eval: Optional[np.ndarray] = None,
        **options
    ) -> Any:
        """Solve an ODE system using scipy.integrate.solve_ivp.
        
        Args:
            ode_func: ODE function dy/dt = f(t, y, *args)
            y0: Initial conditions
            t_span: (t_start, t_end) integration interval
            t_eval: Time points to evaluate solution at
            **options: Solver options (rtol, atol, max_step, etc.)
            
        Returns:
            scipy solve_ivp result object
        """
        # Merge options
        solver_options = self.default_options.copy()
        solver_options.update(options)
        
        # Extract scipy-specific options
        method = solver_options.pop('method', self.method)
        
        try:
            # Solve ODE system
            solution = solve_ivp(
                ode_func,
                t_span,
                y0,
                method=method,
                t_eval=t_eval,
                **solver_options
            )
            
            # Store convergence info
            self.convergence_info = {
                'success': solution.success,
                'message': solution.message,
                'nfev': solution.nfev,
                'njev': getattr(solution, 'njev', None),
                'nlu': getattr(solution, 'nlu', None),
                't_events': getattr(solution, 't_events', None),
                'y_events': getattr(solution, 'y_events', None),
                'method': method
            }
            
            self.last_result = solution
            return solution
            
        except Exception as e:
            self.convergence_info = {
                'success': False,
                'message': f"ODE solver failed: {str(e)}",
                'error': e
            }
            raise
    
    def solve_pbbm_system(
        self,
        pbbm_func: Callable,
        initial_state: np.ndarray,
        time_points: np.ndarray,
        model_params: Dict[str, Any],
        **solver_options
    ) -> Any:
        """Solve PBBM ODE system.
        
        Args:
            pbbm_func: PBBM ODE function
            initial_state: Initial drug amounts/concentrations
            time_points: Time points for solution output
            model_params: Model parameters for PBBM
            **solver_options: Additional solver options
            
        Returns:
            Solution with time courses for all compartments
        """
        # Create wrapped ODE function
        def wrapped_ode(t, y):
            return pbbm_func(t, y, **model_params)
        
        # Solve
        t_span = (time_points[0], time_points[-1])
        return self.solve_ode(wrapped_ode, initial_state, t_span, time_points, **solver_options)
    
    def solve_pk_system(
        self,
        pk_func: Callable,
        initial_amounts: np.ndarray,
        time_points: np.ndarray,
        pk_params: Any,
        input_function: Optional[Callable] = None,
        **solver_options
    ) -> Any:
        """Solve PK ODE system.
        
        Args:
            pk_func: PK ODE function
            initial_amounts: Initial amounts in PK compartments
            time_points: Time points for solution output  
            pk_params: PK model parameters
            input_function: Function providing systemic input rate vs time
            **solver_options: Additional solver options
            
        Returns:
            Solution with PK compartment time courses
        """
        # Create wrapped ODE function
        def wrapped_ode(t, y):
            systemic_input = input_function(t) if input_function else 0.0
            return pk_func(y, pk_params, systemic_input)
        
        # Solve
        t_span = (time_points[0], time_points[-1])
        return self.solve_ode(wrapped_ode, initial_amounts, t_span, time_points, **solver_options)
    
    def solve_chained_system(
        self,
        chained_func: Callable,
        initial_state: np.ndarray,
        time_points: np.ndarray,
        all_params: Dict[str, Any],
        **solver_options
    ) -> Any:
        """Solve chained PBBM + PK system.
        
        Args:
            chained_func: Chained model ODE function
            initial_state: Combined initial state vector
            time_points: Time points for solution output
            all_params: All model parameters (lung, GI, PK)
            **solver_options: Additional solver options
            
        Returns:
            Solution for complete chained system
        """
        # Create wrapped ODE function  
        def wrapped_ode(t, y):
            return chained_func(t, y, **all_params)
        
        # Solve with appropriate settings for larger systems
        default_chained_options = {
            'rtol': 1e-7,
            'atol': 1e-10,
            'max_step': 3600.0  # 1-hour max step for stability
        }
        
        chained_options = default_chained_options.copy()
        chained_options.update(solver_options)
        
        t_span = (time_points[0], time_points[-1])
        return self.solve_ode(wrapped_ode, initial_state, t_span, time_points, **chained_options)


def solve_ode_system(
    ode_func: Callable,
    y0: np.ndarray,
    t_span: Tuple[float, float],
    t_eval: Optional[np.ndarray] = None,
    method: str = "BDF",
    **options
) -> Any:
    """Convenience function for solving ODE systems.
    
    Args:
        ode_func: ODE function dy/dt = f(t, y)
        y0: Initial conditions
        t_span: Integration time span (t_start, t_end)
        t_eval: Time points to evaluate solution at
        method: Integration method ("BDF", "RK45", "LSODA", etc.)
        **options: Solver options
        
    Returns:
        scipy solve_ivp result
    """
    solver = ODESolver(method=method, **options)
    return solver.solve_ode(ode_func, y0, t_span, t_eval)


def create_fallback_solution(
    time_points: np.ndarray,
    n_states: int,
    initial_state: Optional[np.ndarray] = None
) -> Dict[str, Any]:
    """Create a fallback solution when ODE solver fails.
    
    Args:
        time_points: Time points
        n_states: Number of state variables
        initial_state: Initial state (optional)
        
    Returns:
        Dictionary with fallback solution data
    """
    if initial_state is None:
        initial_state = np.zeros(n_states)
    
    # Simple exponential decay fallback
    y_fallback = np.zeros((n_states, len(time_points)))
    for i in range(n_states):
        y_fallback[i, :] = initial_state[i] * np.exp(-time_points / 3600.0)  # 1-hour decay
    
    return {
        't': time_points,
        'y': y_fallback,
        'success': False,
        'message': 'Fallback solution - ODE solver failed',
        'nfev': 0
    }