"""Base classes for numerical solvers."""

from __future__ import annotations
from abc import ABC, abstractmethod
import numpy as np
from typing import Dict, Any, Callable, Optional, Tuple


class SolverBase(ABC):
    """Base class for all numerical solvers."""
    
    def __init__(self, name: str):
        self.name = name
        self.last_result: Optional[Any] = None
        self.convergence_info: Dict[str, Any] = {}
    
    @abstractmethod
    def solve(self, *args, **kwargs) -> Any:
        """Solve the numerical problem."""
        pass
    
    def get_convergence_info(self) -> Dict[str, Any]:
        """Get information about the last solve convergence."""
        return self.convergence_info.copy()


class ODESolverBase(SolverBase):
    """Base class for ODE solvers."""
    
    def __init__(self, name: str, method: str = "BDF"):
        super().__init__(name)
        self.method = method
        self.default_options = {
            'rtol': 1e-6,
            'atol': 1e-9,
            'max_step': np.inf,
            'first_step': None
        }
    
    @abstractmethod
    def solve_ode(
        self, 
        ode_func: Callable,
        y0: np.ndarray,
        t_span: Tuple[float, float],
        t_eval: Optional[np.ndarray] = None,
        **options
    ) -> Any:
        """Solve an ODE system."""
        pass


class OptimizationSolverBase(SolverBase):
    """Base class for optimization solvers."""
    
    def __init__(self, name: str, method: str = "L-BFGS-B"):
        super().__init__(name)
        self.method = method
        self.default_options = {
            'maxiter': 1000,
            'ftol': 1e-9,
            'gtol': 1e-6
        }
    
    @abstractmethod
    def optimize(
        self,
        objective: Callable,
        x0: np.ndarray,
        bounds: Optional[list] = None,
        **options
    ) -> Any:
        """Solve an optimization problem."""
        pass