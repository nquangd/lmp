"""Variance-based sensitivity analysis built on the stage pipeline."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Callable, Sequence, Dict, Any, Tuple

import numpy as np

from .. import app_api
from ..config.model import AppConfig
from ..contracts.types import RunResult
from ..solver.optimization import ParameterDefinition, _vector_to_overrides


@dataclass
class SensitivityParameter(ParameterDefinition):
    """Parameter definition augmented with a sampling routine."""

    sampler: Callable[[int], np.ndarray] = None

    def sample(self, n: int) -> np.ndarray:
        if self.sampler is None:
            low, high = self.bounds
            return np.random.uniform(low, high, size=n)
        return self.sampler(n)


class PipelineSobolAnalyzer:
    """Compute Sobol sensitivity indices by running the LMP pipeline."""

    def __init__(
        self,
        base_config: AppConfig,
        parameters: Sequence[SensitivityParameter],
        metric: Callable[[RunResult], np.ndarray],
    ) -> None:
        self.base_config = base_config
        self.parameters = list(parameters)
        self.metric = metric

    def _simulate(self, vector: np.ndarray) -> np.ndarray:
        overrides = _vector_to_overrides(self.parameters, vector)
        result = app_api.run_single_simulation(copy.deepcopy(self.base_config), parameter_overrides=overrides)
        output = self.metric(result)
        return np.atleast_1d(np.asarray(output, dtype=float))

    def _sample_matrix(self, n: int) -> np.ndarray:
        samples = [p.sample(n) for p in self.parameters]
        return np.column_stack(samples)

    def _saltelli(self, n: int) -> Tuple[np.ndarray, np.ndarray, Sequence[np.ndarray]]:
        A = self._sample_matrix(n)
        B = self._sample_matrix(n)
        Ci = []
        for i in range(len(self.parameters)):
            Ci_matrix = A.copy()
            Ci_matrix[:, i] = B[:, i]
            Ci.append(Ci_matrix)
        return A, B, Ci

    def compute_indices(self, n_samples: int = 1000) -> Dict[str, Dict[str, Dict[str, float]]]:
        A, B, Ci_matrices = self._saltelli(n_samples)

        Y_A = np.array([self._simulate(row) for row in A])
        Y_B = np.array([self._simulate(row) for row in B])
        Y_Ci = [np.array([self._simulate(row) for row in Ci]) for Ci in Ci_matrices]

        n_outputs = Y_A.shape[1]
        results: Dict[str, Dict[str, Dict[str, float]]] = {}

        for output_idx in range(n_outputs):
            yA = Y_A[:, output_idx]
            yB = Y_B[:, output_idx]
            yCi = [Y_Ci[i][:, output_idx] for i in range(len(self.parameters))]

            total_variance = np.var(np.concatenate([yA, yB]), ddof=0)
            if total_variance == 0:
                total_variance = 1e-12

            first_order = {}
            total_order = {}

            for param, yc in zip(self.parameters, yCi):
                numerator_first = np.mean(yB * (yc - yA))
                numerator_total = 0.5 * np.mean((yA - yc) ** 2)
                first_order[param.name] = float(numerator_first / total_variance)
                total_order[param.name] = float(numerator_total / total_variance)

            results[f'output_{output_idx}'] = {
                'first_order': first_order,
                'total_order': total_order,
                'variance': float(total_variance),
            }

        return results
