"""Unit handling for data structures with a shared context.

Provides a UnitContext that defines default output units and conversion
helpers between internal model units and desired presentation units.
Internal units used by data structures:
 - Amount: pmol
 - Concentration: pmol/mL
 - Time: seconds (derived hours provided where helpful)

Molecular weight is expected in μg/μmol (equivalent to g/mol).
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Literal, Optional
import numpy as np


ConcentrationUnit = Literal['pmol/mL', 'ng/mL', 'pg/mL']
AmountUnit = Literal['pmol', 'ng', 'pg', 'ug']


@dataclass
class UnitContext:
    """Controls default output units and performs conversions."""

    molecular_weight_ug_per_umol: float = 250.0
    concentration_unit: ConcentrationUnit = 'ng/mL'
    amount_unit: AmountUnit = 'ng'

    # ---- Amount conversions ----
    def amount_from_pmol(self, pmol: np.ndarray, unit: Optional[AmountUnit] = None) -> np.ndarray:
        unit = unit or self.amount_unit
        mw = float(self.molecular_weight_ug_per_umol)

        if unit == 'pmol':
            return pmol
        if unit == 'pg':
            return pmol * mw  # 1 pmol = MW pg
        if unit == 'ng':
            return pmol * (mw / 1000.0)  # 1 pmol = MW/1000 ng
        if unit == 'ug':
            return pmol * (mw / 1e6)  # 1 pmol = MW/1e6 μg
        raise ValueError(f"Unsupported amount unit: {unit}")

    # ---- Concentration conversions ----
    def concentration_from_pmol_per_ml(self, pmol_per_ml: np.ndarray, unit: Optional[ConcentrationUnit] = None) -> np.ndarray:
        unit = unit or self.concentration_unit
        mw = float(self.molecular_weight_ug_per_umol)

        if unit == 'pmol/mL':
            return pmol_per_ml
        if unit == 'pg/mL':
            return pmol_per_ml * mw  # 1 pmol/mL = MW pg/mL
        if unit == 'ng/mL':
            return pmol_per_ml * (mw / 1000.0)  # 1 pmol/mL = MW/1000 ng/mL
        raise ValueError(f"Unsupported concentration unit: {unit}")

    def concentration_to_pmol_per_ml(self, values: np.ndarray, unit: ConcentrationUnit) -> np.ndarray:
        mw = float(self.molecular_weight_ug_per_umol)
        if unit == 'pmol/mL':
            return values
        if unit == 'pg/mL':
            return values / mw
        if unit == 'ng/mL':
            return values * (1000.0 / mw)
        raise ValueError(f"Unsupported concentration unit: {unit}")

