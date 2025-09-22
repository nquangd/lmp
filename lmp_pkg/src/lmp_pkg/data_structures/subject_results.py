"""Subject-scoped results containers and builders.

These classes attach computed results to a Subject instance for convenient
access, with unit-aware properties powered by UnitContext.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Dict
import numpy as np

from .units import UnitContext


@dataclass
class Results_Deposition:
    """Per-subject deposition results wrapper.

    Stores initial ELF amounts per region in pmol and exposes unit-aware
    views via UnitContext.
    """
    region_ids: np.ndarray
    region_names: Optional[np.ndarray]
    elf_initial_amounts_pmol: np.ndarray
    units: UnitContext

    @classmethod
    def from_deposition_result(cls, deposition_result, molecular_weight_ug_per_umol: float) -> "Results_Deposition":
        names = None
        if deposition_result.metadata and 'regional_names' in deposition_result.metadata:
            names = np.array(deposition_result.metadata['regional_names'])
        return cls(
            region_ids=deposition_result.region_ids,
            region_names=names,
            elf_initial_amounts_pmol=deposition_result.elf_initial_amounts,
            units=UnitContext(molecular_weight_ug_per_umol=molecular_weight_ug_per_umol,
                              amount_unit='ng', concentration_unit='ng/mL')
        )

    def elf_initial_amounts(self) -> np.ndarray:
        """Amounts in the UnitContext's amount unit (default ng)."""
        return self.units.amount_from_pmol(self.elf_initial_amounts_pmol)

    def by_region(self) -> Dict[str, float]:
        if self.region_names is None:
            return {str(int(i)): v for i, v in zip(self.region_ids, self.elf_initial_amounts())}
        return {str(name): v for name, v in zip(self.region_names, self.elf_initial_amounts())}


@dataclass
class Results_PK:
    """Per-subject PK results wrapper with unit-aware access."""
    time_s: np.ndarray
    plasma_conc_pmol_per_ml: np.ndarray
    central_amounts_pmol: np.ndarray
    units: UnitContext

    @classmethod
    def from_orchestrator(cls, orchestrator_result, molecular_weight_ug_per_umol: float) -> "Results_PK":
        # Prefer pmol/mL if available; otherwise convert from ng/mL
        pk_block = orchestrator_result.results_data.get('pk', {})
        t_s = np.array(orchestrator_result.time_points)

        conc_ng_ml = pk_block.get('plasma_concentration_ng_ml')
        units = UnitContext(molecular_weight_ug_per_umol=molecular_weight_ug_per_umol,
                            concentration_unit='ng/mL', amount_unit='ng')
        if conc_ng_ml is not None:
            conc_pmol_ml = units.concentration_to_pmol_per_ml(np.array(conc_ng_ml), 'ng/mL')
        else:
            conc_pmol_ml = np.zeros_like(t_s)

        central = np.array(pk_block.get('central', np.zeros_like(t_s)))

        return cls(
            time_s=t_s,
            plasma_conc_pmol_per_ml=conc_pmol_ml,
            central_amounts_pmol=central,
            units=units
        )

    @property
    def time_h(self) -> np.ndarray:
        return self.time_s / 3600.0

    @property
    def plasma_concentration(self) -> np.ndarray:
        """Concentration in the UnitContext's concentration unit (default ng/mL)."""
        return self.units.concentration_from_pmol_per_ml(self.plasma_conc_pmol_per_ml)

    @property
    def central_amount(self) -> np.ndarray:
        """Amount in the UnitContext's amount unit (default ng)."""
        return self.units.amount_from_pmol(self.central_amounts_pmol)

