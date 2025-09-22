"""Null deposition model used for tests and fallback scenarios."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Set

import numpy as np

from .base import DepositionModel
from ...contracts.types import DepositionInput, DepositionResult


@dataclass
class NullDeposition(DepositionModel):
    """No-op deposition model that returns zero deposition."""

    name: str = "null"

    @property
    def provides(self) -> Set[str]:
        return {"deposition"}

    @property
    def requires(self) -> Set[str]:
        return set()

    def run(self, data: DepositionInput) -> DepositionResult:
        region_ids = np.array(["ET", "TB", "P1", "P2"], dtype=object)
        elf_initial_amounts = np.zeros(len(region_ids))
        return DepositionResult(
            region_ids=region_ids,
            elf_initial_amounts=elf_initial_amounts,
            metadata={
                "model": "null",
                "total_deposited_mass_mg": 0.0,
            },
        )
