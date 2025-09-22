"""Base class for deposition models."""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Set

from ...contracts.stage import Stage
from ...contracts.types import DepositionInput, DepositionResult


class DepositionModel(Stage[DepositionInput, DepositionResult], ABC):
    """Base class for deposition models.
    
    All deposition models must implement the Stage protocol and provide
    deposition capabilities.
    """
    
    @property
    def provides(self) -> Set[str]:
        """Deposition models provide deposition capability."""
        return {"deposition"}
    
    @property  
    def requires(self) -> Set[str]:
        """Deposition models don't require upstream capabilities."""
        return set()
    
    @abstractmethod
    def run(self, data: DepositionInput) -> DepositionResult:
        """Run deposition calculation.
        
        Args:
            data: Deposition input containing subject, product, maneuver
            
        Returns:
            Deposition result with regional amounts and metadata
        """
        pass