"""Base class for efficacy models."""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Dict, Any, Set
import numpy as np

from ...contracts.stage import Stage
from ...contracts.types import EfficacyInput, EfficacyResult


class EfficacyModel(Stage[EfficacyInput, EfficacyResult], ABC):
    """Base class for efficacy models.
    
    Efficacy models predict clinical endpoints (e.g., FEV1, symptoms)
    based on drug exposure and patient characteristics.
    """
    
    @property
    def provides(self) -> Set[str]:
        """Efficacy models provide efficacy predictions."""
        return {"efficacy"}
    
    @property  
    def requires(self) -> Set[str]:
        """Efficacy models typically require PK data."""
        return {"pk"}
    
    @abstractmethod
    def predict_efficacy(self,
                        exposure_data: Dict[str, np.ndarray],
                        patient_data: Dict[str, Any]) -> Dict[str, Any]:
        """Predict efficacy outcomes from drug exposure.
        
        Args:
            exposure_data: Drug exposure metrics (AUC, Cmax, etc.)
            patient_data: Patient characteristics
            
        Returns:
            Efficacy predictions
        """
        pass
    
    @abstractmethod
    def run(self, data: EfficacyInput) -> EfficacyResult:
        """Run efficacy model simulation."""
        pass