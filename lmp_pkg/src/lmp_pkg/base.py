"""Base interfaces for the LMP package."""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Dict, Any


class DataStructure(ABC):
    """Base interface for PBBM data structures."""
    
    @abstractmethod
    def validate(self) -> bool:
        """Validate data structure consistency."""
        pass
    
    @abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        pass
    
    @abstractmethod
    def from_dict(self, data: Dict[str, Any]) -> 'DataStructure':
        """Create from dictionary data."""
        pass