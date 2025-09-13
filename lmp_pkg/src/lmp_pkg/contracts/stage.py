"""Stage protocol definition."""

from typing import Generic, TypeVar, Protocol, Set, runtime_checkable

TIn = TypeVar("TIn")
TOut = TypeVar("TOut")


@runtime_checkable
class Stage(Protocol, Generic[TIn, TOut]):
    """Protocol for pipeline stages.
    
    Stages are the core building blocks of the LMP pipeline. Each stage:
    - Has a unique name for identification
    - Declares what capabilities it provides  
    - Declares what capabilities it requires from upstream stages
    - Implements a run method that transforms input to output
    """
    
    name: str
    """Unique identifier for this stage implementation."""
    
    provides: Set[str]
    """Capability tags this stage provides.
    
    Examples:
    - {"deposition"} for deposition models
    - {"pbbm"} for lung PBPK models  
    - {"pk"} for systemic PK models
    - {"analysis:bioequivalence"} for specific analysis types
    """
    
    requires: Set[str]
    """Capability tags required from upstream stages.
    
    Examples:
    - set() for stages that can run independently
    - {"deposition"} for stages requiring deposition output
    - {"pbbm", "pk"} for analysis stages requiring both models
    """
    
    def run(self, data: TIn) -> TOut:
        """Execute this stage with the given input data.
        
        Args:
            data: Input data matching the stage's expected input type
            
        Returns:
            Output data in the stage's declared output type
            
        Raises:
            ModelError: If stage execution fails
            ValidationError: If input data is invalid
        """
        ...