"""Tests for contracts and type definitions."""

import pytest
import numpy as np
from typing import Set

from lmp_pkg.contracts import (
    LMPError, ConfigError, ModelError, ValidationError,
    Stage, DepositionInput, DepositionResult, PBBKInput, PBBKResult, 
    PKInput, PKResult, RunResult
)


class TestErrors:
    """Test error hierarchy."""
    
    def test_lmp_error_base(self):
        """Test base LMP error."""
        error = LMPError("Test message", {"key": "value"})
        
        assert str(error) == "Test message"
        assert error.message == "Test message"
        assert error.details == {"key": "value"}
    
    def test_lmp_error_no_details(self):
        """Test LMP error without details."""
        error = LMPError("Test message")
        
        assert str(error) == "Test message"
        assert error.message == "Test message"
        assert error.details == {}
    
    def test_config_error_inheritance(self):
        """Test ConfigError inherits from LMPError."""
        error = ConfigError("Config issue")
        
        assert isinstance(error, LMPError)
        assert str(error) == "Config issue"
    
    def test_validation_error_inheritance(self):
        """Test ValidationError inherits from ConfigError."""
        error = ValidationError("Validation failed")
        
        assert isinstance(error, ConfigError)
        assert isinstance(error, LMPError)


class MockStage:
    """Mock stage implementation for testing."""
    
    def __init__(self, name: str, provides: Set[str], requires: Set[str]):
        self.name = name
        self.provides = provides
        self.requires = requires
    
    def run(self, data):
        return f"processed_{data}"


class TestStageProtocol:
    """Test Stage protocol."""
    
    def test_stage_protocol_compliance(self):
        """Test that MockStage is compliant with Stage protocol."""
        stage = MockStage(
            name="test_stage",
            provides={"test_output"}, 
            requires={"test_input"}
        )
        
        # Should pass isinstance check
        assert isinstance(stage, Stage)
        
        assert stage.name == "test_stage"
        assert stage.provides == {"test_output"}
        assert stage.requires == {"test_input"}
        
        result = stage.run("input_data")
        assert result == "processed_input_data"
    
    def test_stage_without_requirements(self):
        """Test stage with no upstream requirements."""
        stage = MockStage(
            name="independent_stage",
            provides={"output"},
            requires=set()
        )
        
        assert isinstance(stage, Stage)
        assert len(stage.requires) == 0


class TestDataTypes:
    """Test data type definitions."""
    
    def test_deposition_input(self):
        """Test DepositionInput dataclass.""" 
        subject = {"weight": 70.0}
        product = {"device": "MDI"}
        maneuver = {"flow_rate": 30.0}
        
        input_data = DepositionInput(
            subject=subject,
            product=product, 
            maneuver=maneuver,
            particle_grid=np.array([0.1, 1.0, 10.0]),
            params={"param1": 1.0}
        )
        
        assert input_data.subject == subject
        assert input_data.product == product
        assert input_data.maneuver == maneuver
        assert np.array_equal(input_data.particle_grid, np.array([0.1, 1.0, 10.0]))
        assert input_data.params == {"param1": 1.0}
        
        # Test immutability
        assert hasattr(input_data, "__dataclass_fields__")
    
    def test_deposition_result(self):
        """Test DepositionResult dataclass."""
        regions = np.array([1, 2, 3])
        amounts = np.array([10.0, 20.0, 30.0])
        
        result = DepositionResult(
            region_ids=regions,
            elf_initial_amounts=amounts,
            metadata={"total_dose": 60.0}
        )
        
        assert np.array_equal(result.region_ids, regions)
        assert np.array_equal(result.elf_initial_amounts, amounts)
        assert result.metadata == {"total_dose": 60.0}
    
    def test_pbbk_input(self):
        """Test PBBKInput dataclass."""
        deposition_result = DepositionResult(
            region_ids=np.array([1]),
            elf_initial_amounts=np.array([10.0])
        )
        
        input_data = PBBKInput(
            subject={"weight": 70.0},
            api={"mw": 239.31},
            lung_seed=deposition_result,
            params={"permeability": 1e-6}
        )
        
        assert input_data.lung_seed == deposition_result
        assert input_data.params == {"permeability": 1e-6}
    
    def test_pbbk_result(self):
        """Test PBBKResult dataclass."""
        t = np.array([0.0, 1.0, 2.0])
        y = np.array([[1.0, 2.0], [0.8, 1.8], [0.6, 1.6]])
        slices = {"ELF": slice(0, 1), "tissue": slice(1, 2)}
        outflow = np.array([0.0, 0.2, 0.4])
        
        result = PBBKResult(
            t=t,
            y=y,
            region_slices=slices,
            pulmonary_outflow=outflow,
            metadata={"solver": "BDF"},
        )
        
        assert np.array_equal(result.t, t)
        assert np.array_equal(result.y, y)
        assert result.region_slices == slices
        assert np.array_equal(result.pulmonary_outflow, outflow)
        assert result.comprehensive is None
    
    def test_pk_input_and_result(self):
        """Test PKInput and PKResult dataclasses."""
        pulmonary_input = np.array([0.0, 0.2, 0.4])
        
        input_data = PKInput(
            subject={"weight": 70.0},
            api={"clearance": 20.0},
            pulmonary_input=pulmonary_input,
            params={"ka": 1.2}
        )
        
        assert np.array_equal(input_data.pulmonary_input, pulmonary_input)
        
        t = np.array([0.0, 1.0, 2.0])
        conc = np.array([0.0, 10.0, 5.0])
        compartments = {"central": conc, "peripheral": conc * 0.5}
        
        result = PKResult(
            t=t,
            conc_plasma=conc,
            compartments=compartments
        )
        
        assert np.array_equal(result.t, t)
        assert np.array_equal(result.conc_plasma, conc)
        assert "central" in result.compartments
    
    def test_run_result(self):
        """Test RunResult dataclass."""
        deposition = DepositionResult(
            region_ids=np.array([1]),
            elf_initial_amounts=np.array([10.0])
        )
        
        result = RunResult(
            run_id="test_123",
            config={"stages": ["deposition"]},
            deposition=deposition,
            runtime_seconds=1.5,
            metadata={"version": "0.1.0"}
        )
        
        assert result.run_id == "test_123"
        assert result.deposition == deposition
        assert result.runtime_seconds == 1.5
        assert result.metadata == {"version": "0.1.0"}
        
        # Test optional fields
        assert result.pbbk is None
        assert result.pk is None
        assert result.analysis is None
