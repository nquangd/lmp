"""Tests for configuration system."""

import pytest
from pathlib import Path

from lmp_pkg.config import AppConfig, load_config, default_config, validate_config
from lmp_pkg.contracts.errors import ConfigError, ValidationError


class TestAppConfig:
    """Test configuration model."""
    
    def test_default_config(self):
        """Test default configuration creation."""
        config = AppConfig()
        
        assert config.run.stages == ["cfd", "deposition", "pbbm", "pk"]
        assert config.run.seed == 123
        assert config.run.threads == 1
        assert not config.run.enable_numba
        
        assert config.deposition.model == "null"
        assert config.pbbm.model == "null" 
        assert config.pk.model == "null"
        
        assert config.subject.ref == "adult_70kg"
        assert config.api.ref == "salbutamol"
    
    def test_validation_valid_stages(self):
        """Test stage validation with valid stages."""
        config = AppConfig()
        config.run.stages = ["cfd", "deposition", "pbbm"]
        
        # Should not raise
        config.model_validate(config.model_dump())
    
    def test_validation_invalid_stages(self):
        """Test stage validation with invalid stages."""
        with pytest.raises(ValueError, match="Invalid stages"):
            AppConfig(run={"stages": ["invalid_stage"]})
    
    def test_solver_config_validation(self):
        """Test solver configuration validation.""" 
        with pytest.raises(ValueError, match="method must be one of"):
            AppConfig(pbbm={"solver": {"method": "INVALID"}})
    
    def test_threads_validation(self):
        """Test threads validation."""
        with pytest.raises(ValueError, match="threads must be positive"):
            AppConfig(run={"threads": 0})


class TestConfigLoading:
    """Test configuration loading."""
    
    def test_load_default_config(self):
        """Test loading default configuration.""" 
        config = default_config()
        assert isinstance(config, AppConfig)
        assert config.run.stages == ["cfd", "deposition", "pbbm", "pk"]
    
    def test_load_from_toml_file(self, sample_toml_config: Path):
        """Test loading from TOML file."""
        config = AppConfig.from_toml_file(sample_toml_config)
        
        assert config.run.seed == 42
        assert config.run.threads == 2
        assert config.pbbm.solver.method == "BDF"
        assert config.pbbm.solver.rtol == 1e-6
    
    def test_load_nonexistent_file(self):
        """Test loading from non-existent file."""
        with pytest.raises(ConfigError, match="Configuration file not found"):
            load_config("nonexistent.toml")
    
    def test_load_invalid_toml(self, temp_dir: Path):
        """Test loading invalid TOML file."""
        bad_config = temp_dir / "bad.toml"
        bad_config.write_text("invalid toml content [[[")
        
        with pytest.raises(ConfigError, match="Failed to load config"):
            load_config(bad_config)


class TestConfigValidation:
    """Test configuration validation."""
    
    def test_valid_config(self, sample_config: AppConfig):
        """Test validation of valid configuration."""
        # Should not raise
        validate_config(sample_config)
    
    def test_pbbm_without_deposition_error(self):
        """Test error when PBBM requires deposition but it's missing."""
        config = AppConfig()
        config.run.stages = ["pbbm"]  # Missing deposition
        config.pbbm.model = "classic_pbbm"  # Non-null model
        
        with pytest.raises(ValidationError, match="PBBM stage requires deposition"):
            validate_config(config)
    
    def test_pk_without_pbbm_error(self):
        """Test error when PK requires PBBM but it's missing."""
        config = AppConfig()
        config.run.stages = ["cfd", "deposition", "pk"]  # Missing PBBM
        config.pk.model = "pk_2c"  # Non-null model
        
        with pytest.raises(ValidationError, match="PK stage typically requires PBBM"):
            validate_config(config)
    
    def test_loose_tolerances_warning(self, sample_config: AppConfig):
        """Test warning for loose solver tolerances."""
        sample_config.pbbm.solver.rtol = 1e-2  # Very loose
        
        # Should not raise, but should log warning
        validate_config(sample_config)
    
    def test_high_thread_count_warning(self, sample_config: AppConfig):
        """Test warning for high thread count."""
        sample_config.run.threads = 32  # Very high
        
        # Should not raise, but should log warning
        validate_config(sample_config)
