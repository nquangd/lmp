"""Configuration loading utilities."""

from __future__ import annotations
import os
from pathlib import Path
from typing import Dict, Any, Union, Optional

from ..contracts.errors import ConfigError
from .model import AppConfig, PopulationVariabilityConfig


def default_config() -> AppConfig:
    """Create default configuration."""
    cfg = AppConfig()
    cfg.population_variability = PopulationVariabilityConfig(
        demographic=False,
        lung_regional=False,
        lung_generation=False,
        gi=False,
        pk=False,
        inhalation=False,
    )
    return cfg


def load_config(path: Optional[Union[str, Path]] = None) -> AppConfig:
    """Load configuration from file or environment.
    
    Args:
        path: Path to configuration file. If None, looks for:
              - LMP_CONFIG environment variable
              - lmp.toml in current directory
              - ~/.lmp/config.toml
              
    Returns:
        Loaded and validated configuration
        
    Raises:
        ConfigError: If configuration file is invalid or not found
    """
    if path is None:
        path = _find_config_file()
        
    if path is None:
        return default_config()
        
    path = Path(path)
    if not path.exists():
        raise ConfigError(f"Configuration file not found: {path}")
        
    try:
        config = AppConfig.from_toml_file(path)
        
        # Apply environment variable overrides
        config = _apply_env_overrides(config)
        
        return config
        
    except Exception as e:
        raise ConfigError(f"Failed to load config from {path}: {e}")


def _find_config_file() -> Optional[Path]:
    """Find configuration file using standard search paths."""
    
    # 1. Environment variable
    env_path = os.environ.get("LMP_CONFIG")
    if env_path:
        return Path(env_path)
    
    # 2. Current directory
    cwd_config = Path("lmp.toml")
    if cwd_config.exists():
        return cwd_config
        
    # 3. User config directory
    user_config = Path.home() / ".lmp" / "config.toml"
    if user_config.exists():
        return user_config
        
    return None


def _apply_env_overrides(config: AppConfig) -> AppConfig:
    """Apply environment variable overrides to configuration.
    
    Environment variables follow pattern: LMP_<SECTION>_<KEY>
    Examples:
        LMP_RUN_THREADS=4
        LMP_PBBM_MODEL=classic_pbbm
        LMP_RUN_ARTIFACT_DIR=/scratch/results
    """
    overrides: Dict[str, Any] = {}
    
    for key, value in os.environ.items():
        if not key.startswith("LMP_"):
            continue
            
        parts = key[4:].lower().split("_", 1)  # Remove LMP_ prefix
        if len(parts) != 2:
            continue
            
        section, field = parts
        
        # Convert string values to appropriate types
        typed_value = _convert_env_value(value)
        
        if section not in overrides:
            overrides[section] = {}
        overrides[section][field] = typed_value
    
    # Apply overrides
    if overrides:
        config_dict = config.model_dump()
        for section, fields in overrides.items():
            if section in config_dict:
                config_dict[section].update(fields)
                
        config = AppConfig.model_validate(config_dict)
    
    return config


def _convert_env_value(value: str) -> Any:
    """Convert string environment variable to appropriate type."""
    # Boolean
    if value.lower() in ("true", "false"):
        return value.lower() == "true"
    
    # Integer
    try:
        return int(value)
    except ValueError:
        pass
        
    # Float
    try:
        return float(value)
    except ValueError:
        pass
        
    # String (default)
    return value
