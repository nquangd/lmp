"""Pytest configuration and fixtures."""

import sys
import tempfile
from pathlib import Path
from typing import Dict, Any

SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import pytest

from lmp_pkg.config import AppConfig


@pytest.fixture
def temp_dir():
    """Temporary directory for test artifacts."""
    with tempfile.TemporaryDirectory() as tmp:
        yield Path(tmp)


@pytest.fixture  
def sample_config() -> AppConfig:
    """Sample configuration for testing."""
    return AppConfig()


@pytest.fixture
def sample_config_dict() -> Dict[str, Any]:
    """Sample configuration as dictionary."""
    return {
        "run": {
            "stages": ["cfd", "deposition", "pbbm", "pk"],
            "seed": 42,
            "threads": 1,
            "artifact_dir": "test_results"
        },
        "deposition": {
            "model": "null"
        },
        "pbbm": {
            "model": "null", 
            "solver": {
                "method": "BDF",
                "rtol": 1e-6,
                "atol": 1e-9
            }
        },
        "pk": {
            "model": "null"
        },
        "subject": {
            "ref": "adult_70kg"
        },
        "api": {
            "ref": "salbutamol"
        }
    }


@pytest.fixture
def sample_toml_config(temp_dir: Path) -> Path:
    """Sample TOML configuration file."""
    config_content = """
[run]
stages = ["cfd", "deposition", "pbbm", "pk"]
seed = 42
threads = 2
artifact_dir = "test_results"

[deposition]
model = "null"

[pbbm]
model = "null"

[pbbm.solver]
method = "BDF"
rtol = 1e-6
atol = 1e-9

[pk]
model = "null"

[subject]
ref = "adult_70kg"

[api]
ref = "salbutamol"

[product] 
ref = "HFA_MDI_PT210"

[maneuver]
ref = "standard_MDI_inhale_hold"
"""
    
    config_file = temp_dir / "test_config.toml"
    config_file.write_text(config_content)
    return config_file
