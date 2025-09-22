"""Integration-style tests for the Typer CLI with patched dependencies."""

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd
import pytest
from typer.testing import CliRunner

from lmp_pkg.cli.main import app
from lmp_pkg.config import AppConfig


@pytest.fixture
def runner():
    return CliRunner()


class TestCLIBasics:
    def test_cli_help(self, runner):
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "Lung Modeling Platform" in result.stdout

    def test_info_command(self, runner):
        with patch("lmp_pkg.cli.main.app_api.list_available_models", return_value={"cfd": []}):
            result = runner.invoke(app, ["info"])
        assert result.exit_code == 0
        assert "LMP Package" in result.stdout

    def test_list_models_command(self, runner):
        fake_models = {
            "cfd": ["ml"],
            "deposition": ["clean_lung"],
            "lung_pbbm": ["numba"],
            "systemic_pk": ["pk_3c"],
        }
        with patch("lmp_pkg.cli.main.app_api.list_available_models", return_value=fake_models):
            result = runner.invoke(app, ["list-models"])
        assert result.exit_code == 0
        assert "DEPOSITION Models" in result.stdout
        assert "numba" in result.stdout

    def test_list_catalog_command(self, runner):
        with patch("lmp_pkg.cli.main.app_api.list_catalog_entries", return_value=["healthy_reference"]):
            result = runner.invoke(app, ["list-catalog", "subject"])
        assert result.exit_code == 0
        assert "Catalog: SUBJECT" in result.stdout
        assert "healthy_reference" in result.stdout

    def test_list_catalog_invalid_category(self, runner):
        with patch("lmp_pkg.cli.main.app_api.list_catalog_entries", side_effect=ValueError("Unknown category: invalid")):
            result = runner.invoke(app, ["list-catalog", "invalid"])
        assert result.exit_code == 1
        assert "Unknown category" in result.stdout


class TestConfigValidation:
    def _patch_load_and_validate(self, config_obj):
        return patch.multiple(
            "lmp_pkg.cli.main.app_api",
            load_config_from_file=patch.DEFAULT,
            validate_configuration=patch.DEFAULT,
        )

    def test_validate_valid_config(self, runner, sample_toml_config):
        cfg = AppConfig()
        with patch("lmp_pkg.cli.main.app_api.load_config_from_file", return_value=cfg), \
             patch("lmp_pkg.cli.main.app_api.validate_configuration"):
            result = runner.invoke(app, ["validate", str(sample_toml_config)])
        assert result.exit_code == 0
        assert "valid" in result.stdout

    def test_validate_nonexistent_config(self, runner):
        result = runner.invoke(app, ["validate", "nonexistent.toml"])
        assert result.exit_code == 1
        assert "not found" in result.stdout


class TestSimulationCommands:
    def _stub_run_result(self) -> SimpleNamespace:
        return SimpleNamespace(
            run_id="stub_run",
            runtime_seconds=0.1,
            config={},
        )

    def test_run_dry_run(self, runner, sample_toml_config):
        cfg = AppConfig()
        with patch("lmp_pkg.cli.main.app_api.load_config_from_file", return_value=cfg), \
             patch("lmp_pkg.cli.main.app_api.validate_configuration"):
            result = runner.invoke(app, ["run", "--config", str(sample_toml_config), "--dry-run"])
        assert result.exit_code == 0
        assert "Dry run completed" in result.stdout

    def test_run_with_config(self, runner, sample_toml_config):
        cfg = AppConfig()
        metrics = {"auc": 1.23}
        with patch("lmp_pkg.cli.main.app_api.load_config_from_file", return_value=cfg), \
             patch("lmp_pkg.cli.main.app_api.validate_configuration"), \
             patch("lmp_pkg.cli.main.app_api.run_single_simulation", return_value=self._stub_run_result()), \
             patch("lmp_pkg.cli.main.app_api.calculate_summary_metrics", return_value=metrics):
            result = runner.invoke(app, ["run", "--config", str(sample_toml_config)])
        assert result.exit_code == 0
        assert "Simulation completed" in result.stdout

    def test_run_with_overrides(self, runner, sample_toml_config):
        cfg = AppConfig()
        with patch("lmp_pkg.cli.main.app_api.load_config_from_file", return_value=cfg), \
             patch("lmp_pkg.cli.main.app_api.validate_configuration"), \
             patch("lmp_pkg.cli.main.app_api.run_single_simulation", return_value=self._stub_run_result()), \
             patch("lmp_pkg.cli.main.app_api.calculate_summary_metrics", return_value={}):
            result = runner.invoke(app, [
                "run",
                "--config", str(sample_toml_config),
                "--set", json.dumps({"pk.model": "pk_2c"})
            ])
        assert result.exit_code == 0
        assert "Parameter overrides" in result.stdout

    def test_run_with_stages_override(self, runner, sample_toml_config):
        cfg = AppConfig()
        with patch("lmp_pkg.cli.main.app_api.load_config_from_file", return_value=cfg), \
             patch("lmp_pkg.cli.main.app_api.validate_configuration"), \
             patch("lmp_pkg.cli.main.app_api.run_single_simulation", return_value=self._stub_run_result()), \
             patch("lmp_pkg.cli.main.app_api.calculate_summary_metrics", return_value={}):
            result = runner.invoke(app, [
                "run",
                "--config", str(sample_toml_config),
                "--stages", "deposition,pbbm"
            ])
        assert result.exit_code == 0
        assert "Stages override" in result.stdout

    def test_run_with_invalid_json_overrides(self, runner, sample_toml_config):
        cfg = AppConfig()
        with patch("lmp_pkg.cli.main.app_api.load_config_from_file", return_value=cfg), \
             patch("lmp_pkg.cli.main.app_api.validate_configuration"):
            result = runner.invoke(app, ["run", "--config", str(sample_toml_config), "--set", "invalid json"])
        assert result.exit_code == 1
        assert "Invalid JSON" in result.stdout


class TestPlanningCommands:
    def test_plan_basic(self, runner, sample_toml_config):
        cfg = AppConfig()
        manifest = pd.DataFrame([{"run_id": "run_1"}])
        with patch("lmp_pkg.cli.main.app_api.load_config_from_file", return_value=cfg), \
             patch("lmp_pkg.cli.main.app_api.plan_simulation_manifest", return_value=manifest):
            result = runner.invoke(app, ["plan", str(sample_toml_config)])
        assert result.exit_code == 0
        assert "Generated manifest with 1 runs" in result.stdout

    def test_plan_with_axes(self, runner, sample_toml_config):
        cfg = AppConfig()
        manifest = pd.DataFrame([
            {"run_id": "run_1", "pk.model": "pk_1c"},
            {"run_id": "run_2", "pk.model": "pk_2c"},
        ])
        with patch("lmp_pkg.cli.main.app_api.load_config_from_file", return_value=cfg), \
             patch("lmp_pkg.cli.main.app_api.plan_simulation_manifest", return_value=manifest):
            result = runner.invoke(app, ["plan", str(sample_toml_config), "--axes", json.dumps({"pk.model": ["pk_1c", "pk_2c"]})])
        assert result.exit_code == 0
        assert "Generated manifest with 2 runs" in result.stdout

    def test_plan_with_output_file(self, runner, sample_toml_config, temp_dir):
        cfg = AppConfig()
        manifest = pd.DataFrame([{"run_id": "run_1"}])
        with patch("lmp_pkg.cli.main.app_api.load_config_from_file", return_value=cfg), \
             patch("lmp_pkg.cli.main.app_api.plan_simulation_manifest", return_value=manifest):
            output_file = temp_dir / "manifest.csv"
            result = runner.invoke(app, ["plan", str(sample_toml_config), "--output", str(output_file)])
        assert result.exit_code == 0
        assert output_file.exists()

    def test_plan_with_invalid_json_axes(self, runner, sample_toml_config):
        cfg = AppConfig()
        with patch("lmp_pkg.cli.main.app_api.load_config_from_file", return_value=cfg), \
             patch("lmp_pkg.cli.main.app_api.plan_simulation_manifest", return_value=pd.DataFrame()):
            result = runner.invoke(app, ["plan", str(sample_toml_config), "--axes", "invalid json"])
        assert result.exit_code == 1
        assert "Invalid JSON" in result.stdout
