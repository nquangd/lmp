"""Lightweight tests for the public app_api facade using stubs."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from lmp_pkg import app_api
from lmp_pkg.config import AppConfig
from lmp_pkg.contracts import RunResult


class TestAppAPI:
    def test_get_default_config(self):
        cfg = app_api.get_default_config()
        assert isinstance(cfg, AppConfig)
        assert cfg.run.stages

    def test_load_config_from_file(self, tmp_path):
        dummy = AppConfig()
        with patch("lmp_pkg.app_api.load_config", return_value=dummy) as load_mock, \
             patch("lmp_pkg.app_api.validate_config") as validate_mock:
            cfg = app_api.load_config_from_file(tmp_path / "config.toml")
        load_mock.assert_called_once()
        validate_mock.assert_called_once_with(dummy)
        assert cfg is dummy

    def test_validate_configuration(self):
        cfg = AppConfig()
        with patch("lmp_pkg.app_api.validate_config") as validate_mock:
            app_api.validate_configuration(cfg)
        validate_mock.assert_called_once_with(cfg)

    def test_list_available_models(self):
        registry = MagicMock()
        registry.list_models.return_value = {"cfd": ["ml"]}
        with patch("lmp_pkg.app_api.get_registry", return_value=registry):
            models = app_api.list_available_models()
        assert models == {"cfd": ["ml"]}

    def test_list_catalog_entries(self):
        catalog = MagicMock()
        catalog.list_entries.return_value = ["entry"]
        with patch("lmp_pkg.app_api.get_default_catalog", return_value=catalog):
            entries = app_api.list_catalog_entries("subject")
        assert entries == ["entry"]
        catalog.list_entries.assert_called_once_with("subject")

    def test_get_catalog_entry(self):
        entity = MagicMock()
        entity.model_dump.return_value = {"name": "entry"}
        catalog = MagicMock()
        catalog.get_entry.return_value = entity
        with patch("lmp_pkg.app_api.get_default_catalog", return_value=catalog):
            data = app_api.get_catalog_entry("subject", "entry")
        assert data["name"] == "entry"


class TestManifestPlanning:
    def test_plan_simulation_manifest_empty(self):
        cfg = AppConfig()
        manifest = app_api.plan_simulation_manifest(cfg, {})
        assert len(manifest) == 1
        assert "run_id" in manifest.columns

    def test_plan_simulation_manifest_axes(self):
        cfg = AppConfig()
        axes = {"pk.model": ["pk_1c", "pk_2c"]}
        manifest = app_api.plan_simulation_manifest(cfg, axes)
        assert len(manifest) == 2
        assert set(manifest["pk.model"]) == {"pk_1c", "pk_2c"}


class TestResultProcessing:
    def test_convert_results_to_dataframes(self):
        result = RunResult(run_id="run_1", config={}, runtime_seconds=0.1, metadata={"entities": {}})
        frames = app_api.convert_results_to_dataframes(result)
        assert isinstance(frames, dict)
        for df in frames.values():
            assert isinstance(df, pd.DataFrame)

    def test_calculate_summary_metrics(self):
        result = RunResult(run_id="run_1", config={})
        metrics = app_api.calculate_summary_metrics(result)
        assert isinstance(metrics, dict)

    def test_try_load_cached_result_missing(self, temp_dir):
        loaded = app_api.try_load_cached_result(temp_dir, "missing")
        assert loaded is None


class TestMetricUtilities:
    def test_make_pk_overlay_data(self):
        result = RunResult(
            run_id="run",
            config={},
            pk=SimpleNamespace(t=np.array([0.0, 3600.0]), conc_plasma=np.array([0.0, 10.0])),
        )
        dataset = app_api.ObservedPK(time_s=np.array([0.0, 3600.0]), concentration_ng_per_ml=np.array([0.0, 9.5]))
        frame = app_api.pk_overlay_dataframe(result, dataset)
        assert isinstance(frame, pd.DataFrame)
        assert set(frame['series']) == {'predicted', 'observed'}

    def test_extract_population_variability_settings(self):
        cfg = AppConfig()
        overrides = app_api._extract_population_variability_settings(cfg)
        assert isinstance(overrides, dict)

