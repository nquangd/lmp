"""Tests for catalog utilities with updated entity schema."""

from pathlib import Path
import pytest

from lmp_pkg.catalog import ReadOnlyCatalog, get_default_catalog
from lmp_pkg.catalog.loaders import TomlReader
from lmp_pkg.domain.entities import Demographic, Subject, API


class TestCatalogReaders:
    def test_toml_reader(self, temp_dir):
        reader = TomlReader()
        toml_file = temp_dir / "entry.toml"
        toml_file.write_text("name='entry'\nvalue=1\n")
        assert reader.can_read(toml_file)
        data = reader.read(toml_file)
        assert data["name"] == "entry"


class TestReadOnlyCatalog:
    @pytest.fixture
    def sample_catalog_dir(self, temp_dir):
        root = temp_dir / "catalog"
        (root / "subject").mkdir(parents=True)
        (root / "api").mkdir(parents=True)

        (root / "subject" / "adult.toml").write_text(
            """
name = "adult"
subject_id = "adult"
age_years = 30.0
weight_kg = 70.0
height_cm = 175.0
sex = "M"
"""
        )

        (root / "api" / "drug.toml").write_text(
            """
name = "drug"
molecular_weight = 250.0
"""
        )
        return root

    def test_list_entries(self, sample_catalog_dir):
        catalog = ReadOnlyCatalog([sample_catalog_dir])
        subjects = catalog.list_entries("subject")
        assert subjects == ["adult"]

    def test_get_entry_subject(self, sample_catalog_dir):
        catalog = ReadOnlyCatalog([sample_catalog_dir])
        subject = catalog.get_entry("subject", "adult")
        assert isinstance(subject, Subject)
        assert subject.name == "adult"

    def test_get_entry_api(self, sample_catalog_dir):
        catalog = ReadOnlyCatalog([sample_catalog_dir])
        api = catalog.get_entry("api", "drug")
        assert isinstance(api, API)
        assert api.molecular_weight == 250.0

    def test_has_entry(self, sample_catalog_dir):
        catalog = ReadOnlyCatalog([sample_catalog_dir])
        assert catalog.has_entry("subject", "adult")
        assert not catalog.has_entry("subject", "missing")

    def test_unknown_category(self, sample_catalog_dir):
        catalog = ReadOnlyCatalog([sample_catalog_dir])
        with pytest.raises(ValueError):
            catalog.list_entries("unknown")


class TestDefaultCatalog:
    def test_get_default_catalog(self):
        catalog = get_default_catalog()
        assert isinstance(catalog, ReadOnlyCatalog)
