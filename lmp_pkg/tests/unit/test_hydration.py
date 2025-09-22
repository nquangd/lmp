"""Tests for configuration hydration with the updated entity models."""

import warnings
from unittest.mock import Mock, patch

import pytest

from lmp_pkg.catalog import ReadOnlyCatalog
from lmp_pkg.config import AppConfig, EntityRef, hydrate_config, validate_hydrated_entities
from lmp_pkg.contracts.errors import ConfigError
from lmp_pkg.domain.entities import Demographic, Subject, API, Product, InhalationProfile


def _add_subject_aliases(subject: Subject) -> Subject:
    demo = subject.demographic
    if demo is not None:
        object.__setattr__(subject, "age_years", demo.age_years)
        object.__setattr__(subject, "weight_kg", demo.weight_kg)
        object.__setattr__(subject, "height_cm", demo.height_cm)
    return subject


def make_subject(**overrides) -> Subject:
    demo_defaults = dict(
        name="test_adult",
        subject_id="test_adult",
        age_years=30.0,
        weight_kg=70.0,
        height_cm=175.0,
        sex="M",
    )
    demo_defaults.update(overrides.pop("demographic", {}))
    demo = Demographic(**demo_defaults)
    subject = Subject(name=overrides.pop("name", "test_adult"), demographic=demo, **overrides)
    return _add_subject_aliases(subject)


def make_api(**overrides) -> API:
    base = dict(name="test_drug", molecular_weight=250.0)
    base.update(overrides)
    return API(**base)


def make_product(**overrides) -> Product:
    base = dict(
        name="test_inhaler",
        device="pMDI",
        propellant="HFA-134a",
        apis={"test_drug": {"dose_pg": 50000.0, "mmad": 3.2, "gsd": 1.6, "usp_depo_fraction": 38.0}},
    )
    base.update(overrides)
    product = Product(**base)
    object.__setattr__(product, "device_type", product.device or "pMDI")
    object.__setattr__(product, "formulation_type", overrides.get("formulation_type", "solution"))
    object.__setattr__(product, "label_claim_mg", overrides.get("label_claim_mg", 0.1))
    return product


def make_maneuver(**overrides) -> InhalationProfile:
    base = dict(
        name="test_maneuver",
        maneuver_type="slow_deep",
        pifr_Lpm=45.0,
        rise_time_s=1.0,
        inhaled_volume_L=1.5,
        hold_time_s=0.5,
        breath_hold_time_s=3.0,
        exhalation_flow_Lpm=30.0,
        bolus_volume_ml=0.0,
        bolus_delay_s=0.0,
    )
    base.update(overrides)
    maneuver = InhalationProfile(**base)
    object.__setattr__(maneuver, "peak_inspiratory_flow_l_min", maneuver.pifr_Lpm)
    object.__setattr__(maneuver, "inhaled_volume_ml", maneuver.inhaled_volume_L * 1000.0)
    object.__setattr__(maneuver, "inhalation_time_s", overrides.get("inhalation_time_s", 2.0))
    object.__setattr__(maneuver, "coordination_efficiency", overrides.get("coordination_efficiency", 0.9))
    return maneuver


class TestConfigHydration:
    @pytest.fixture
    def mock_catalog(self):
        catalog = Mock(spec=ReadOnlyCatalog)
        entities = {
            ("subject", "test_adult"): make_subject(),
            ("api", "test_drug"): make_api(),
            ("product", "test_inhaler"): make_product(),
            ("maneuver", "test_maneuver"): make_maneuver(),
        }
        catalog.get_entry.side_effect = lambda cat, name: entities[(cat, name)]
        return catalog

    def test_basic_hydration(self, mock_catalog):
        config = AppConfig(
            subject=EntityRef(ref="test_adult"),
            api=EntityRef(ref="test_drug"),
            product=EntityRef(ref="test_inhaler"),
            maneuver=EntityRef(ref="test_maneuver"),
        )

        hydrated = hydrate_config(config, mock_catalog)
        assert hydrated["subject"].demographic.name == "test_adult"
        assert hydrated["maneuver"].pifr_Lpm == 45.0

    def test_hydration_with_overrides(self, mock_catalog):
        demo_override = make_subject().demographic.model_dump()
        demo_override["weight_kg"] = 80.0
        config = AppConfig(
            subject=EntityRef(ref="test_adult", overrides={"demographic": demo_override}),
            api=EntityRef(ref="test_drug"),
            product=EntityRef(ref="test_inhaler"),
            maneuver=EntityRef(ref="test_maneuver"),
        )

        hydrated = hydrate_config(config, mock_catalog)
        assert hydrated["subject"].demographic.weight_kg == 80.0

    def test_hydration_missing_reference(self, mock_catalog):
        config = AppConfig(
            subject=EntityRef(),
            api=EntityRef(ref="test_drug"),
            product=EntityRef(ref="test_inhaler"),
            maneuver=EntityRef(ref="test_maneuver"),
        )

        with pytest.raises(ConfigError):
            hydrate_config(config, mock_catalog)

    def test_hydration_entity_not_found(self, mock_catalog):
        mock_catalog.get_entry.side_effect = ValueError("missing")
        config = AppConfig(
            subject=EntityRef(ref="missing"),
            api=EntityRef(ref="test_drug"),
            product=EntityRef(ref="test_inhaler"),
            maneuver=EntityRef(ref="test_maneuver"),
        )
        with pytest.raises(ConfigError):
            hydrate_config(config, mock_catalog)

    def test_hydration_invalid_override(self, mock_catalog):
        demo_override = make_subject().demographic.model_dump()
        demo_override["age_years"] = -5
        config = AppConfig(
            subject=EntityRef(ref="test_adult", overrides={"demographic": demo_override}),
            api=EntityRef(ref="test_drug"),
            product=EntityRef(ref="test_inhaler"),
            maneuver=EntityRef(ref="test_maneuver"),
        )
        with pytest.raises(ConfigError):
            hydrate_config(config, mock_catalog)

    def test_hydration_uses_default_catalog(self):
        config = AppConfig()
        with patch("lmp_pkg.config.hydration.get_default_catalog") as mock_get_catalog:
            catalog = Mock(spec=ReadOnlyCatalog)
            mock_get_catalog.return_value = catalog
            catalog.get_entry.side_effect = lambda cat, name: {
                ("subject", "adult_70kg"): make_subject(name="adult_70kg"),
                ("api", "salbutamol"): make_api(name="salbutamol"),
                ("product", "HFA_MDI_PT210"): make_product(name="HFA_MDI_PT210"),
                ("maneuver", "standard_MDI_inhale_hold"): make_maneuver(name="standard_MDI_inhale_hold"),
            }[(cat, name)]
            hydrate_config(config)
            mock_get_catalog.assert_called_once()


class TestEntityValidation:
    def create_hydrated(self, **overrides):
        hydrated = {
            "subject": make_subject(),
            "api": make_api(),
            "product": make_product(),
            "maneuver": make_maneuver(),
        }
        hydrated.update(overrides)
        hydrated.setdefault("config", {})
        return hydrated

    def collect_warnings(self, hydrated):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            validate_hydrated_entities(hydrated)
        return [str(item.message) for item in caught]

    def test_valid_entities_no_warnings(self):
        messages = self.collect_warnings(self.create_hydrated())
        assert all("Entity validation warning" not in msg for msg in messages)

    def test_pediatric_high_flow_warning(self):
        subject = make_subject(demographic={"age_years": 8, "weight_kg": 25, "height_cm": 125, "sex": "F"})
        maneuver = make_maneuver(pifr_Lpm=95.0)
        messages = self.collect_warnings(self.create_hydrated(subject=subject, maneuver=maneuver))
        assert any("High flow rate" in msg for msg in messages)

    def test_dpi_maneuver_compatibility(self):
        product = make_product(device="DPI")
        object.__setattr__(product, "device_type", "DPI")
        maneuver = make_maneuver(maneuver_type="gentle", pifr_Lpm=20.0)
        messages = self.collect_warnings(self.create_hydrated(product=product, maneuver=maneuver))
        assert any("DPI devices" in msg for msg in messages)

    def test_pmdi_high_flow_warning(self):
        maneuver = make_maneuver(pifr_Lpm=110.0)
        messages = self.collect_warnings(self.create_hydrated(maneuver=maneuver))
        assert any("pMDI efficiency" in msg for msg in messages)

    def test_pediatric_high_dose_warning(self):
        subject = make_subject(demographic={"age_years": 10, "weight_kg": 30, "height_cm": 135})
        product = make_product()
        object.__setattr__(product, "label_claim_mg", 0.5)
        messages = self.collect_warnings(self.create_hydrated(subject=subject, product=product))
        assert any("High dose" in msg for msg in messages)

    def test_low_coordination_pmdi_warning(self):
        maneuver = make_maneuver(coordination_efficiency=0.5)
        messages = self.collect_warnings(self.create_hydrated(maneuver=maneuver))
        assert any("Low coordination efficiency" in msg for msg in messages)
