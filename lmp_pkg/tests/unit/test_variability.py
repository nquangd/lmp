"""Tests for variability system aligned with the updated data model."""

import numpy as np
import pytest

from lmp_pkg.domain import Subject, API, Product, InhalationProfile, Demographic
from lmp_pkg.variability import (
    VariabilitySpec,
    DistributionSpec,
    build_inter_subject,
    build_intra_subject,
    get_pk_scale,
    convert_gcv_to_sigma_log,
    apply_population_variability_settings,
)


def make_subject(**overrides) -> Subject:
    demo_defaults = dict(
        name="adult",
        subject_id="adult",
        age_years=30.0,
        weight_kg=70.0,
        height_cm=175.0,
        sex="M",
        frc_ml=3300.0,
        frc_ref_ml=3300.0,
    )
    demo_defaults.update(overrides.pop("demographic", {}))
    demo = Demographic(**demo_defaults)
    return Subject(name=overrides.pop("name", "adult"), demographic=demo, **overrides)


def make_api(**overrides) -> API:
    base = dict(name="drug", molecular_weight=250.0)
    base.update(overrides)
    return API(**base)


def make_product(**overrides) -> Product:
    base = dict(
        name="product",
        device="pMDI",
        propellant="HFA-134a",
        apis={"drug": {"dose_pg": 1000.0, "mmad": 3.0, "gsd": 1.6, "usp_depo_fraction": 40.0}},
    )
    base.update(overrides)
    product = Product(**base)
    object.__setattr__(product, "device_type", product.device or "pMDI")
    return product


def make_maneuver(**overrides) -> InhalationProfile:
    base = dict(
        name="maneuver",
        maneuver_type="slow_deep",
        pifr_Lpm=60.0,
        rise_time_s=1.0,
        inhaled_volume_L=2.0,
        hold_time_s=0.5,
        breath_hold_time_s=5.0,
        exhalation_flow_Lpm=30.0,
        bolus_volume_ml=0.0,
        bolus_delay_s=0.0,
    )
    base.update(overrides)
    return InhalationProfile(**base)


class TestDistributionSpec:
    def test_lognormal_with_gcv(self):
        spec = DistributionSpec(dist="lognormal", gcv=0.3)
        assert spec.get_effective_sigma_log() == pytest.approx(convert_gcv_to_sigma_log(0.3))

    def test_lognormal_with_sigma(self):
        spec = DistributionSpec(dist="lognormal", sigma_log=0.25)
        assert spec.get_effective_sigma_log() == 0.25

    def test_lognormal_missing_params(self):
        with pytest.raises(ValueError):
            DistributionSpec(dist="lognormal")

    def test_uniform_validation(self):
        with pytest.raises(ValueError):
            DistributionSpec(dist="uniform", min=2.0, max=1.0)


class TestVariabilitySpec:
    def test_from_original_format(self):
        spec = VariabilitySpec.from_original_format()
        assert spec.layers == ["inter", "intra"]
        assert "pifr_Lpm" in spec.inter.inhalation
        assert spec.inter.physiology["FRC"].mean == pytest.approx(3300.0)

    def test_disable_all_variability(self):
        disabled = VariabilitySpec.from_original_format().disable_all_variability()
        assert disabled.inter.inhalation["pifr_Lpm"].sigma_log == 0.0
        assert disabled.inter.physiology["FRC"].sd == 0.0


class TestFactorGeneration:
    @pytest.fixture
    def rng(self):
        return np.random.default_rng(123)

    def test_convert_gcv(self):
        assert convert_gcv_to_sigma_log(0.3) == pytest.approx(0.2936, abs=1e-3)

    def test_log_sample(self, rng):
        from lmp_pkg.variability.factors import sample_multiplicative_factor

        spec = DistributionSpec(dist="lognormal", gcv=0.2)
        factor = sample_multiplicative_factor(spec, rng)
        assert factor > 0

    def test_absolute_sample(self, rng):
        from lmp_pkg.variability.factors import sample_absolute_value

        spec = DistributionSpec(dist="normal_absolute", mean=3300.0, sd=600.0, mode="absolute")
        value = sample_absolute_value(spec, rng)
        assert 1500 < value < 5000


class TestInterIntraGeneration:
    @pytest.fixture
    def base_entities(self):
        return {
            "subject": make_subject(),
            "api": make_api(),
            "product": make_product(),
            "maneuver": make_maneuver(),
        }

    @pytest.fixture
    def spec(self):
        return VariabilitySpec.from_original_format()

    @pytest.fixture
    def rng(self):
        return np.random.default_rng(42)

    def test_build_inter_subject(self, base_entities, spec, rng):
        modified, factors = build_inter_subject(base_entities, spec, rng)
        assert set(factors.keys()) == {"inhalation", "pk", "physiology"}
        assert isinstance(modified["subject"], Subject)
        assert isinstance(modified["maneuver"], InhalationProfile)

    def test_build_intra_subject(self, base_entities, spec, rng):
        inter_entities, inter_factors = build_inter_subject(base_entities, spec, rng)
        modified = build_intra_subject(inter_entities, inter_factors, spec, rng, replicate_id=1)
        assert "metadata" in modified
        assert modified["metadata"]["replicate_id"] == 1

    def test_get_pk_scale(self):
        factors = {"CL": {"BD": 1.2}}
        assert get_pk_scale(factors)["CL"]["BD"] == 1.2


class TestSettings:
    def test_apply_population_variability_settings(self):
        spec = VariabilitySpec.from_original_format()
        filtered = apply_population_variability_settings(spec, {"inhalation": False, "pk": False})
        assert filtered.inter.inhalation == {}
        assert filtered.inter.pk == {}
