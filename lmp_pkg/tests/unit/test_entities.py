"""Tests for domain entities aligned with the updated schema."""

import numpy as np
import pytest
from pydantic import ValidationError

from lmp_pkg.domain.entities import (
    API,
    Demographic,
    EntityCollection,
    InhalationProfile,
    Product,
    Subject,
)


def make_demographic(**overrides) -> Demographic:
    base = dict(
        name="demo_subject",
        subject_id="demo_subject",
        age_years=35.0,
        weight_kg=72.0,
        height_cm=176.0,
        sex="M",
    )
    base.update(overrides)
    return Demographic(**base)


class TestSubject:
    def test_valid_subject(self):
        demo = make_demographic(name="adult", subject_id="adult")
        subject = Subject(name="adult", demographic=demo)

        assert subject.name == "adult"
        assert subject.demographic.age_years == 35.0
        assert subject.demographic.weight_kg == 72.0
        assert subject.demographic.sex == "M"

    def test_computed_properties(self):
        demo = make_demographic(weight_kg=70.0, height_cm=175.0)
        subject = Subject(name="metrics", demographic=demo)

        assert pytest.approx(subject.demographic.bmi_kg_m2, rel=1e-3) == 22.857
        assert 1.5 < subject.demographic.bsa_m2 < 2.5

    def test_sex_validation(self):
        for raw, expected in [("M", "M"), ("F", "F"), ("male", "M"), ("female", "F")]:
            demo = make_demographic(sex=raw)
            subject = Subject(name="sex", demographic=demo)
            assert subject.demographic.sex == expected

    def test_validation_errors(self):
        base = dict(name="demo_subject", subject_id="demo_subject", height_cm=175.0, weight_kg=70.0, sex="M")

        with pytest.raises(ValidationError):
            Demographic(**{**base, "age_years": -1})
        with pytest.raises(ValidationError):
            Demographic(**{**base, "weight_kg": 0})
        with pytest.raises(ValidationError):
            Demographic(**{**base, "height_cm": 10})


class TestAPI:
    def test_valid_api(self):
        api = API(
            name="salbutamol",
            molecular_weight=239.3,
            volume_central_L=35.0,
            clearance_L_h=15.0,
            fraction_unbound={"plasma": 0.75},
        )

        assert api.molecular_weight == 239.3
        assert api.volume_central_L == 35.0
        assert api.fraction_unbound["plasma"] == pytest.approx(0.75)

    def test_fraction_storage(self):
        data = {"plasma": 0.5, "tissue": 0.2}
        api = API(name="drug", molecular_weight=100.0, fraction_unbound=data)
        assert api.fraction_unbound == data

    def test_required_fields(self):
        with pytest.raises(ValidationError):
            API()


class TestProduct:
    def test_valid_product(self):
        product = Product(
            name="HFA_MDI",
            device="pMDI",
            propellant="HFA-134a",
            apis={
                "salbutamol": {
                    "dose_pg": 50000.0,
                    "mmad": 3.2,
                    "gsd": 1.6,
                    "usp_depo_fraction": 38.0,
                }
            },
        )

        params = product.get_api_parameters("salbutamol")
        assert params["dose_pg"] == 50000.0
        assert params["mmad"] == pytest.approx(3.2)

    def test_get_final_values(self):
        product = Product(
            name="device",
            device="pMDI",
            propellant="HFA-134a",
            apis={"drug": {"dose_pg": 1000.0, "mmad": 3.0, "gsd": 1.6, "usp_depo_fraction": 35.0}},
        )

        final_product = product.get_final_values("drug")
        assert final_product._final_dose_pg == 1000.0
        assert final_product._final_mmad == pytest.approx(3.0)
        assert final_product._final_usp_depo_fraction == pytest.approx(35.0)


class TestInhalationProfile:
    def make_profile(self, **overrides) -> InhalationProfile:
        base = dict(
            name="maneuver",
            maneuver_type="constant",
            pifr_Lpm=45.0,
            rise_time_s=0.0,
            inhaled_volume_L=1.5,
            hold_time_s=0.0,
            breath_hold_time_s=2.0,
            exhalation_flow_Lpm=30.0,
            bolus_volume_ml=0.0,
            bolus_delay_s=0.0,
        )
        base.update(overrides)
        return InhalationProfile(**base)

    def test_valid_profile(self):
        profile = self.make_profile()
        assert profile.pifr_Lpm == 45.0
        object.__setattr__(profile, "peak_inspiratory_flow_l_min", profile.pifr_Lpm)
        assert profile.peak_inspiratory_flow_l_min == 45.0

    def test_calculated_flow_profile(self):
        profile = self.make_profile(pifr_Lpm=60.0, inhaled_volume_L=1.2)
        flow = profile.calculate_inhalation_maneuver_flow_profile()
        assert flow.shape[1] == 2
        assert np.isclose(flow[:, 1].max(), 60.0)


class TestEntityCollection:
    def test_get_and_list(self):
        demo = make_demographic()
        subject = Subject(name="subj", demographic=demo)
        product = Product(name="prod", device="pMDI", propellant="HFA-134a", apis={})
        collection = EntityCollection(name="test", subjects={"subj": subject}, products={"prod": product})

        assert collection.get_entity("subject", "subj") is subject
        assert "subj" in collection.list_entities("subject")
