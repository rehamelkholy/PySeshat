import pytest

from pyseshat.evolution.interpolate import EvolutionSample
from pyseshat.evolution.observables import StellarObservables
from pyseshat.evolution.residuals import (
    StellarResiduals,
    compute_residuals,
)

def test_empty_residuals():

    r = StellarResiduals()

    assert r.available == ()


def test_available_fields():

    r = StellarResiduals(
        log_teff=0.02,
        log_g=-0.10,
    )

    assert r.available == (
        "log_teff",
        "log_g",
    )

def make_sample():

    return EvolutionSample(
        age=1e9,
        metallicity=0.0,
        initial_mass=1.0,
        log_teff=3.80,
        log_luminosity=0.10,
        log_radius=0.00,
        log_g=4.50,
    )


def test_compute_residuals():

    sample = make_sample()

    obs = StellarObservables(
        log_teff=3.82,
        sigma_log_teff=0.02,
        log_luminosity=0.05,
        sigma_log_luminosity=0.01,
    )

    residuals = compute_residuals(
        sample,
        obs,
    )

    assert isinstance(
        residuals,
        StellarResiduals,
    )

    assert residuals.log_teff == pytest.approx(0.02)

    assert residuals.log_luminosity == pytest.approx(-0.05)

    assert residuals.log_radius is None

    assert residuals.log_g is None