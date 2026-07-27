import numpy as np
import pytest

from pyseshat.evolution.interpolate import EvolutionSample
from pyseshat.evolution.isochrone import (
    IsochroneFitResult,
    fit_isochrone,
)
from pyseshat.evolution.models import Isochrone
from pyseshat.evolution.observables import StellarObservables


def make_isochrone():

    return Isochrone(
        age=5.0,
        metallicity=0.0,

        mass=np.array([
            0.8,
            1.0,
            1.2,
            1.4,
        ]),

        log_teff=np.array([
            3.68,
            3.75,
            3.82,
            3.88,
        ]),

        log_luminosity=np.array([
            -0.20,
            0.00,
            0.25,
            0.55,
        ]),

        log_radius=np.array([
            -0.10,
            0.00,
            0.10,
            0.25,
        ]),

        log_g=np.array([
            4.60,
            4.45,
            4.30,
            4.10,
        ]),
    )

def test_fit_isochrone_returns_result():

    result = fit_isochrone(
        make_isochrone(),
        observables=StellarObservables(
            log_teff=3.76,
            sigma_log_teff=0.02,
            log_luminosity=0.02,
            sigma_log_luminosity=0.05,
        ),
    )

    assert isinstance(
        result,
        IsochroneFitResult,
    )

    assert isinstance(
        result.sample,
        EvolutionSample,
    )

def test_fit_isochrone_best_mass():

    result = fit_isochrone(
        make_isochrone(),
        observables=StellarObservables(
            log_teff=3.75,
            sigma_log_teff=0.02,
            log_luminosity=0.00,
            sigma_log_luminosity=0.02,
        ),
    )

    assert result.index == 1

    assert result.sample.mass == pytest.approx(1.0)

def test_fit_isochrone_exact_match():

    result = fit_isochrone(
        make_isochrone(),
        observables=StellarObservables(
            log_teff=3.82,
            sigma_log_teff=0.01,
            log_luminosity=0.25,
            sigma_log_luminosity=0.01,
        ),
    )

    assert result.chi2 == pytest.approx(0.0)

def test_fit_isochrone_returns_sample():

    result = fit_isochrone(
        make_isochrone(),
        observables=StellarObservables(
            log_teff=3.88,
            sigma_log_teff=0.02,
            log_luminosity=0.55,
            sigma_log_luminosity=0.02,
        ),
    )

    assert result.sample.log_teff == pytest.approx(3.88)

    assert result.sample.log_luminosity == pytest.approx(0.55)