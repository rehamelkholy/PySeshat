import numpy as np
import pytest

from pyseshat.evolution.isochrone import IsochroneFitResult
from pyseshat.evolution.model_fit import (
    EvolutionModelFitResult,
    fit_evolution_model,
)
from pyseshat.evolution.models import (
    EvolutionModel,
    Isochrone,
)
from pyseshat.evolution.observables import StellarObservables


def make_isochrone_1():

    return Isochrone(
        age=1.0,
        metallicity=0.0,

        mass=np.array([
            0.8,
            1.0,
            1.2,
        ]),

        log_teff=np.array([
            3.65,
            3.70,
            3.75,
        ]),

        log_luminosity=np.array([
            -0.20,
            0.00,
            0.20,
        ]),

        log_radius=np.array([
            -0.10,
            0.00,
            0.10,
        ]),

        log_g=np.array([
            4.60,
            4.50,
            4.40,
        ]),
    )


def make_isochrone_2():

    return Isochrone(
        age=5.0,
        metallicity=0.0,

        mass=np.array([
            0.8,
            1.0,
            1.2,
        ]),

        log_teff=np.array([
            3.72,
            3.80,
            3.88,
        ]),

        log_luminosity=np.array([
            -0.05,
            0.15,
            0.40,
        ]),

        log_radius=np.array([
            0.00,
            0.10,
            0.20,
        ]),

        log_g=np.array([
            4.55,
            4.40,
            4.25,
        ]),
    )


def make_model():

    return EvolutionModel(
        name="Test Model",
        tracks=(),
        isochrones=(
            make_isochrone_1(),
            make_isochrone_2(),
        ),
    )


def test_empty_model():

    model = EvolutionModel(
        name="Empty",
        tracks=(),
        isochrones=(),
    )

    with pytest.raises(ValueError):
        fit_evolution_model(
            model,
            observables=StellarObservables(
                log_teff=3.80,
                sigma_log_teff=0.02,
            ),
        )


def test_returns_result():

    result = fit_evolution_model(
        make_model(),
        observables=StellarObservables(
            log_teff=3.80,
            sigma_log_teff=0.02,
            log_luminosity=0.15,
            sigma_log_luminosity=0.02,
        ),
    )

    assert isinstance(
        result,
        EvolutionModelFitResult,
    )

    assert isinstance(
        result.result,
        IsochroneFitResult,
    )


def test_best_isochrone_selected():

    result = fit_evolution_model(
        make_model(),
        observables=StellarObservables(
            log_teff=3.80,
            sigma_log_teff=0.01,
            log_luminosity=0.15,
            sigma_log_luminosity=0.01,
        ),
    )

    assert result.age == pytest.approx(5.0)

    assert result.metallicity == pytest.approx(0.0)

    assert result.chi2 == pytest.approx(0.0)


def test_preserves_inner_result():

    result = fit_evolution_model(
        make_model(),
        observables=StellarObservables(
            log_teff=3.80,
            sigma_log_teff=0.01,
            log_luminosity=0.15,
            sigma_log_luminosity=0.01,
        ),
    )

    assert result.result.index == 1

    assert result.result.sample.log_teff == pytest.approx(3.80)

    assert result.result.sample.log_luminosity == pytest.approx(0.15)