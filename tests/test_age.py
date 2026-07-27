import numpy as np

import pytest

from pyseshat.evolution.age import (
    TrackFitResult,
    CommonAgeFitResult,
    _track_chi2,
    fit_common_age,
    fit_track,
    _common_age_grid,
)
from pyseshat.evolution.models import EvolutionTrack

from pyseshat.evolution.interpolate import (sample_track)

from pyseshat.evolution.observables import (
    StellarObservables,
)


def make_track():

    return EvolutionTrack(
        initial_mass=1.0,
        metallicity=0.0,
        age=np.array([1.0, 2.0, 3.0, 4.0]),
        log_teff=np.array([3.90, 3.80, 3.70, 3.60]),
        log_luminosity=np.array([-0.2, 0.0, 0.2, 0.4]),
        log_radius=np.array([0.0, 0.1, 0.2, 0.3]),
        log_g=np.array([4.6, 4.5, 4.4, 4.3]),
    )


def test_fit_track():

    result = fit_track(
        make_track(),
        observables=StellarObservables(
            log_teff=3.79,
            sigma_log_teff=0.02,
            log_luminosity=0.02,
            sigma_log_luminosity=0.05,
        ),
    )

    assert isinstance(result, TrackFitResult)

    assert result.index == 1

    assert result.sample.age == pytest.approx(2.0)

    assert result.chi2 >= 0.0


def test_exact_match():

    result = fit_track(
        make_track(),
        observables=StellarObservables(
            log_teff=3.70,
            sigma_log_teff=0.01,
            log_luminosity=0.20,
            sigma_log_luminosity=0.01,
        ),
    )

    assert result.index == 2

    assert result.chi2 == pytest.approx(0.0)

def make_secondary_track():

    return EvolutionTrack(
        initial_mass=0.8,
        metallicity=0.0,
        age=np.array([2.0, 3.0, 4.0]),
        log_teff=np.array([3.78, 3.70, 3.62]),
        log_luminosity=np.array([-0.05, 0.10, 0.25]),
        log_radius=np.array([0.0, 0.1, 0.2]),
        log_g=np.array([4.55, 4.45, 4.35]),
    )

def test_fit_common_age():

    result = fit_common_age(
        primary_track=make_track(),
        secondary_track=make_secondary_track(),

        primary_observables=StellarObservables(
            log_teff=3.70,
            sigma_log_teff=0.02,
            log_luminosity=0.20,
            sigma_log_luminosity=0.02,
        ),

        secondary_observables=StellarObservables(
            log_teff=3.70,
            sigma_log_teff=0.02,
            log_luminosity=0.10,
            sigma_log_luminosity=0.02,
        ),
    )

    assert isinstance(result, CommonAgeFitResult)

    assert result.age == pytest.approx(3.0)

    assert result.chi2 >= 0.0

def test_fit_common_age_no_overlap():

    track = EvolutionTrack(
        initial_mass=1.0,
        metallicity=0.0,
        age=np.array([10.0, 11.0]),
        log_teff=np.array([3.8, 3.8]),
        log_luminosity=np.array([0.0, 0.0]),
        log_radius=np.array([0.0, 0.0]),
        log_g=np.array([4.5, 4.5]),
    )

    with pytest.raises(ValueError):
        fit_common_age(
            primary_track=make_track(),
            secondary_track=track,

            primary_observables=StellarObservables(
                log_teff=3.8,
                sigma_log_teff=0.1,
                log_luminosity=0.0,
                sigma_log_luminosity=0.1,
            ),

            secondary_observables=StellarObservables(
                log_teff=3.8,
                sigma_log_teff=0.1,
                log_luminosity=0.0,
                sigma_log_luminosity=0.1,
            ),
        )

def test_common_age_grid_identical():

    ages = _common_age_grid(
        make_track(),
        make_track(),
    )

    assert np.array_equal(
        ages,
        np.array([1.0, 2.0, 3.0, 4.0]),
    )

def test_common_age_grid_union():

    ages = _common_age_grid(
        make_track(),
        make_secondary_track(),
    )

    assert np.array_equal(
        ages,
        np.array([2.0, 3.0, 4.0]),
    )

def test_common_age_grid_interleaved():

    primary = EvolutionTrack(
        initial_mass=1.0,
        metallicity=0.0,
        age=np.array([1.0, 3.0, 5.0]),
        log_teff=np.zeros(3),
        log_luminosity=np.zeros(3),
        log_radius=np.zeros(3),
        log_g=np.zeros(3),
    )

    secondary = EvolutionTrack(
        initial_mass=0.8,
        metallicity=0.0,
        age=np.array([2.0, 4.0, 6.0]),
        log_teff=np.zeros(3),
        log_luminosity=np.zeros(3),
        log_radius=np.zeros(3),
        log_g=np.zeros(3),
    )

    ages = _common_age_grid(
        primary,
        secondary,
    )

    assert np.array_equal(
        ages,
        np.array([2.0, 3.0, 4.0, 5.0]),
    )

def test_track_chi2():

    sample = sample_track(
        make_track(),
        2.0,
    )

    obs = StellarObservables(
        log_teff=3.80,
        sigma_log_teff=0.01,
        log_luminosity=0.00,
        sigma_log_luminosity=0.01,
    )

    assert _track_chi2(sample, obs) == pytest.approx(0.0)

def test_track_chi2_single_observable():

    sample = sample_track(
        make_track(),
        2.0,
    )

    obs = StellarObservables(
        log_teff=3.80,
        sigma_log_teff=0.02,
    )

    assert _track_chi2(sample, obs) == pytest.approx(0.0)

def test_track_chi2_no_observables():

    sample = sample_track(
        make_track(),
        2.0,
    )

    with pytest.raises(ValueError):

        _track_chi2(
            sample,
            StellarObservables(),
        )