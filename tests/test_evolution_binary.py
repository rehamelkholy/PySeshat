"""
Tests for the evolutionary-track search/fitting layer:
pyseshat.evolution.search, pyseshat.evolution.age, and
pyseshat.evolution.binary.

Before this file was added, none of this layer had direct test
coverage -- only the downstream photometry adapter
(tests/test_evolution_adapter.py), track/isochrone interpolation
(tests/test_interpolate.py), and HR-diagram plotting
(tests/test_hrdiagram_plot.py) were tested. See
docs/guide/evolutionary_checks.md's 2026-08-29 entry for the audit
that found this gap.

All tracks/isochrones here are small synthetic fixtures built
in-process (no external grid files), consistent with this project's
"never fabricate scientific data" rule -- these are unit tests of the
fitting *mechanics* (nearest-neighbor search, chi-square minimization,
common-age solving, mass-from-orbit wiring), not claims about any real
star.
"""

import numpy as np

import pytest

from pyseshat.evolution.age import (
    CommonAgeFitResult,
    TrackFitResult,
    fit_common_age,
    fit_track,
)

from pyseshat.evolution.binary import (
    BinaryTrackFitResult,
    fit_binary_evolution_from_orbit,
    fit_binary_tracks,
)

from pyseshat.evolution.models import (
    EvolutionModel,
    EvolutionTrack,
    Isochrone,
)

from pyseshat.evolution.observables import (
    StellarObservables,
)

from pyseshat.evolution.search import (
    nearest_isochrone,
    nearest_track,
)

from pyseshat.evolution.solution import (
    BinaryEvolutionSolution,
)

from pyseshat.orbit.orbital_elements import (
    OrbitalElements,
)


# ==========================================================
# Synthetic fixtures
# ==========================================================

def _make_track(initial_mass, metallicity=0.019, n=6):
    """
    A monotonically-cooling, monotonically-brightening
    synthetic main-sequence-like track, scaled loosely with
    mass so that different masses land at visibly different
    (log_teff, log_luminosity) -- enough to exercise
    nearest-neighbor search and chi-square minimization
    meaningfully, without representing any real star.
    """

    age = np.linspace(1.0e8, 1.0e10, n)

    log_teff = 3.80 - 0.01 * (age / 1.0e9) + 0.05 * (
        initial_mass - 1.0
    )

    log_luminosity = 0.10 * (age / 1.0e9) + 0.30 * (
        initial_mass - 1.0
    )

    log_radius = 0.05 * (age / 1.0e9)

    log_g = 4.5 - 0.02 * (age / 1.0e9)

    return EvolutionTrack(
        initial_mass=initial_mass,
        metallicity=metallicity,
        age=age,
        log_teff=log_teff,
        log_luminosity=log_luminosity,
        log_radius=log_radius,
        log_g=log_g,
    )


def _make_isochrone(age, metallicity=0.019):

    mass = np.array([0.8, 1.0, 1.2, 1.5])

    return Isochrone(
        age=age,
        metallicity=metallicity,
        mass=mass,
        log_teff=3.70 + 0.05 * (mass - 0.8),
        log_luminosity=-0.2 + 0.5 * (mass - 0.8),
        log_radius=-0.1 + 0.3 * (mass - 0.8),
        log_g=4.6 - 0.1 * (mass - 0.8),
    )


def _make_model():

    tracks = tuple(
        _make_track(m) for m in (0.8, 1.0, 1.2, 1.5, 2.0)
    )

    isochrones = tuple(
        _make_isochrone(age)
        for age in (1.0e9, 5.0e9, 1.0e10)
    )

    return EvolutionModel(
        name="SyntheticTestGrid",
        tracks=tracks,
        isochrones=isochrones,
    )


# ==========================================================
# search.py
# ==========================================================

def test_nearest_track_picks_closest_mass():

    model = _make_model()

    track = nearest_track(
        model, initial_mass=1.05, metallicity=0.019
    )

    assert track.initial_mass == 1.0


def test_nearest_track_breaks_ties_with_metallicity():

    tracks = (
        _make_track(1.0, metallicity=0.008),
        _make_track(1.0, metallicity=0.019),
    )

    model = EvolutionModel(
        name="test", tracks=tracks, isochrones=()
    )

    track = nearest_track(
        model, initial_mass=1.0, metallicity=0.02
    )

    assert track.metallicity == 0.019


def test_nearest_track_empty_model_raises():

    model = EvolutionModel(
        name="empty", tracks=(), isochrones=()
    )

    with pytest.raises(ValueError):
        nearest_track(
            model, initial_mass=1.0, metallicity=0.019
        )


def test_nearest_isochrone_picks_closest_age():

    model = _make_model()

    iso = nearest_isochrone(
        model, age=4.5e9, metallicity=0.019
    )

    assert iso.age == 5.0e9


def test_nearest_isochrone_empty_model_raises():

    model = EvolutionModel(
        name="empty", tracks=(), isochrones=()
    )

    with pytest.raises(ValueError):
        nearest_isochrone(
            model, age=1.0e9, metallicity=0.019
        )


# ==========================================================
# age.py
# ==========================================================

def test_fit_track_recovers_a_known_point_on_its_own_track():
    """
    Fitting a track against observables copied exactly from
    one of its own tabulated points must recover that point
    (chi2 ~ 0), independent of any external grid data.
    """

    track = _make_track(1.0)

    target_index = 3

    observables = StellarObservables(
        log_teff=float(track.log_teff[target_index]),
        sigma_log_teff=0.01,
        log_luminosity=float(
            track.log_luminosity[target_index]
        ),
        sigma_log_luminosity=0.01,
    )

    result = fit_track(track, observables=observables)

    assert isinstance(result, TrackFitResult)
    assert result.chi2 < 1e-6
    assert result.sample.age == pytest.approx(
        track.age[target_index], rel=1e-6
    )


def test_fit_common_age_recovers_common_age_for_coeval_components():
    """
    Two components drawn from tracks built with the same
    age grid, sampled at the same true age, must recover
    that common age when fit jointly.
    """

    primary_track = _make_track(1.2)
    secondary_track = _make_track(0.9)

    true_age = primary_track.age[2]

    primary_obs = StellarObservables(
        log_teff=float(
            np.interp(
                true_age, primary_track.age,
                primary_track.log_teff,
            )
        ),
        sigma_log_teff=0.01,
        log_luminosity=float(
            np.interp(
                true_age, primary_track.age,
                primary_track.log_luminosity,
            )
        ),
        sigma_log_luminosity=0.01,
    )

    secondary_obs = StellarObservables(
        log_teff=float(
            np.interp(
                true_age, secondary_track.age,
                secondary_track.log_teff,
            )
        ),
        sigma_log_teff=0.01,
        log_luminosity=float(
            np.interp(
                true_age, secondary_track.age,
                secondary_track.log_luminosity,
            )
        ),
        sigma_log_luminosity=0.01,
    )

    result = fit_common_age(
        primary_track=primary_track,
        secondary_track=secondary_track,
        primary_observables=primary_obs,
        secondary_observables=secondary_obs,
    )

    assert isinstance(result, CommonAgeFitResult)
    assert result.age == pytest.approx(true_age, rel=1e-6)


def test_fit_common_age_no_overlap_raises():

    track_a = EvolutionTrack(
        initial_mass=1.0,
        metallicity=0.019,
        age=np.array([1.0e8, 2.0e8]),
        log_teff=np.array([3.8, 3.7]),
        log_luminosity=np.array([0.0, 0.1]),
        log_radius=np.array([0.0, 0.1]),
        log_g=np.array([4.5, 4.4]),
    )

    track_b = EvolutionTrack(
        initial_mass=1.0,
        metallicity=0.019,
        age=np.array([5.0e9, 6.0e9]),
        log_teff=np.array([3.6, 3.5]),
        log_luminosity=np.array([0.5, 0.6]),
        log_radius=np.array([0.3, 0.4]),
        log_g=np.array([4.0, 3.9]),
    )

    obs = StellarObservables(
        log_teff=3.7, sigma_log_teff=0.01,
        log_luminosity=0.05, sigma_log_luminosity=0.01,
    )

    with pytest.raises(ValueError):
        fit_common_age(
            primary_track=track_a,
            secondary_track=track_b,
            primary_observables=obs,
            secondary_observables=obs,
        )


# ==========================================================
# binary.py
# ==========================================================

def test_fit_binary_tracks_reports_zero_age_difference_for_coeval_input():

    primary_track = _make_track(1.2)
    secondary_track = _make_track(0.9)

    true_age = primary_track.age[4]

    def _obs(track):
        return StellarObservables(
            log_teff=float(
                np.interp(
                    true_age, track.age, track.log_teff
                )
            ),
            sigma_log_teff=0.01,
            log_luminosity=float(
                np.interp(
                    true_age, track.age,
                    track.log_luminosity,
                )
            ),
            sigma_log_luminosity=0.01,
        )

    primary_obs = _obs(primary_track)
    secondary_obs = _obs(secondary_track)

    result = fit_binary_tracks(
        primary_track=primary_track,
        secondary_track=secondary_track,
        primary_log_teff=primary_obs.log_teff,
        primary_log_luminosity=primary_obs.log_luminosity,
        primary_sigma_log_teff=primary_obs.sigma_log_teff,
        primary_sigma_log_luminosity=(
            primary_obs.sigma_log_luminosity
        ),
        secondary_log_teff=secondary_obs.log_teff,
        secondary_log_luminosity=(
            secondary_obs.log_luminosity
        ),
        secondary_sigma_log_teff=(
            secondary_obs.sigma_log_teff
        ),
        secondary_sigma_log_luminosity=(
            secondary_obs.sigma_log_luminosity
        ),
    )

    assert isinstance(result, BinaryTrackFitResult)
    assert result.age_difference == pytest.approx(
        0.0, abs=1e-3 * true_age
    )


def test_fit_binary_evolution_from_orbit_end_to_end():
    """
    Exercises the full OrbitalElements -> component masses ->
    nearest tracks -> common-age fit -> BinaryEvolutionSolution
    pipeline in binary.fit_binary_evolution_from_orbit(), the
    function docs/guide/evolutionary_checks.md documents as the
    connective tissue this project's own benchmark wiring would
    call. Elements are an arbitrary, physically-plausible SB2
    solution (not any real system's literature values) --
    this test is about the wiring being correct, not about
    reproducing any particular star.
    """

    elements = OrbitalElements(
        P=10.0,
        T0=0.0,
        e=0.3,
        a=1.0,
        Omega=45.0,
        omega=90.0,
        i=60.0,
        K1=20.0,
        K2=25.0,
        V0=0.0,
    )

    model = _make_model()

    primary_obs = StellarObservables(
        log_teff=3.78, sigma_log_teff=0.02,
        log_luminosity=0.05, sigma_log_luminosity=0.05,
    )

    secondary_obs = StellarObservables(
        log_teff=3.75, sigma_log_teff=0.02,
        log_luminosity=-0.10, sigma_log_luminosity=0.05,
    )

    solution = fit_binary_evolution_from_orbit(
        orbital_elements=elements,
        evolution_model=model,
        metallicity=0.019,
        primary_observables=primary_obs,
        secondary_observables=secondary_obs,
    )

    assert isinstance(solution, BinaryEvolutionSolution)
    assert solution.metallicity == 0.019
    # solution.age comes from the common-age fit over the two
    # nearest *tracks* (age.fit_common_age), not from the
    # model's isochrone-age grid (EvolutionModel.available_ages
    # only enumerates isochrone ages) -- so check it against
    # the tracks' own overlapping age range instead.
    all_track_ages = np.concatenate(
        [t.age for t in model.tracks]
    )
    assert all_track_ages.min() <= solution.age <= all_track_ages.max()
    assert solution.chi2 >= 0.0
