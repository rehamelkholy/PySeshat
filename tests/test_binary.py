import numpy as np

import pytest

from pyseshat.evolution.binary import (
    BinaryTrackFitResult,
    fit_binary_evolution_from_orbit,
    fit_binary_tracks,
)

from pyseshat.evolution.models import (
    EvolutionModel,
    EvolutionTrack,
)

from pyseshat.evolution.observables import (
    StellarObservables,
)

from pyseshat.evolution.solution import (
    BinaryEvolutionSolution,
)

from pyseshat.orbit.orbital_elements import (
    OrbitalElements,
)


def make_track():

    return EvolutionTrack(
        initial_mass=1.0,
        metallicity=0.0,
        age=np.array([1.0, 2.0, 3.0]),
        log_teff=np.array([3.90, 3.80, 3.70]),
        log_luminosity=np.array([-0.2, 0.0, 0.2]),
        log_radius=np.array([0.0, 0.1, 0.2]),
        log_g=np.array([4.6, 4.5, 4.4]),
    )


def test_binary_fit():

    result = fit_binary_tracks(
        primary_track=make_track(),
        secondary_track=make_track(),

        primary_log_teff=3.80,
        primary_log_luminosity=0.00,
        primary_sigma_log_teff=0.01,
        primary_sigma_log_luminosity=0.01,

        secondary_log_teff=3.70,
        secondary_log_luminosity=0.20,
        secondary_sigma_log_teff=0.01,
        secondary_sigma_log_luminosity=0.01,
    )

    assert isinstance(
        result,
        BinaryTrackFitResult,
    )

    assert result.primary.sample.age == pytest.approx(
        2.0
    )

    assert result.secondary.sample.age == pytest.approx(
        3.0
    )

    assert result.age_difference == pytest.approx(
        1.0
    )


def make_evolution_model():

    return EvolutionModel(

        name="Test Model",

        tracks=(

            EvolutionTrack(
                initial_mass=1.0,
                metallicity=0.0,
                age=np.array(
                    [1.0, 2.0, 3.0]
                ),
                log_teff=np.array(
                    [3.90, 3.80, 3.70]
                ),
                log_luminosity=np.array(
                    [-0.20, 0.00, 0.20]
                ),
                log_radius=np.array(
                    [0.00, 0.10, 0.20]
                ),
                log_g=np.array(
                    [4.60, 4.50, 4.40]
                ),
            ),

            EvolutionTrack(
                initial_mass=0.8,
                metallicity=0.0,
                age=np.array(
                    [1.0, 2.0, 3.0]
                ),
                log_teff=np.array(
                    [3.85, 3.75, 3.65]
                ),
                log_luminosity=np.array(
                    [-0.30, -0.10, 0.10]
                ),
                log_radius=np.array(
                    [-0.05, 0.05, 0.15]
                ),
                log_g=np.array(
                    [4.70, 4.60, 4.50]
                ),
            ),

        ),

        isochrones=(),
    )


def make_sb2_orbital_elements():

    #
    # P = 1 yr
    # e = 0
    # i = 90 deg
    #
    # These semi-amplitudes give:
    #
    # M1 = 1.0 Msun
    # M2 = 0.8 Msun
    #
    # using the existing SB2 component-mass
    # implementation.
    #

    return OrbitalElements(

        P=1.0,

        T0=0.0,

        e=0.0,

        a=1.0,

        Omega=0.0,

        omega=0.0,

        i=90.0,

        K1=16.10290279498156,

        K2=20.12862849372695,
    )


def make_primary_observables():

    return StellarObservables(

        log_teff=3.80,

        sigma_log_teff=0.01,

        log_luminosity=0.00,

        sigma_log_luminosity=0.01,
    )


def make_secondary_observables():

    return StellarObservables(

        log_teff=3.75,

        sigma_log_teff=0.01,

        log_luminosity=-0.10,

        sigma_log_luminosity=0.01,
    )


def test_binary_evolution_from_orbit():

    result = fit_binary_evolution_from_orbit(

        orbital_elements=(
            make_sb2_orbital_elements()
        ),

        evolution_model=(
            make_evolution_model()
        ),

        metallicity=0.0,

        primary_observables=(
            make_primary_observables()
        ),

        secondary_observables=(
            make_secondary_observables()
        ),
    )

    assert isinstance(
        result,
        BinaryEvolutionSolution,
    )

    #
    # The orbital component masses should select
    # the 1.0 and 0.8 Msun tracks.
    #

    assert result.primary.initial_mass == pytest.approx(
        1.0
    )

    assert result.secondary.initial_mass == pytest.approx(
        0.8
    )

    #
    # Both tracks share the same age grid, so the
    # resulting solution must have a common age.
    #

    assert result.age in (
        1.0,
        2.0,
        3.0,
    )

    assert result.primary.age == pytest.approx(
        result.age
    )

    assert result.secondary.age == pytest.approx(
        result.age
    )

    assert result.metallicity == pytest.approx(
        0.0
    )

    assert result.chi2 >= 0.0


def test_binary_evolution_from_orbit_requires_tracks():

    model = EvolutionModel(

        name="Empty",

        tracks=(),

        isochrones=(),
    )

    with pytest.raises(ValueError):

        fit_binary_evolution_from_orbit(

            orbital_elements=(
                make_sb2_orbital_elements()
            ),

            evolution_model=model,

            metallicity=0.0,

            primary_observables=(
                make_primary_observables()
            ),

            secondary_observables=(
                make_secondary_observables()
            ),
        )