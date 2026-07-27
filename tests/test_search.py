import numpy as np

import pytest

from pyseshat.evolution.models import (
    EvolutionModel,
    EvolutionTrack,
    Isochrone,
)

from pyseshat.evolution.search import (
    nearest_track,
    nearest_isochrone,
)


def make_track(mass, metallicity):

    return EvolutionTrack(
        initial_mass=mass,
        metallicity=metallicity,
        age=np.array([1.0]),
        log_teff=np.array([3.8]),
        log_luminosity=np.array([0.0]),
        log_radius=np.array([0.0]),
        log_g=np.array([4.5]),
    )


def make_iso(age, metallicity):

    return Isochrone(
        age=age,
        metallicity=metallicity,
        mass=np.array([1.0]),
        log_teff=np.array([3.8]),
        log_luminosity=np.array([0.0]),
        log_radius=np.array([0.0]),
        log_g=np.array([4.5]),
    )


def test_nearest_track():

    model = EvolutionModel(
        name="PARSEC",
        tracks=(
            make_track(1.0, 0.0),
            make_track(2.0, 0.0),
            make_track(3.0, -0.5),
        ),
        isochrones=(),
    )

    result = nearest_track(
        model,
        initial_mass=2.2,
        metallicity=0.0,
    )

    assert result.initial_mass == 2.0


def test_nearest_isochrone():

    model = EvolutionModel(
        name="PARSEC",
        tracks=(),
        isochrones=(
            make_iso(1e9, 0.0),
            make_iso(5e9, 0.0),
            make_iso(1e10, -0.5),
        ),
    )

    result = nearest_isochrone(
        model,
        age=4e9,
        metallicity=0.0,
    )

    assert result.age == 5e9


def test_empty_tracks():

    model = EvolutionModel(
        name="PARSEC",
        tracks=(),
        isochrones=(),
    )

    with pytest.raises(ValueError):
        nearest_track(
            model,
            initial_mass=1.0,
            metallicity=0.0,
        )


def test_empty_isochrones():

    model = EvolutionModel(
        name="PARSEC",
        tracks=(),
        isochrones=(),
    )

    with pytest.raises(ValueError):
        nearest_isochrone(
            model,
            age=1e9,
            metallicity=0.0,
        )