import numpy as np

import pytest

from pyseshat.evolution.interpolate import (
    EvolutionSample,
    _interp,
    contains_age,
    contains_mass,
    sample_isochrone,
    sample_track,
)

from pyseshat.evolution.models import (
    EvolutionTrack,
    Isochrone,
)


def make_track():

    return EvolutionTrack(
        initial_mass=1.0,
        metallicity=0.0,
        age=np.array([1.0, 2.0, 3.0]),
        log_teff=np.array([3.80, 3.70, 3.60]),
        log_luminosity=np.array([0.0, 0.2, 0.4]),
        log_radius=np.array([0.0, 0.1, 0.2]),
        log_g=np.array([4.5, 4.4, 4.3]),
        phase=np.array([1, 2, 3]),
    )


def make_iso():

    return Isochrone(
        age=1e9,
        metallicity=0.0,
        mass=np.array([0.8, 1.0, 1.2]),
        log_teff=np.array([3.70, 3.80, 3.90]),
        log_luminosity=np.array([-0.2, 0.0, 0.2]),
        log_radius=np.array([-0.1, 0.0, 0.1]),
        log_g=np.array([4.6, 4.5, 4.4]),
        phase=np.array([1, 2, 3]),
    )


def test_interp():

    assert _interp(
        1.5,
        [1.0, 2.0],
        [10.0, 20.0],
    ) == pytest.approx(15.0)


def test_interp_outside_range():

    with pytest.raises(ValueError):
        _interp(
            5.0,
            [1.0, 2.0],
            [1.0, 2.0],
        )


def test_sample_track():

    sample = sample_track(
        make_track(),
        2.5,
    )

    assert isinstance(sample, EvolutionSample)

    assert sample.age == 2.5

    assert sample.initial_mass == 1.0

    assert sample.log_teff == pytest.approx(3.65)

    assert sample.phase == 2


def test_sample_isochrone():

    sample = sample_isochrone(
        make_iso(),
        0.9,
    )

    assert isinstance(sample, EvolutionSample)

    assert sample.mass == 0.9

    assert sample.age == 1e9

    assert sample.log_luminosity == pytest.approx(-0.1)

    assert sample.phase == 1

def test_contains_age():

    track = make_track()

    assert contains_age(track, 1.0)

    assert contains_age(track, 2.5)

    assert contains_age(track, 3.0)

    assert not contains_age(track, 0.5)

    assert not contains_age(track, 3.5)


def test_contains_mass():

    iso = make_iso()

    assert contains_mass(iso, 0.8)

    assert contains_mass(iso, 1.1)

    assert contains_mass(iso, 1.2)

    assert not contains_mass(iso, 0.7)

    assert not contains_mass(iso, 1.3)