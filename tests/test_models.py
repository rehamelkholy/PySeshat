import numpy as np

import pytest

from pyseshat.evolution.models import (
    EvolutionTrack,
    Isochrone,
    EvolutionModel,
)


# ==========================================================
# Helpers
# ==========================================================

def make_track():

    n = 5

    return EvolutionTrack(

        initial_mass=1.0,

        metallicity=0.0,

        age=np.linspace(1e6, 1e9, n),

        log_teff=np.linspace(3.80, 3.75, n),

        log_luminosity=np.linspace(0.0, 0.4, n),

        log_radius=np.linspace(0.0, 0.2, n),

        log_g=np.linspace(4.5, 4.2, n),

        phase=np.arange(n)

    )


def make_isochrone():

    n = 4

    return Isochrone(

        age=1e9,

        metallicity=0.0,

        mass=np.array([0.8, 1.0, 1.2, 1.5]),

        log_teff=np.array([3.72, 3.76, 3.80, 3.85]),

        log_luminosity=np.array([-0.4, 0.0, 0.3, 0.8]),

        log_radius=np.array([-0.1, 0.0, 0.1, 0.2]),

        log_g=np.array([4.6, 4.5, 4.3, 4.1]),

        phase=np.arange(n)

    )


# ==========================================================
# EvolutionTrack
# ==========================================================

def test_track_creation():

    track = make_track()

    assert track.initial_mass == 1.0

    assert track.metallicity == 0.0

    assert len(track.age) == 5


def test_track_requires_equal_lengths():

    with pytest.raises(ValueError):

        EvolutionTrack(

            initial_mass=1.0,

            metallicity=0.0,

            age=np.arange(5),

            log_teff=np.arange(5),

            log_luminosity=np.arange(4),

            log_radius=np.arange(5),

            log_g=np.arange(5)

        )


# ==========================================================
# Isochrone
# ==========================================================

def test_isochrone_creation():

    iso = make_isochrone()

    assert iso.age == 1e9

    assert len(iso.mass) == 4


def test_isochrone_requires_equal_lengths():

    with pytest.raises(ValueError):

        Isochrone(

            age=1e9,

            metallicity=0.0,

            mass=np.arange(4),

            log_teff=np.arange(4),

            log_luminosity=np.arange(3),

            log_radius=np.arange(4),

            log_g=np.arange(4)

        )


# ==========================================================
# EvolutionModel
# ==========================================================

def test_model_available_metallicities():

    track1 = make_track()

    track2 = EvolutionTrack(

        initial_mass=2.0,

        metallicity=-0.5,

        age=track1.age,

        log_teff=track1.log_teff,

        log_luminosity=track1.log_luminosity,

        log_radius=track1.log_radius,

        log_g=track1.log_g

    )

    iso = make_isochrone()

    model = EvolutionModel(

        name="PARSEC",

        tracks=(track1, track2),

        isochrones=(iso,)

    )

    assert model.available_metallicities == (-0.5, 0.0)


def test_model_available_track_masses():

    track1 = make_track()

    track2 = EvolutionTrack(

        initial_mass=2.0,

        metallicity=0.0,

        age=track1.age,

        log_teff=track1.log_teff,

        log_luminosity=track1.log_luminosity,

        log_radius=track1.log_radius,

        log_g=track1.log_g

    )

    model = EvolutionModel(

        name="PARSEC",

        tracks=(track2, track1),

        isochrones=()

    )

    assert model.available_track_masses == (1.0, 2.0)


def test_model_available_ages():

    iso1 = make_isochrone()

    iso2 = Isochrone(

        age=5e9,

        metallicity=0.0,

        mass=iso1.mass,

        log_teff=iso1.log_teff,

        log_luminosity=iso1.log_luminosity,

        log_radius=iso1.log_radius,

        log_g=iso1.log_g

    )

    model = EvolutionModel(

        name="PARSEC",

        tracks=(),

        isochrones=(iso2, iso1)

    )

    assert model.available_ages == (1e9, 5e9)