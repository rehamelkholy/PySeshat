import numpy as np

import pytest

from pyseshat.evolution.binary import (
    BinaryTrackFitResult,
    fit_binary_tracks,
)
from pyseshat.evolution.models import EvolutionTrack


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

    assert isinstance(result, BinaryTrackFitResult)

    assert result.primary.sample.age == pytest.approx(2.0)

    assert result.secondary.sample.age == pytest.approx(3.0)

    assert result.age_difference == pytest.approx(1.0)