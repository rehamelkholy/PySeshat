import numpy as np

import pytest

from pyseshat.evolution.models import EvolutionTrack
from pyseshat.evolution.interpolate import sample_track
from pyseshat.evolution.observables import StellarObservables

from pyseshat.evolution.age import (
    track_chi2,
    track_residuals,
)

from pyseshat.inference.evolution_likelihood import (
    evolution_log_likelihood,
)

from pyseshat.inference.photometric_likelihood import (
    chi_square,
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


def test_track_residuals_and_chi_square_are_consistent():

    sample = sample_track(
        make_track(),
        2.0,
    )

    observables = StellarObservables(
        log_teff=3.80,
        sigma_log_teff=0.01,
        log_luminosity=0.00,
        sigma_log_luminosity=0.01,
    )

    residuals = track_residuals(
        sample,
        observables,
    )

    assert np.isclose(
        track_chi2(
            sample,
            observables,
        ),
        chi_square(
            residuals,
        ),
    )


def test_evolution_log_likelihood_matches_chi_square():

    sample = sample_track(
        make_track(),
        2.0,
    )

    observables = StellarObservables(
        log_teff=3.80,
        sigma_log_teff=0.01,
        log_luminosity=0.00,
        sigma_log_luminosity=0.01,
    )

    chi2 = track_chi2(
        sample,
        observables,
    )

    assert np.isclose(
        evolution_log_likelihood(
            sample,
            observables,
        ),
        -0.5 * chi2,
    )


def test_missing_uncertainty_raises():

    sample = sample_track(
        make_track(),
        2.0,
    )

    observables = StellarObservables(
        log_teff=3.80,
        sigma_log_teff=None,
    )

    with pytest.raises(ValueError):

        evolution_log_likelihood(
            sample,
            observables,
        )