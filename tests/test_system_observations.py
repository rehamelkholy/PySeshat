import numpy as np

from pyseshat.evolution.observables import StellarObservables
from pyseshat.observations import (
    RVObservationSet,
    SystemObservations,
    VisualObservationSet,
)


def test_empty_system():

    system = SystemObservations()

    assert system.visual is None
    assert system.radial_velocity is None
    assert system.stellar is None


def test_system_with_visual_only():

    visual = VisualObservationSet(
        epoch=np.array([1.0]),
        x=np.array([0.1]),
        y=np.array([0.2]),
        sigma_x=np.array([0.01]),
        sigma_y=np.array([0.01]),
    )

    system = SystemObservations(
        visual=visual,
    )

    assert system.visual is visual
    assert system.radial_velocity is None


def test_system_with_multiple_constraints():

    visual = VisualObservationSet(
        epoch=np.array([1.0]),
        x=np.array([0.1]),
        y=np.array([0.2]),
        sigma_x=np.array([0.01]),
        sigma_y=np.array([0.01]),
    )

    rv = RVObservationSet(
        epoch=np.array([1.0]),
        velocity=np.array([10.0]),
        sigma=np.array([0.5]),
        component=np.array([0]),
    )

    stellar = StellarObservables(
        log_teff=3.76,
        sigma_log_teff=0.01,
    )

    system = SystemObservations(
        visual=visual,
        radial_velocity=rv,
        stellar=stellar,
    )

    assert system.visual is visual
    assert system.radial_velocity is rv
    assert system.stellar is stellar