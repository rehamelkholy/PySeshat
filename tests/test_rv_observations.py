import numpy as np

from pyseshat.observations import RVObservationSet


def test_rv_observation_set():

    obs = RVObservationSet(
        epoch=np.array([1.0, 2.0]),
        velocity=np.array([10.0, -12.0]),
        sigma=np.array([0.5, 0.5]),
        component=np.array([0, 1]),
    )

    assert obs.epoch.shape == (2,)
    assert obs.velocity.shape == (2,)
    assert obs.sigma.shape == (2,)
    assert obs.component.shape == (2,)