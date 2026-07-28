import numpy as np

from pyseshat.observations import VisualObservationSet
from pyseshat.observations.adapters import visual_arrays


def test_visual_arrays():

    obs = VisualObservationSet(
        epoch=np.array([1.0]),
        x=np.array([0.1]),
        y=np.array([0.2]),
        sigma_x=np.array([0.01]),
        sigma_y=np.array([0.02]),
    )

    t, x, y, sx, sy = visual_arrays(obs)

    assert np.array_equal(t, obs.epoch)
    assert np.array_equal(x, obs.x)
    assert np.array_equal(y, obs.y)
    assert np.array_equal(sx, obs.sigma_x)
    assert np.array_equal(sy, obs.sigma_y)