import numpy as np

from pyseshat.experiments.observation_set import (
    VisualObservationSet,
)
from pyseshat.experiments.synthetic import (
    perturb_values,
    perturb_xy,
)


def test_perturb_values():

    rng = np.random.default_rng(42)

    values = np.ones(10)

    perturbed = perturb_values(
        values,
        0.1,
        rng=rng,
    )

    assert perturbed.shape == values.shape

    assert not np.allclose(
        perturbed,
        values,
    )


def test_perturb_xy():

    observations = VisualObservationSet(
        epoch=np.arange(5.0),
        x=np.zeros(5),
        y=np.zeros(5),
        sigma_x=np.full(5, 0.01),
        sigma_y=np.full(5, 0.02),
    )

    result = perturb_xy(
        observations,
        rng=np.random.default_rng(1),
    )

    assert isinstance(
        result,
        VisualObservationSet,
    )

    assert result.x.shape == (5,)
    assert result.y.shape == (5,)

    assert np.any(result.x != observations.x)
    assert np.any(result.y != observations.y)

    np.testing.assert_array_equal(
        result.epoch,
        observations.epoch,
    )

    np.testing.assert_array_equal(
        result.sigma_x,
        observations.sigma_x,
    )

    np.testing.assert_array_equal(
        result.sigma_y,
        observations.sigma_y,
    )