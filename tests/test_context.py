import numpy as np

from pyseshat.inference.context import (
    LikelihoodContext,
)


def test_context_stores_arrays():

    ctx = LikelihoodContext(
        t=np.array([1.0, 2.0]),
        X_obs=np.array([0.1, 0.2]),
        Y_obs=np.array([0.3, 0.4]),
        sx=np.array([0.01, 0.01]),
        sy=np.array([0.02, 0.02]),
    )

    np.testing.assert_array_equal(
        ctx.t,
        np.array([1.0, 2.0]),
    )

    np.testing.assert_array_equal(
        ctx.X_obs,
        np.array([0.1, 0.2]),
    )

    assert ctx.rv_data is None