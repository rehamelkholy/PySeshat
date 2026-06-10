import numpy as np


def orbital_phase(
    t,
    P,
    T0
):
    """
    Orbital phase in [0,1).
    """

    t = np.asarray(
        t,
        dtype=float
    )

    return (
        (t - T0) / P
    ) % 1.0