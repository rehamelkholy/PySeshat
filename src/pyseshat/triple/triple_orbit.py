import numpy as np

from pyseshat.orbit.orbit_model import (
    compute_visual_orbit
)

from pyseshat.triple.triple_elements import (
    TripleOrbitalElements
)


def compute_triple_orbit(
    epochs,
    elements: TripleOrbitalElements
):
    """
    Compute hierarchical triple orbit.

    Returns
    -------
    dict
    """

    outer = compute_visual_orbit(
        epochs,
        elements.outer
    )

    inner = compute_visual_orbit(
        epochs,
        elements.inner
    )

    X = (
        outer["X"]
        +
        elements.scale_inner
        * inner["X"]
    )

    Y = (
        outer["Y"]
        +
        elements.scale_inner
        * inner["Y"]
    )

    rho = np.sqrt(
        X**2 + Y**2
    )

    theta = (
        np.degrees(
            np.arctan2(Y, X)
        )
        % 360.0
    )

    return {

        "X": X,
        "Y": Y,

        "rho": rho,
        "theta": theta,

        "outer": outer,
        "inner": inner
    }