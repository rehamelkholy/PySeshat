import numpy as np

from pyseshat.inference.core import (
    update_elements,
)

from pyseshat.orbit.orbit_model import (
    compute_visual_orbit,
)

from pyseshat.orbit.orbital_elements import (
    OrbitalElements,
)


def visual_xy_residuals(
    fitted_values,
    elements: OrbitalElements,
    t,
    X_obs,
    Y_obs,
    sx,
    sy,
):
    """
    Compute weighted visual binary residuals
    in Cartesian coordinates.

    Returns
    -------
    ndarray
        Concatenated weighted X/Y residuals.
    """

    updated_elements = update_elements(
        elements,
        fitted_values,
    )

    orbit = compute_visual_orbit(
        t,
        updated_elements,
    )

    X_model = orbit["X"]
    Y_model = orbit["Y"]

    resX = (X_obs - X_model) / sx
    resY = (Y_obs - Y_model) / sy

    return np.concatenate(
        [
            resX,
            resY,
        ]
    )