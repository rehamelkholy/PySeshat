import numpy as np

from .visual import (
    visual_xy_residuals,
)

from .radial_velocity import (
    rv_residuals,
)


def joint_residuals(
    fitted_values,
    elements,
    t,
    X_obs,
    Y_obs,
    sx,
    sy,
    rv_data=None,
):
    """
    Compute the combined residual vector from all
    enabled likelihood terms.
    """

    residuals = []

    residuals.append(

        visual_xy_residuals(

            fitted_values,

            elements,

            t,

            X_obs,

            Y_obs,

            sx,

            sy,

        )

    )

    if rv_data is not None and len(rv_data) > 0:

        residuals.append(

            rv_residuals(

                fitted_values,

                elements,

                rv_data,

            )

        )

    return np.concatenate(residuals)