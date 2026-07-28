import numpy as np

from .visual import (
    visual_xy_residuals,
)

from .radial_velocity import (
    rv_residuals,
)

from pyseshat.inference.context import (
    LikelihoodContext,
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

def joint_residuals_from_context(
    fitted_values,
    elements,
    context: LikelihoodContext,
):
    """
    Compute residuals from a LikelihoodContext.

    This is the preferred interface for new code.
    """

    return joint_residuals(
        fitted_values=fitted_values,
        elements=elements,
        t=context.t,
        X_obs=context.X_obs,
        Y_obs=context.Y_obs,
        sx=context.sx,
        sy=context.sy,
        rv_data=context.rv_data,
    )