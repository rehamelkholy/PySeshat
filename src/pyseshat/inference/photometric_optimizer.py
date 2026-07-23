import numpy as np

from pyseshat.inference.optimizer_engine import (
    run_least_squares
)

from pyseshat.inference.base_fit_result import (
    BaseFitResult
)

from pyseshat.inference.photometric_likelihood import (
    photometric_residuals
)

from pyseshat.photometry.parameters import (
    update_parameters
)


# ==========================================================
# Physical parameter bounds
# ==========================================================

def build_parameter_bounds():
    """
    Construct physically reasonable bounds for the
    photometric inference parameters.

    Returns
    -------
    tuple(ndarray, ndarray)
        Lower and upper parameter bounds.
    """

    lower = np.array(
        [
            2500.0,     # Primary Teff [K]
            2500.0,     # Secondary Teff [K]
            0.01,       # Primary radius [Rsun]
            0.01,       # Secondary radius [Rsun]
            0.0         # E(B-V)
        ],
        dtype=float
    )

    upper = np.array(
        [
            50000.0,    # Primary Teff
            50000.0,    # Secondary Teff
            100.0,      # Primary radius
            100.0,      # Secondary radius
            10.0        # E(B-V)
        ],
        dtype=float
    )

    return lower, upper


# ==========================================================
# Photometric fitting
# ==========================================================

def fit_photometry(
    parameters,
    atmosphere_grid,
    passbands,
    zero_points,
    measurements,
    primary_logg,
    secondary_logg,
    metallicity,
    distance,
    rv=3.1,
    method="trf",
    loss="linear",
    f_scale=1.0,
    bounds=None
):
    """
    Fit unresolved-binary photometry using nonlinear
    least squares.

    Parameters
    ----------
    parameters : PhotometricParameters
        Initial parameter estimates.

    atmosphere_grid
        Stellar atmosphere grid.

    passbands : mapping
        Photometric passbands.

    zero_points : mapping
        Photometric zero points.

    measurements : iterable
        Photometric observations.

    primary_logg : float

    secondary_logg : float

    metallicity : float

    distance : float

    rv : float, default=3.1

    method : str
        SciPy least-squares method.

    loss : str
        Robust loss function.

    f_scale : float
        Robust loss scale.

    bounds : tuple(ndarray, ndarray), optional
        Optimizer bounds.

    Returns
    -------
    BaseFitResult
    """

    p0 = parameters.fitted_parameters()

    if bounds is None:

        bounds = build_parameter_bounds()

    residual_args = (

        atmosphere_grid,

        passbands,

        zero_points,

        measurements,

        primary_logg,

        secondary_logg,

        metallicity,

        distance,

        rv

    )

    return run_least_squares(

        initial_parameters=p0,

        residual_function=photometric_residuals,

        residual_args=residual_args,

        solution_builder=lambda values:

            update_parameters(
                parameters,
                values
            ),

        parameter_names=(
            parameters.fitted_parameter_names()
        ),

        fit_result_class=BaseFitResult,

        method=method,

        loss=loss,

        f_scale=f_scale,

        bounds=bounds

    )