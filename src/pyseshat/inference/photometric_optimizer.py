import numpy as np

from pyseshat.inference.optimizer_engine import (
    run_least_squares,
)

from pyseshat.inference.base_fit_result import (
    BaseFitResult,
)

from pyseshat.inference.photometric_likelihood import (
    photometric_residuals,
    photometric_residuals_from_problem,
)

from pyseshat.photometry.parameters import (
    update_parameters,
)

from pyseshat.inference.photometric_context import (
    PhotometricContext,
)

from pyseshat.inference.photometric_problem import (
    PhotometricInferenceProblem,
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
            2500.0,
            2500.0,
            0.01,
            0.01,
            0.0,
        ],
        dtype=float,
    )

    upper = np.array(
        [
            50000.0,
            50000.0,
            100.0,
            100.0,
            10.0,
        ],
        dtype=float,
    )

    return lower, upper


# ==========================================================
# Canonical photometric fitting
# ==========================================================

def fit_photometry_from_problem(
    problem,
    *,
    method="trf",
    loss="linear",
    f_scale=1.0,
    bounds=None,
):
    """
    Canonical least-squares interface for photometric
    inference.
    """

    parameters = problem.parameters

    if bounds is None:
        bounds = build_parameter_bounds()

    return run_least_squares(

        initial_parameters=parameters.fitted_parameters(),

        residual_function=photometric_residuals_from_problem,

        residual_args=(problem,),

        solution_builder=lambda values: PhotometricInferenceProblem(
            parameters=update_parameters(
                parameters,
                values,
            ),
            context=problem.context,
        ),

        parameter_names=(
            parameters.fitted_parameter_names()
        ),

        fit_result_class=BaseFitResult,

        method=method,

        loss=loss,

        f_scale=f_scale,

        bounds=bounds,
    )


# ==========================================================
# Legacy wrapper
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
    bounds=None,
):
    """
    Fit unresolved-binary photometry using nonlinear
    least squares.
    """

    problem = PhotometricInferenceProblem(

        parameters=parameters,

        context=PhotometricContext(

            atmosphere_grid=atmosphere_grid,

            passbands=passbands,

            zero_points=zero_points,

            measurements=measurements,

            primary_logg=primary_logg,

            secondary_logg=secondary_logg,

            metallicity=metallicity,

            distance=distance,

            rv=rv,

        ),
    )

    return fit_photometry_from_problem(

        problem,

        method=method,

        loss=loss,

        f_scale=f_scale,

        bounds=bounds,

    )