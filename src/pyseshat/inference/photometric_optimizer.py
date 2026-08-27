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
# scipy-compatible residual wrapper
# ==========================================================
#
# photometric_residuals_from_problem(problem) takes a single,
# already-built PhotometricInferenceProblem -- it has no
# (values, *args) signature, because a "problem" already
# carries a specific parameter vector, not a trial one. But
# scipy.optimize.least_squares (via run_least_squares) always
# calls its residual_function as residual_function(x, *args),
# where x is the CURRENT trial parameter vector the optimizer
# is evaluating, and args is whatever was passed as
# residual_args. Passing photometric_residuals_from_problem
# straight through as residual_function (as this module
# previously did) means scipy calls it as
# photometric_residuals_from_problem(x, problem) -- two
# positional arguments against a one-argument function -- and
# every call raises TypeError before a single residual is ever
# evaluated. This was never caught because nothing exercised
# fit_photometry_from_problem()/fit_photometry() end-to-end
# before the Al-Wardat SED validation (see
# docs/project_decisions.md); every existing photometric test
# calls photometric_residuals_from_problem() directly with one
# fixed problem, which is a valid call and never hits this
# path. This wrapper rebuilds a fresh problem at the optimizer's
# current trial vector on every call, matching the convention
# every other residual function in this package already follows
# (e.g. inference.likelihood.visual_xy_residuals(parameters,
# elements, ...)).

def _photometric_residuals_for_optimizer(
    values,
    problem,
):
    """
    scipy-compatible residual function for photometric
    least-squares fitting: rebuild the trial
    PhotometricInferenceProblem at ``values`` and delegate to
    photometric_residuals_from_problem().
    """

    trial_problem = PhotometricInferenceProblem(

        parameters=update_parameters(
            problem.parameters,
            values,
        ),

        context=problem.context,

    )

    return photometric_residuals_from_problem(
        trial_problem
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

        residual_function=_photometric_residuals_for_optimizer,

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