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
    photometric_residuals_with_priors_from_problem,
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

    return photometric_residuals_with_priors_from_problem(
        trial_problem
    )


# ==========================================================
# Physical parameter bounds
# ==========================================================
#
# Default (generic) physical bounds per parameter name. Used for
# every FITTED parameter that doesn't have a more specific range
# supplied. Temperature bounds default to these generic limits
# too, but ``build_parameter_bounds()`` will use an
# ``atmosphere_grid``'s actual loaded Teff range instead when one
# is given -- the optimizer's initial guess must always fall
# inside its own bounds, and a problem's atmosphere_grid is
# almost always a small subset of the full CK04 grid (see
# validation/benchmark_runner.py), not the full 2500-50000K
# range.

_DEFAULT_PHYSICAL_BOUNDS = {
    "primary_temperature": (2500.0, 50000.0),
    "secondary_temperature": (2500.0, 50000.0),
    "primary_radius": (0.01, 100.0),
    "secondary_radius": (0.01, 100.0),
    "ebv": (0.0, 10.0),
}


def build_parameter_bounds(
    parameters=None,
    atmosphere_grid=None,
):
    """
    Construct physically reasonable bounds for the photometric
    inference parameters.

    Parameters
    ----------
    parameters : PhotometricParameters, optional
        If given, bounds are built ONLY for
        ``parameters.fitted_parameter_names()``, in that order --
        matching the shape of ``parameters.fitted_parameters()``,
        which is what ``run_least_squares`` actually receives as
        ``initial_parameters``. If omitted (the pre-fit-flags
        default), bounds are built for all five parameters, in
        canonical order -- the original, unconditional behavior.

    atmosphere_grid : AtmosphereGrid, optional
        If given, ``primary_temperature``/``secondary_temperature``
        bounds are taken from this grid's own loaded
        ``min(temperatures)``/``max(temperatures)`` instead of the
        generic 2500-50000K physical range, so a fitted
        temperature's initial guess (the system's literature
        value) is guaranteed to fall inside its own bounds. This
        replaces the equivalent ad hoc ``teff_min``/``teff_max``
        logic that used to live in
        ``validation/benchmark_runner.py::run_alwardat_benchmark()``.

    Returns
    -------
    tuple(ndarray, ndarray)
        Lower and upper parameter bounds, sized and ordered to
        match ``parameters.fitted_parameters()`` (or all five
        parameters, canonical order, if ``parameters`` is
        omitted).
    """

    if parameters is None:

        names = list(
            _DEFAULT_PHYSICAL_BOUNDS.keys()
        )

    else:

        names = parameters.fitted_parameter_names()

    if atmosphere_grid is not None:

        teff_bounds = (
            float(min(atmosphere_grid.temperatures)),
            float(max(atmosphere_grid.temperatures)),
        )

    else:

        teff_bounds = None

    lower = []

    upper = []

    for name in names:

        if (
            teff_bounds is not None
            and name in (
                "primary_temperature",
                "secondary_temperature",
            )
        ):

            bound_lower, bound_upper = teff_bounds

        else:

            bound_lower, bound_upper = (
                _DEFAULT_PHYSICAL_BOUNDS[name]
            )

        lower.append(bound_lower)

        upper.append(bound_upper)

    return (
        np.array(lower, dtype=float),
        np.array(upper, dtype=float),
    )


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
        bounds = build_parameter_bounds(
            parameters,
            atmosphere_grid=problem.context.atmosphere_grid,
        )

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