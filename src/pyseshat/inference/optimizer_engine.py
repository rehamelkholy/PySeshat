import numpy as np

from scipy.optimize import least_squares

from pyseshat.inference.covariance import (
    estimate_covariance,
    parameter_uncertainties
)


# ==========================================================
# Generic optimization engine
# ==========================================================

def run_least_squares(

    *,
    initial_parameters,
    residual_function,
    residual_args,
    solution_builder,
    parameter_names,
    fit_result_class,
    method="trf",
    loss="linear",
    f_scale=1.0,
    bounds=None,
    jac="2-point"
):
    """
    Generic least-squares optimization engine.

    This function performs the common optimization workflow
    shared by all orbit fitters.

    Parameters
    ----------
    initial_parameters : ndarray
        Initial parameter vector.

    residual_function : callable
        Residual function compatible with
        scipy.optimize.least_squares.

    residual_args : tuple
        Arguments passed to residual_function.

    solution_builder : callable
        Function that converts an optimized parameter vector
        into the corresponding fitted orbital-element object.

    parameter_names : list[str]
        Names of fitted parameters.

    fit_result_class : type
        Class used to construct the returned fit result.

    method : str
        SciPy least-squares method.

    loss : str
        Loss function.

    f_scale : float
        Robust loss scale.

    bounds : tuple(ndarray, ndarray) or None
        Lower and upper parameter bounds.

    jac : {'2-point', '3-point', 'cs'} or callable
        Jacobian specification passed directly to SciPy.

    Returns
    -------
    FitResult-like object
    """

    # ------------------------------------------------------
    # Optimization
    # ------------------------------------------------------

    if bounds is None:

        bounds_for_optimizer = (
            (-float("inf"), float("inf"))
        )

    else:

        bounds_for_optimizer = bounds

    # ------------------------------------------------------
    # SciPy's LM method does not support finite bounds.
    #
    # ORBIT3 itself is unbounded, so an ORBIT3 LM call
    # reaches this branch with infinite bounds.
    # ------------------------------------------------------

    if method == "lm":

        lower, upper = bounds_for_optimizer

        if np.any(
            np.isfinite(lower)
        ) or np.any(
            np.isfinite(upper)
        ):

            raise ValueError(
                "SciPy method='lm' does not support "
                "finite parameter bounds."
            )

        bounds_for_optimizer = (
            (-float("inf"), float("inf"))
        )

    result = least_squares(

        residual_function,

        initial_parameters,

        args=residual_args,

        method=method,

        jac=jac,

        loss=loss,

        f_scale=f_scale,

        bounds=bounds_for_optimizer

    )

    # ------------------------------------------------------
    # Build fitted solution
    # ------------------------------------------------------

    fitted_elements = solution_builder(
        result.x
    )

    # ------------------------------------------------------
    # Final residual vector
    # ------------------------------------------------------

    residuals = residual_function(
        result.x,
        *residual_args
    )

    # ------------------------------------------------------
    # Covariance
    # ------------------------------------------------------

    covariance = estimate_covariance(
        result,
        residuals
    )

    # ------------------------------------------------------
    # Parameter uncertainties
    # ------------------------------------------------------

    uncertainties = parameter_uncertainties(
        covariance,
        parameter_names
    )

    # ------------------------------------------------------
    # FitResult
    # ------------------------------------------------------

    return fit_result_class(

        elements=fitted_elements,

        optimizer_result=result,

        residuals=residuals,

        covariance=covariance,

        uncertainties=uncertainties

    )