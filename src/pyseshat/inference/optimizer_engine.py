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
    bounds=None
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
        Residual function compatible with scipy.optimize.least_squares.

    residual_args : tuple
        Arguments passed to residual_function.

    solution_builder : callable
        Function that converts an optimized parameter vector
        into the corresponding fitted orbital-element object.

    template_elements : object
        Template element object used by update_function.

    parameter_names : list[str]
        Names of fitted parameters.

    fit_result_class : type
        Class used to construct the returned fit result.

    method : str
        SciPy least-squares method.

    loss : str
        Robust loss function.

    f_scale : float
        Robust loss scale.

    bounds : tuple(ndarray, ndarray)
        Lower and upper parameter bounds.

    Returns
    -------
    FitResult-like object
    """

    # ------------------------------------------------------
    # Optimization
    # ------------------------------------------------------

    result = least_squares(
        residual_function,
        initial_parameters,
        args=residual_args,
        method=method,
        loss=loss,
        f_scale=f_scale,
        bounds=bounds
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