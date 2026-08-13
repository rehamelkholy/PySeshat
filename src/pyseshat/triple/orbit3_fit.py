import numpy as np


from pyseshat.inference.optimizer_engine import (
    run_least_squares
)


from pyseshat.triple.orbit3_bounds import (
    build_orbit3_bounds
)


from pyseshat.triple.triple_fit_result import (
    TripleFitResult
)


from pyseshat.triple.triple_fit import (
    orbit3_to_array,
    update_orbit3_fitted_parameters,
)


from pyseshat.triple.triple_residuals import (
    orbit3_residuals
)


# ==========================================================
# ORBIT3 residual wrapper
# ==========================================================

def orbit3_fit_flag_residuals(
    fitted_values,
    elements,
    observations,
    rv_observations
):
    """
    Construct an ORBIT3 element set from the currently
    fitted parameters and return the complete weighted
    residual vector.

    Parameters
    ----------
    fitted_values : ndarray
        Values corresponding only to parameters whose
        ORBIT3 fit flags are True.

    elements : TripleOrbitalElements
        Template ORBIT3 element set.

    observations : list[TripleObservation]
        Visual observations.

    rv_observations : list[TripleObservation]
        Radial-velocity observations.

    Returns
    -------
    ndarray
        Combined weighted ORBIT3 residual vector.
    """

    updated = (
        update_orbit3_fitted_parameters(

            elements,

            fitted_values

        )
    )


    return orbit3_residuals(

        observations,

        rv_observations,

        updated

    )


# ==========================================================
# ORBIT3 fit
# ==========================================================

def fit_orbit3(
    elements,
    observations,
    rv_observations,
    method="trf",
    loss="linear",
    f_scale=1.0,
    bounds=None
):
    """
    Fit a hierarchical triple system using the modern
    ORBIT3 formulation.

    Parameters
    ----------
    elements : TripleOrbitalElements
        Initial ORBIT3 orbital and spectroscopic elements.

    observations : list[TripleObservation]
        Visual observations, normally tagged I1 or I2.

    rv_observations : list[TripleObservation]
        Radial-velocity observations, normally tagged Va,
        Vb, or Vc.

    method : str, optional
        Least-squares method passed to SciPy through the
        generic optimization engine.

    loss : str, optional
        Loss function passed to scipy.optimize.least_squares.

    f_scale : float, optional
        Scale parameter for robust loss functions.

    bounds : tuple(ndarray, ndarray), optional
        User-supplied bounds. If omitted, physically
        admissible ORBIT3 bounds are constructed from the
        fit flags.

    Returns
    -------
    TripleFitResult
        Optimized ORBIT3 solution.

    Notes
    -----
    This function is intentionally separate from
    ``fit_triple_orbit()``.

    ``fit_triple_orbit()`` remains the validated legacy
    ORBITT2 fitter.

    ``fit_orbit3()`` uses the complete 20-parameter ORBIT3
    representation and the combined visual + radial-velocity
    residual model.
    """

    p0 = (
        orbit3_to_array(
            elements
        )
    )[
        elements.fitted_mask()
    ]


    if bounds is None:

        bounds = build_orbit3_bounds(
            elements
        )


    lower, upper = bounds


    if len(lower) != len(p0):

        raise ValueError(

            "Number of lower bounds does not match "
            "the number of fitted ORBIT3 parameters: "
            f"{len(lower)} != {len(p0)}."

        )


    if len(upper) != len(p0):

        raise ValueError(

            "Number of upper bounds does not match "
            "the number of fitted ORBIT3 parameters: "
            f"{len(upper)} != {len(p0)}."

        )


    if np.any(
        p0 < lower
    ):

        raise ValueError(

            "Initial ORBIT3 parameter vector contains "
            "values below the supplied lower bounds."

        )


    if np.any(
        p0 > upper
    ):

        raise ValueError(

            "Initial ORBIT3 parameter vector contains "
            "values above the supplied upper bounds."

        )


    return run_least_squares(

        initial_parameters=p0,

        residual_function=(
            orbit3_fit_flag_residuals
        ),

        residual_args=(

            elements,

            observations,

            rv_observations

        ),

        solution_builder=lambda p:

            update_orbit3_fitted_parameters(

                elements,

                p

            ),

        parameter_names=(

            elements.fitted_parameter_names()

        ),

        fit_result_class=TripleFitResult,

        method=method,

        loss=loss,

        f_scale=f_scale,

        bounds=bounds

    )