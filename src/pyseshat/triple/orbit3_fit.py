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
# ORBIT3 finite-difference scales
# ==========================================================

def orbit3_parameter_scales(
    elements
):
    """
    Return the ORBIT3 finite-difference increment vector.

    The original ORBIT3 ``alleph`` routine constructs its
    ``del`` vector from the input element vector.

    These increments are retained in PySeshat so that the
    original ORBIT3 finite-difference Jacobian is reproduced.
    """

    full = orbit3_to_array(
        elements
    )

    e = 0.01

    return np.array([

        # --------------------------------------------------
        # Inner orbit
        # --------------------------------------------------

        e * full[0],       # P
        e * full[0],       # T0
        e,                 # e
        e * full[3],       # a
        1.0,               # Omega
        1.0,               # omega
        1.0,               # i

        # --------------------------------------------------
        # Inner spectroscopy / wobble
        # --------------------------------------------------

        e * full[7],       # K1
        e * full[8],       # K2
        e,                 # f

        # --------------------------------------------------
        # Outer orbit
        # --------------------------------------------------

        e * full[10],      # P2
        e * full[11],      # T02
        e,                 # e2
        e * full[13],      # a2
        1.0,               # Omega2
        1.0,               # omega2
        1.0,               # i2

        # --------------------------------------------------
        # Outer spectroscopy
        # --------------------------------------------------

        e * full[17],      # K3
        e * full[18],      # K4
        e * full[17],      # V0

    ], dtype=float)


# ==========================================================
# ORBIT3 finite-difference Jacobian
# ==========================================================

def orbit3_jacobian(
    fitted_values,
    elements,
    observations,
    rv_observations
):
    """
    Construct the ORBIT3 forward finite-difference Jacobian.

    The derivative for each fitted parameter is calculated as

        [R(p + delta) - R(p)] / delta

    using the finite-difference increments defined by
    orbit3_parameter_scales().
    """

    fitted_values = np.asarray(
        fitted_values,
        dtype=float
    )

    mask = np.asarray(
        elements.fitted_mask(),
        dtype=bool
    )

    if mask.size != 20:

        raise ValueError(
            "ORBIT3 fit mask must contain exactly "
            f"20 entries; received {mask.size}."
        )

    expected = int(
        np.count_nonzero(mask)
    )

    if fitted_values.size != expected:

        raise ValueError(
            "The number of fitted values does not match "
            "the number of fitted ORBIT3 parameters: "
            f"{fitted_values.size} != {expected}."
        )

    # ------------------------------------------------------
    # Current ORBIT3 element vector
    # ------------------------------------------------------

    current_elements = (
        update_orbit3_fitted_parameters(
            elements,
            fitted_values
        )
    )

    current_full = orbit3_to_array(
        current_elements
    )

    # ------------------------------------------------------
    # ORBIT3 finite-difference increments
    #
    # These are calculated from the original element set.
    # ------------------------------------------------------

    delta_full = orbit3_parameter_scales(
        elements
    )

    fitted_indices = np.flatnonzero(
        mask
    )

    # ------------------------------------------------------
    # Current residual vector
    # ------------------------------------------------------

    residual0 = orbit3_fit_flag_residuals(
        fitted_values,
        elements,
        observations,
        rv_observations
    )

    jacobian = np.empty(
        (
            residual0.size,
            fitted_values.size
        ),
        dtype=float
    )

    # ------------------------------------------------------
    # Forward finite differences
    # ------------------------------------------------------

    for column, full_index in enumerate(
        fitted_indices
    ):

        delta = (
            delta_full[
                full_index
            ]
        )

        if delta == 0.0:

            raise ValueError(
                "ORBIT3 finite-difference increment is zero "
                f"for parameter index {full_index}."
            )

        trial_full = current_full.copy()

        trial_full[
            full_index
        ] += delta

        trial_fitted = (
            trial_full[
                mask
            ]
        )

        residual_trial = (
            orbit3_fit_flag_residuals(
                trial_fitted,
                elements,
                observations,
                rv_observations
            )
        )

        jacobian[
            :,
            column
        ] = (
            residual_trial
            -
            residual0
        ) / delta

    return jacobian


# ==========================================================
# ORBIT3 fit
# ==========================================================

def fit_orbit3(
    elements,
    observations,
    rv_observations,
    method="lm",
    loss="linear",
    f_scale=1.0,
    bounds=None
):
    """
    Fit a hierarchical triple system using the ORBIT3
    formulation.

    The forward model, parameterization, fit mask,
    residual ordering, and ORBIT3 finite-difference
    Jacobian are retained.

    The LM path uses the translated ORBIT3 finite-difference
    Jacobian rather than SciPy's default local numerical
    Jacobian.

    Non-LM methods use SciPy's local numerical Jacobian.
    """

    # ------------------------------------------------------
    # Initial fitted parameter vector
    # ------------------------------------------------------

    p0 = (
        orbit3_to_array(
            elements
        )
    )[
        elements.fitted_mask()
    ]

    # ------------------------------------------------------
    # Optimizer bounds
    # ------------------------------------------------------

    if method == "lm":

        if bounds is not None:

            lower, upper = bounds

            lower = np.asarray(
                lower,
                dtype=float
            )

            upper = np.asarray(
                upper,
                dtype=float
            )

            if np.any(
                np.isfinite(lower)
            ):

                raise ValueError(
                    "ORBIT3 method='lm' is unbounded and "
                    "cannot use finite lower bounds."
                )

            if np.any(
                np.isfinite(upper)
            ):

                raise ValueError(
                    "ORBIT3 method='lm' is unbounded and "
                    "cannot use finite upper bounds."
                )

        optimizer_bounds = None

    else:

        if bounds is None:

            optimizer_bounds = (
                build_orbit3_bounds(
                    elements
                )
            )

        else:

            optimizer_bounds = bounds

        lower, upper = optimizer_bounds

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

    # ------------------------------------------------------
    # Jacobian
    # ------------------------------------------------------
    #
    # ORBIT3 uses its own forward finite-difference
    # construction for the differential-correction path.
    #
    # Therefore the translated ORBIT3 Jacobian is supplied
    # to the LM optimizer.
    #
    # Non-LM methods retain SciPy's numerical Jacobian.
    # ------------------------------------------------------

    if method == "lm":

        jacobian = orbit3_jacobian

    else:

        jacobian = "2-point"

    # ------------------------------------------------------
    # Optimization
    # ------------------------------------------------------

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

        bounds=optimizer_bounds,

        jac=jacobian

    )