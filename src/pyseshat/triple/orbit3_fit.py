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


from pyseshat.triple.orbit3_likelihood import (
    log_posterior_orbit3
)


from pyseshat.inference.mcmc_engine import (
    run_emcee
)


from pyseshat.inference.mcmc_result import (
    MCMCResult
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


# ==========================================================
# ORBIT3 Bayesian (MCMC) fit
# ==========================================================

def fit_orbit3_mcmc(
    elements,
    observations,
    rv_observations,
    method="lm",
    loss="linear",
    f_scale=1.0,
    bounds=None,
    n_walkers=32,
    n_steps=5000,
    burn_in=1000,
    thin=1,
    random_scale=1e-4,
    rng=None
):
    """
    Bayesian ORBIT3 hierarchical-triple fitting.

    This is the ORBIT3 counterpart of
    ``pyseshat.inference.optimizer.fit_visual_orbit_mcmc``: it
    first computes the maximum-likelihood ORBIT3 solution
    using ``fit_orbit3`` (deterministic least squares), then
    samples the posterior distribution of the fitted
    parameters using emcee, seeded from the least-squares
    solution and its covariance.

    Until this function, no MCMC/posterior-inference entry
    point existed for triple/ORBIT3 orbits at all --
    ``inference.mcmc_engine`` and ``inference.mcmc_result`` are
    fully generic (operating on a flat parameter vector and a
    caller-supplied log-probability function) and needed no
    changes; only an ORBIT3-specific log-posterior
    (``triple.orbit3_likelihood.log_posterior_orbit3``, itself
    built on ``triple.orbit3_priors.log_prior_orbit3`` and
    ``triple.triple_residuals.orbit3_residuals``) was missing.

    Parameters
    ----------
    elements : TripleOrbitalElements
        Initial ORBIT3 element set, including the fit mask
        (``elements.fit_flags``) that determines which of the
        20 ORBIT3 parameters are fitted.

    observations : list[TripleObservation]
        Astrometric (I1/I2) observations.

    rv_observations : list[TripleObservation]
        Radial-velocity (Va/Vb/Vc) observations. Pass an empty
        list for systems with no usable RV data (e.g. ZetaAqr).

    method, loss, f_scale, bounds
        Passed through to the initial ``fit_orbit3`` call.

    n_walkers, n_steps, burn_in, thin, random_scale, rng
        Passed through to ``inference.mcmc_engine.run_emcee``.
        See that function's docstring for details.

    Returns
    -------
    MCMCResult
        Posterior samples and diagnostics, plus the underlying
        least-squares solution (elements, residuals,
        covariance, uncertainties) inherited from
        ``BaseFitResult``.
    """

    # ------------------------------------------------------
    # Maximum-likelihood solution
    # ------------------------------------------------------

    fit_result = fit_orbit3(

        elements,

        observations,

        rv_observations,

        method=method,

        loss=loss,

        f_scale=f_scale,

        bounds=bounds

    )

    fitted_elements = fit_result.elements

    initial_parameters = orbit3_to_array(
        fitted_elements
    )[
        fitted_elements.fitted_mask()
    ]

    # ------------------------------------------------------
    # Posterior sampling
    # ------------------------------------------------------

    engine = run_emcee(

        log_probability_function=log_posterior_orbit3,

        log_probability_args=(

            fitted_elements,

            observations,

            rv_observations

        ),

        initial_parameters=initial_parameters,

        covariance=fit_result.covariance,

        n_walkers=n_walkers,

        n_steps=n_steps,

        burn_in=burn_in,

        thin=thin,

        random_scale=random_scale,

        rng=rng

    )

    return MCMCResult(

        elements=fitted_elements,

        optimizer_result=fit_result.optimizer_result,

        residuals=fit_result.residuals,

        covariance=fit_result.covariance,

        uncertainties=fit_result.uncertainties,

        samples=engine["samples"],

        log_probability=engine["log_probability"],

        parameter_names=(
            fitted_elements.fitted_parameter_names()
        ),

        acceptance_fraction=engine[
            "acceptance_fraction"
        ],

        autocorrelation_time=engine[
            "autocorrelation_time"
        ],

        n_walkers=n_walkers,

        n_steps=n_steps,

        burn_in=burn_in,

        thin=thin

    )