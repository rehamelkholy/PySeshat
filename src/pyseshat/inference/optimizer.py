import numpy as np

from scipy.optimize import least_squares

from pyseshat.inference.optimizer_engine import (
    run_least_squares
)
from pyseshat.inference.node_disambiguation import (
    node_flip
)

from pyseshat.inference.likelihood import (
    combined_residuals,
)

from pyseshat.inference.terms.visual import (
    visual_xy_residuals,
)

from pyseshat.inference.core import (
    update_elements,
)

from pyseshat.inference.fit_result import (
    FitResult
)

from pyseshat.inference.covariance import (
    estimate_covariance,
    parameter_uncertainties
)

from pyseshat.inference.posterior import (
    log_posterior
)

from pyseshat.inference.mcmc_engine import (
    run_emcee
)

from pyseshat.inference.mcmc_result import (
    MCMCResult
)

from pyseshat.analysis.statistics import (
    combined_fit_summary
)

from pyseshat.orbit.orbital_elements import OrbitalElements

# ==========================================================
# Parameter-aware physical bounds
# ==========================================================

def build_parameter_bounds(
    elements: OrbitalElements
):
    """
    Construct physical parameter bounds
    matching fitted parameter mask.

    Returns
    -------
    lower, upper : ndarray
    """

    # ------------------------------------------------------
    # Full parameter bounds in canonical order
    # ------------------------------------------------------

    lower_full = np.array([
        1e-6,      # P
        -np.inf,   # T0
        0.0,       # e
        0.0,       # a
        0.0,       # Omega
        0.0,       # omega
        0.0,       # i
        0.0,       # K1
        0.0,       # K2
        -np.inf    # V0
    ])

    upper_full = np.array([
        np.inf,    # P
        np.inf,    # T0
        0.99,      # e
        np.inf,    # a
        360.0,     # Omega
        360.0,     # omega
        180.0,     # i
        np.inf,    # K1
        np.inf,    # K2
        np.inf     # V0
    ])

    # ------------------------------------------------------
    # Apply fit mask
    # ------------------------------------------------------

    mask = elements.fitted_mask()

    lower = lower_full[mask]

    upper = upper_full[mask]

    return lower, upper

def _chi2_from_fit(
    fit_result
):
    """
    Compute chi-square from a FitResult.
    """

    return np.sum(
        fit_result.residuals**2
    )

# ==========================================================
# Visual orbit fitting
# ==========================================================

def fit_visual_orbit(
    elements,
    t,
    X_obs,
    Y_obs,
    sx,
    sy,
    rv_data=None,
    method="trf",
    loss="linear",
    f_scale=1.0,
    bounds=None,
    node_disambiguation=True,
    node_rtol=0.01
):
    """
    Public orbit-fitting interface.

    Automatically resolves visual-binary
    node ambiguity when RV data are present.
    """

    # --------------------------------------------------
    # Visual-only systems
    # --------------------------------------------------

    if (
        rv_data is None
        or len(rv_data) == 0
        or not node_disambiguation
    ):

        return _fit_single_solution(
            elements,
            t,
            X_obs,
            Y_obs,
            sx,
            sy,
            rv_data=rv_data,
            method=method,
            loss=loss,
            f_scale=f_scale,
            bounds=bounds
        )

    # --------------------------------------------------
    # Original node assignment
    # --------------------------------------------------

    fit_original = _fit_single_solution(
        elements,
        t,
        X_obs,
        Y_obs,
        sx,
        sy,
        rv_data=rv_data,
        method=method,
        loss=loss,
        f_scale=f_scale,
        bounds=bounds
    )

    # --------------------------------------------------
    # Flipped node assignment
    # --------------------------------------------------

    flipped = node_flip(
        elements
    )

    fit_flipped = _fit_single_solution(
        flipped,
        t,
        X_obs,
        Y_obs,
        sx,
        sy,
        rv_data=rv_data,
        method=method,
        loss=loss,
        f_scale=f_scale,
        bounds=bounds
    )

    # --------------------------------------------------
    # Compare final chi-square
    # --------------------------------------------------

    chi2_original = _chi2_from_fit(
        fit_original
    )

    chi2_flipped = _chi2_from_fit(
        fit_flipped
    )

    # --------------------------------------------------
    # Nearly equivalent solutions
    # --------------------------------------------------

    relative_difference = (
        abs(
            chi2_original
            -
            chi2_flipped
        )
        /
        min(
            chi2_original,
            chi2_flipped
        )
    )

    if (
        relative_difference
        <
        node_rtol
    ):

        return fit_original

    # --------------------------------------------------
    # Choose lower chi-square
    # --------------------------------------------------

    if (
        chi2_original
        <
        chi2_flipped
    ):

        return fit_original

    return fit_flipped

def _fit_single_solution(
    elements: OrbitalElements,
    t,
    X_obs,
    Y_obs,
    sx,
    sy,
    rv_data=None,
    method="trf",
    loss="linear",
    f_scale=1.0,
    bounds=None
):
    """
    Fit a single binary-orbit solution using the
    generic optimization engine.
    """

    # ------------------------------------------------------
    # Initial parameter vector
    # ------------------------------------------------------

    p0 = elements.fitted_parameters()

    # ------------------------------------------------------
    # Bounds
    # ------------------------------------------------------

    if bounds is None:

        bounds = build_parameter_bounds(
            elements
        )

    # ------------------------------------------------------
    # Residual function
    # ------------------------------------------------------

    if (
        rv_data is None
        or len(rv_data) == 0
    ):

        residual_function = (
            visual_xy_residuals
        )

        residual_args = (
            elements,
            t,
            X_obs,
            Y_obs,
            sx,
            sy
        )

    else:

        residual_function = (
            combined_residuals
        )

        residual_args = (
            elements,
            t,
            X_obs,
            Y_obs,
            sx,
            sy,
            rv_data
        )

    # ------------------------------------------------------
    # Generic optimization
    # ------------------------------------------------------

    return run_least_squares(

        initial_parameters=p0,

        residual_function=residual_function,

        residual_args=residual_args,

        solution_builder=lambda p: update_elements(
            elements,
            p
        ),

        parameter_names=(
            elements.fitted_parameter_names()
        ),

        fit_result_class=FitResult,

        method=method,

        loss=loss,

        f_scale=f_scale,

        bounds=bounds
    )

# ==========================================================
# Sample a single orbital solution
# ==========================================================

def _sample_single_solution(
    fit_result,
    t,
    X_obs,
    Y_obs,
    sx,
    sy,
    rv_data=None,
    n_walkers=32,
    n_steps=5000,
    burn_in=1000,
    thin=1,
    random_scale=1e-4,
    rng=None
):
    """
    Sample the posterior distribution of a single orbital
    solution using emcee.

    Parameters
    ----------
    fit_result : FitResult
        Least-squares solution used to initialize the sampler.

    Returns
    -------
    MCMCResult
    """

    elements = fit_result.elements

    initial_parameters = (
        elements.fitted_parameters()
    )

    engine = run_emcee(

        log_probability_function=log_posterior,

        log_probability_args=(

            elements,

            t,

            X_obs,

            Y_obs,

            sx,

            sy,

            rv_data

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

        elements=elements,

        optimizer_result=fit_result.optimizer_result,

        residuals=fit_result.residuals,

        covariance=fit_result.covariance,

        uncertainties=fit_result.uncertainties,

        samples=engine["samples"],

        log_probability=engine["log_probability"],

        parameter_names=elements.fitted_parameter_names(),

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

# ==========================================================
# Bayesian visual-orbit fitting
# ==========================================================

def fit_visual_orbit_mcmc(
    elements,
    t,
    X_obs,
    Y_obs,
    sx,
    sy,
    rv_data=None,
    method="trf",
    loss="linear",
    f_scale=1.0,
    bounds=None,
    node_disambiguation=True,
    node_rtol=0.01,
    n_walkers=32,
    n_steps=5000,
    burn_in=1000,
    thin=1,
    random_scale=1e-4,
    rng=None
):
    """
    Bayesian visual-orbit fitting.

    This function first computes the maximum-likelihood
    solution using least squares, then samples the posterior
    distribution using emcee.
    """

    fit = fit_visual_orbit(

        elements,

        t,

        X_obs,

        Y_obs,

        sx,

        sy,

        rv_data=rv_data,

        method=method,

        loss=loss,

        f_scale=f_scale,

        bounds=bounds,

        node_disambiguation=node_disambiguation,

        node_rtol=node_rtol
    )

    return _sample_single_solution(

        fit,

        t,

        X_obs,

        Y_obs,

        sx,

        sy,

        rv_data=rv_data,

        n_walkers=n_walkers,

        n_steps=n_steps,

        burn_in=burn_in,

        thin=thin,

        random_scale=random_scale,

        rng=rng
    )