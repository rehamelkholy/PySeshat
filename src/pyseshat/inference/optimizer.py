import numpy as np

from pyseshat.inference.context import LikelihoodContext

from pyseshat.inference.optimizer_engine import (
    run_least_squares
)

from pyseshat.inference.node_disambiguation import (
    node_flip
)

from pyseshat.inference.likelihood import (
    combined_residuals,
)

from pyseshat.inference.problem import InferenceProblem

from pyseshat.inference.terms.visual import (
    visual_xy_residuals,
)

from pyseshat.inference.core import (
    update_elements,
)

from pyseshat.inference.fit_result import (
    FitResult
)

from pyseshat.inference.posterior import (
    log_posterior,
    log_posterior_from_problem,
)

from pyseshat.inference.mcmc_engine import (
    run_emcee
)

from pyseshat.inference.mcmc_result import (
    MCMCResult
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

    context = LikelihoodContext(
        t=t,
        X_obs=X_obs,
        Y_obs=Y_obs,
        sx=sx,
        sy=sy,
        rv_data=rv_data,
    )

    problem = InferenceProblem(
        elements=elements,
        context=context,
        method=method,
        loss=loss,
        f_scale=f_scale,
        bounds=bounds,
        node_disambiguation=node_disambiguation,
        node_rtol=node_rtol,
    )

    return fit_visual_orbit_from_problem(
        problem
    )

# ==========================================================
# Internal optimizer implementation
# ==========================================================

def _fit_visual_orbit_impl(
    problem,
):
    """
    Shared implementation for visual-orbit fitting.

    Parameters
    ----------
    problem : InferenceProblem

    Returns
    -------
    FitResult
    """

    elements = problem.elements

    t = problem.t

    X_obs = problem.X_obs

    Y_obs = problem.Y_obs

    sx = problem.sx

    sy = problem.sy

    rv_data = problem.rv_data

    method = problem.method

    loss = problem.loss

    f_scale = problem.f_scale

    bounds = problem.bounds

    node_disambiguation = problem.node_disambiguation

    node_rtol = problem.node_rtol

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
            bounds=bounds,
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
        bounds=bounds,
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
        bounds=bounds,
    )

    chi2_original = _chi2_from_fit(
        fit_original
    )

    chi2_flipped = _chi2_from_fit(
        fit_flipped
    )

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

    if relative_difference < node_rtol:

        return fit_original

    if chi2_original < chi2_flipped:

        return fit_original

    return fit_flipped

def fit_visual_orbit_from_problem(
    problem,
):
    """
    Preferred orbit-fitting interface operating on an
    InferenceProblem.
    """

    return _fit_visual_orbit_impl(
        problem
    )

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
# Angle wrapping for periodic MCMC parameters
# ==========================================================

# Fitted parameters that are periodic with period 360 deg under
# this codebase's own prior (see priors.py's log_prior:
# ``log_uniform_prior(elements.Omega, 0.0, 360.0)`` and the same
# for ``omega``) and are wrapped into [0, 360) by
# ``OrbitalElements.normalize_angles()`` every time
# ``inference/core.py``'s ``update_elements()`` builds the
# elements object the prior/likelihood actually evaluate.
_PERIODIC_360_PARAMETERS = ("Omega", "omega")


def _wrap_periodic_mcmc_samples(samples, parameter_names):
    """
    Wrap posterior samples for periodic angular parameters
    (``Omega``, ``omega``) into their canonical [0, 360) range.

    Why this exists (found 2026-09-02, Capella corner-plot
    investigation)
    --------------------------------------------------------------
    ``update_elements()`` calls ``normalize_angles()`` on the
    reconstructed ``OrbitalElements`` BEFORE ``log_prior``/
    ``log_likelihood`` ever see it -- so, as far as the posterior
    DENSITY is concerned, ``Omega`` and ``omega`` are exactly
    360-deg-periodic coordinates: a raw walker position of
    omega=770 deg and omega=50 deg are the physically identical
    point and receive the identical log-posterior. But ``emcee``
    itself only ever stores the RAW, pre-normalization position it
    proposed -- nothing in the sampling loop folds a walker's raw
    Omega/omega coordinate back into [0, 360) between steps. For a
    tightly-constrained angle (Omega, in every benchmark system
    exercised so far) this is harmless in practice: the likelihood
    gradient is steep enough that no accepted step ever crosses a
    full period. For a WEAKLY constrained angle -- omega on a
    near-circular orbit (e -> 0), where the periastron reference
    point and therefore omega itself is only loosely informative --
    walkers can and do random-walk through many equivalent periodic
    images with no restoring force to stop them. Confirmed on
    Capella's real MCMC chain before this fix: raw ``omega`` samples
    spanned roughly [-72500, +79000] degrees (up to ~220 full
    periods), continuously across the whole post-burn-in chain, not
    a handful of early pre-convergence outliers -- while the
    physically-identical WRAPPED distribution is a normal, tightly
    bounded posterior around ~285 deg, exactly matching what the
    marginal histogram already showed.

    Wrapping here is the correct fix, not a cosmetic one, precisely
    because the log-posterior is exactly invariant under a +/-360
    deg shift in these coordinates: mapping a raw sample to its
    representative value in [0, 360) does not discard, reject, or
    reweight anything, and does not change the sampled posterior
    distribution in any way -- it only removes an accounting
    artifact of the *unwrapped* coordinate emcee happens to store.
    Applied once here, immediately after ``run_emcee`` returns the
    flattened chain, the wrapped range becomes part of
    ``MCMCResult.samples`` itself (and therefore of every saved
    ``.npz`` chain going forward), so every downstream consumer --
    ``corner_plot()``, ``posterior_median``/``credible_interval_16``/
    ``credible_interval_84``/``posterior_std``, ``maximum_posterior_sample``
    -- sees the physically meaningful range without its own
    workaround.

    Parameters
    ----------
    samples : ndarray, shape (n_samples, n_parameters)
        Flattened posterior chain, in the same column order as
        ``parameter_names``.

    parameter_names : sequence of str

    Returns
    -------
    ndarray
        A new array (the input is not modified in place) with
        every column named in ``_PERIODIC_360_PARAMETERS`` wrapped
        via ``np.mod(column, 360.0)``. Columns for every other
        parameter are returned unchanged.
    """

    samples = np.array(
        samples,
        dtype=float,
        copy=True
    )

    parameter_names = list(
        parameter_names
    )

    for name in _PERIODIC_360_PARAMETERS:

        if name in parameter_names:

            idx = parameter_names.index(
                name
            )

            samples[:, idx] = np.mod(
                samples[:, idx],
                360.0
            )

    return samples


def _sample_single_solution_from_problem(
    fit_result,
    problem,
    n_walkers=32,
    n_steps=5000,
    burn_in=1000,
    thin=1,
    random_scale=1e-4,
    rng=None
):
    """
    Sample the posterior distribution of a single orbital
    solution using an InferenceProblem.
    """

    elements = fit_result.elements

    initial_parameters = (
        elements.fitted_parameters()
    )

    engine = run_emcee(

        log_probability_function=log_posterior_from_problem,

        log_probability_args=(
            problem,
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

    parameter_names = elements.fitted_parameter_names()

    # See _wrap_periodic_mcmc_samples's own docstring: Omega/omega
    # are periodic in the posterior emcee actually samples, but the
    # raw chain emcee returns is NOT wrapped -- fold it into
    # [0, 360) here, once, so every downstream consumer of
    # MCMCResult.samples sees the physically meaningful range.
    wrapped_samples = _wrap_periodic_mcmc_samples(
        engine["samples"],
        parameter_names
    )

    return MCMCResult(

        elements=elements,

        optimizer_result=fit_result.optimizer_result,

        residuals=fit_result.residuals,

        covariance=fit_result.covariance,

        uncertainties=fit_result.uncertainties,

        samples=wrapped_samples,

        log_probability=engine["log_probability"],

        parameter_names=parameter_names,

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

    context = LikelihoodContext(

        t=t,

        X_obs=X_obs,

        Y_obs=Y_obs,

        sx=sx,

        sy=sy,

        rv_data=rv_data,
    )

    problem = InferenceProblem(

        elements=fit_result.elements,

        context=context,
    )

    return _sample_single_solution_from_problem(

        fit_result,

        problem,

        n_walkers=n_walkers,

        n_steps=n_steps,

        burn_in=burn_in,

        thin=thin,

        random_scale=random_scale,

        rng=rng,
    )

# ==========================================================
# Bayesian visual-orbit fitting from inference problem
# ==========================================================

def fit_visual_orbit_mcmc_from_problem(
    problem,
    n_walkers=32,
    n_steps=5000,
    burn_in=1000,
    thin=1,
    random_scale=1e-4,
    rng=None
):
    """
    Preferred Bayesian orbit-fitting interface operating on
    an InferenceProblem.

    This function first computes the maximum-likelihood
    solution, then samples the posterior using emcee.
    """

    fit = fit_visual_orbit_from_problem(
        problem
    )

    return _sample_single_solution(

        fit,

        problem.t,

        problem.X_obs,

        problem.Y_obs,

        problem.sx,

        problem.sy,

        rv_data=problem.rv_data,

        n_walkers=n_walkers,

        n_steps=n_steps,

        burn_in=burn_in,

        thin=thin,

        random_scale=random_scale,

        rng=rng
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

    context = LikelihoodContext(

        t=t,

        X_obs=X_obs,

        Y_obs=Y_obs,

        sx=sx,

        sy=sy,

        rv_data=rv_data
    )

    problem = InferenceProblem(

        elements=elements,

        context=context,

        method=method,

        loss=loss,

        f_scale=f_scale,

        bounds=bounds,

        node_disambiguation=node_disambiguation,

        node_rtol=node_rtol
    )

    return fit_visual_orbit_mcmc_from_problem(

        problem,

        n_walkers=n_walkers,

        n_steps=n_steps,

        burn_in=burn_in,

        thin=thin,

        random_scale=random_scale,

        rng=rng
    )