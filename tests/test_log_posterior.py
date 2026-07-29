import numpy as np
import pytest

from pyseshat.io.orbit_parser import (
    parse_orbit_file,
)

from pyseshat.inference.optimizer import (
    fit_visual_orbit,
    fit_visual_orbit_from_problem,
)

from pyseshat.inference.context import (
    LikelihoodContext,
)

from pyseshat.inference.problem import (
    InferenceProblem,
)

from pyseshat.inference.priors import (
    log_prior,
)

from pyseshat.inference.likelihood import (
    log_likelihood,
)

from pyseshat.inference.posterior import (
    log_posterior,
    log_posterior_from_problem,
)


# ==========================================================
# Shared benchmark fixture
# ==========================================================

@pytest.fixture(scope="module")
def fin379_problem():

    parsed = parse_orbit_file(
        "validation/benchmark_systems/FIN379/FIN379.inp"
    )

    elements = parsed["elements"]

    visual = parsed["visual_data"]

    rv_data = parsed["rv_data"]

    t = visual[:, 0]

    theta = np.deg2rad(
        visual[:, 1]
    )

    rho = visual[:, 2]

    sigma = visual[:, 3]

    X_obs = rho * np.cos(theta)

    Y_obs = rho * np.sin(theta)

    fit = fit_visual_orbit(
        elements,
        t,
        X_obs,
        Y_obs,
        sigma,
        sigma,
        rv_data=rv_data,
    )

    params = fit.elements.fitted_parameters()

    context = LikelihoodContext(
        t=t,
        X_obs=X_obs,
        Y_obs=Y_obs,
        sx=sigma,
        sy=sigma,
        rv_data=rv_data,
    )

    problem = InferenceProblem(
        elements=fit.elements,
        context=context,
    )

    return {
        "fit": fit,
        "params": params,
        "problem": problem,
        "t": t,
        "X_obs": X_obs,
        "Y_obs": Y_obs,
        "sigma": sigma,
        "rv_data": rv_data,
    }


# ==========================================================
# Posterior = Prior + Likelihood
# ==========================================================

def test_log_posterior_identity(fin379_problem):

    fit = fin379_problem["fit"]
    params = fin379_problem["params"]
    t = fin379_problem["t"]
    X_obs = fin379_problem["X_obs"]
    Y_obs = fin379_problem["Y_obs"]
    sigma = fin379_problem["sigma"]
    rv_data = fin379_problem["rv_data"]

    lp = log_prior(
        fit.elements
    )

    ll = log_likelihood(
        params,
        fit.elements,
        t,
        X_obs,
        Y_obs,
        sigma,
        sigma,
        rv_data=rv_data,
    )

    post = log_posterior(
        params,
        fit.elements,
        t,
        X_obs,
        Y_obs,
        sigma,
        sigma,
        rv_data=rv_data,
    )

    assert np.isclose(
        post,
        lp + ll,
        rtol=1e-12,
        atol=1e-12,
    )


# ==========================================================
# Legacy API == Problem API
# ==========================================================

def test_problem_api_matches_legacy_api(fin379_problem):

    fit = fin379_problem["fit"]
    params = fin379_problem["params"]
    problem = fin379_problem["problem"]
    t = fin379_problem["t"]
    X_obs = fin379_problem["X_obs"]
    Y_obs = fin379_problem["Y_obs"]
    sigma = fin379_problem["sigma"]
    rv_data = fin379_problem["rv_data"]

    legacy = log_posterior(
        params,
        fit.elements,
        t,
        X_obs,
        Y_obs,
        sigma,
        sigma,
        rv_data=rv_data,
    )

    modern = log_posterior_from_problem(
        params,
        problem,
    )

    assert np.isclose(
        legacy,
        modern,
        rtol=1e-12,
        atol=1e-12,
    )

def test_fit_visual_orbit_from_problem_matches_legacy(
    fin379_problem,
):
    """
    Verify that the modern InferenceProblem API produces the
    same least-squares solution as the legacy interface.
    """

    problem = fin379_problem["problem"]

    # ------------------------------------------------------
    # Legacy API
    # ------------------------------------------------------

    legacy_fit = fit_visual_orbit(

        problem.elements,

        problem.t,

        problem.X_obs,

        problem.Y_obs,

        problem.sx,

        problem.sy,

        rv_data=problem.rv_data,

        method=problem.method,

        loss=problem.loss,

        f_scale=problem.f_scale,

        bounds=problem.bounds,

        node_disambiguation=problem.node_disambiguation,

        node_rtol=problem.node_rtol,
    )

    # ------------------------------------------------------
    # Modern API
    # ------------------------------------------------------

    problem_fit = fit_visual_orbit_from_problem(
        problem
    )

    # ------------------------------------------------------
    # Fitted orbital parameters
    # ------------------------------------------------------

    assert np.allclose(

        legacy_fit.elements.fitted_parameters(),

        problem_fit.elements.fitted_parameters(),

        rtol=1e-12,

        atol=1e-12,
    )

    # ------------------------------------------------------
    # Residual vector
    # ------------------------------------------------------

    assert np.allclose(

        legacy_fit.residuals,

        problem_fit.residuals,

        rtol=1e-12,

        atol=1e-12,
    )

    # ------------------------------------------------------
    # Chi-square
    # ------------------------------------------------------

    assert np.isclose(

        np.sum(legacy_fit.residuals**2),

        np.sum(problem_fit.residuals**2),

        rtol=1e-12,

        atol=1e-12,
    )

    # ------------------------------------------------------
    # Covariance matrix
    # ------------------------------------------------------

    assert np.allclose(

        legacy_fit.covariance,

        problem_fit.covariance,

        rtol=1e-12,

        atol=1e-12,
    )

    # ------------------------------------------------------
    # Parameter uncertainties
    # ------------------------------------------------------

    assert set(legacy_fit.uncertainties) == set(problem_fit.uncertainties)

    for parameter in legacy_fit.uncertainties:

        assert np.isclose(
            legacy_fit.uncertainties[parameter],
            problem_fit.uncertainties[parameter],
            rtol=1e-12,
            atol=1e-12,
        )