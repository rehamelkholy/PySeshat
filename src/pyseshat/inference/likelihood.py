import numpy as np

from pyseshat.orbit.orbit_model import compute_visual_orbit

from pyseshat.orbit.rv_model import (
    compute_rv_model
)
from pyseshat.orbit.orbital_elements import OrbitalElements

from pyseshat.inference.core import (
    update_elements,
)

from pyseshat.inference.terms import (
    visual_xy_residuals,
    rv_residuals,
    joint_residuals,
    joint_residuals_from_context,
)

from pyseshat.inference.context import (
    LikelihoodContext,
)

from pyseshat.inference.problem import (
    InferenceProblem,
)

# ==========================================================
# Combined visual + RV residuals
# ==========================================================

def combined_residuals(
    fitted_values,
    elements,
    t,
    X_obs,
    Y_obs,
    sx,
    sy,
    rv_data=None,
):
    """
    Backward-compatible wrapper around the context-based
    likelihood implementation.
    """

    context = LikelihoodContext(
        t=t,
        X_obs=X_obs,
        Y_obs=Y_obs,
        sx=sx,
        sy=sy,
        rv_data=rv_data,
    )

    return joint_residuals_from_context(
        fitted_values=fitted_values,
        elements=elements,
        context=context,
    )

# ==========================================================
# Chi-square
# ==========================================================

def chi_square(
    fitted_values,
    elements,
    t,
    X_obs,
    Y_obs,
    sx,
    sy,
    rv_data=None
):
    """
    Compute chi-square statistic.
    """

    residuals = combined_residuals(
        fitted_values=fitted_values,
        elements=elements,
        t=t,
        X_obs=X_obs,
        Y_obs=Y_obs,
        sx=sx,
        sy=sy,
        rv_data=rv_data
    )

    return np.sum(
        residuals ** 2
    )


# ==========================================================
# Reduced chi-square
# ==========================================================

def reduced_chi_square(
    fitted_values,
    elements,
    t,
    X_obs,
    Y_obs,
    sx,
    sy,
    rv_data=None
):
    """
    Compute reduced chi-square.
    """

    chi2 = chi_square(
        fitted_values,
        elements,
        t,
        X_obs,
        Y_obs,
        sx,
        sy,
        rv_data=rv_data
    )

    # ------------------------------------------------------
    # Number of data points
    # ------------------------------------------------------

    n_visual = 2 * len(t)

    n_rv = 0

    if (
        rv_data is not None
        and len(rv_data) > 0
    ):

        n_rv = len(rv_data)

    n_obs = n_visual + n_rv

    # ------------------------------------------------------
    # Number of fitted parameters
    # ------------------------------------------------------

    n_fit = np.sum(
        elements.fitted_mask()
    )

    dof = n_obs - n_fit

    if dof <= 0:
        raise ValueError(
            f"Degrees of freedom must be positive. "
            f"Got {dof} (n_obs={n_obs}, n_fit={n_fit})."
        )

    return chi2 / dof

# ==========================================================
# Log-likelihood
# ==========================================================

def log_likelihood(
    fitted_values,
    elements,
    t,
    X_obs,
    Y_obs,
    sx,
    sy,
    rv_data=None
):
    """
    Gaussian log-likelihood.

    Public API retained for backward compatibility.
    """

    context = LikelihoodContext(
        t=t,
        X_obs=X_obs,
        Y_obs=Y_obs,
        sx=sx,
        sy=sy,
        rv_data=rv_data,
    )

    residuals = joint_residuals_from_context(
        fitted_values=fitted_values,
        elements=elements,
        context=context,
    )

    return -0.5 * np.sum(residuals ** 2)


def log_likelihood_from_context(
    fitted_values,
    elements,
    context: LikelihoodContext,
):
    """
    Preferred implementation operating directly on
    LikelihoodContext.
    """

    residuals = joint_residuals_from_context(
        fitted_values=fitted_values,
        elements=elements,
        context=context,
    )

    return -0.5 * np.sum(residuals ** 2)

def log_likelihood_from_problem(
    fitted_values,
    problem: InferenceProblem,
):
    """
    Preferred likelihood interface operating on an
    InferenceProblem.
    """

    return log_likelihood_from_context(
        fitted_values=fitted_values,
        elements=problem.elements,
        context=problem.context,
    )