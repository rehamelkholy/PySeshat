import numpy as np

from scipy.optimize import least_squares

from pyseshat.inference.optimizer_engine import (
    run_least_squares
)
from pyseshat.inference.node_disambiguation import (
    node_flip
)

from pyseshat.inference.likelihood import (
    visual_xy_residuals,
    combined_residuals,
    update_elements
)

from pyseshat.inference.fit_result import (
    FitResult
)

from pyseshat.inference.covariance import (
    estimate_covariance,
    parameter_uncertainties
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