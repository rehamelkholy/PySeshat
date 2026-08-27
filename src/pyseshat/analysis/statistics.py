import numpy as np

from pyseshat.orbit.orbital_elements import OrbitalElements


# ==========================================================
# Chi-square
# ==========================================================

def chi_square(residuals):
    """
    Compute chi-square statistic.

    Parameters
    ----------
    residuals : ndarray
        Weighted residual vector

    Returns
    -------
    float
    """

    residuals = np.asarray(residuals)

    return np.sum(residuals**2)


# ==========================================================
# Reduced chi-square
# ==========================================================

def reduced_chi_square(residuals, n_fit):
    """
    Compute reduced chi-square.

    Parameters
    ----------
    residuals : ndarray
        Weighted residual vector

    n_fit : int
        Number of fitted parameters

    Returns
    -------
    float
    """

    residuals = np.asarray(residuals)

    n_obs = len(residuals)

    dof = n_obs - n_fit

    if dof <= 0:
        raise ValueError(
            "Degrees of freedom must be positive. "
            f"Got {dof} (n_obs={n_obs}, n_fit={n_fit})."
        )

    return chi_square(residuals) / dof


# ==========================================================
# RMS residual
# ==========================================================

def rms_residual(residuals):
    """
    Compute RMS of residual vector.

    Parameters
    ----------
    residuals : ndarray

    Returns
    -------
    float
    """

    residuals = np.asarray(residuals)

    return np.sqrt(np.mean(residuals**2))


# ==========================================================
# Parameter comparison
# ==========================================================

def parameter_comparison(
    initial: OrbitalElements,
    fitted: OrbitalElements
):
    """
    Compare initial and fitted orbital elements.

    Returns
    -------
    dict
    """

    names = initial.PARAMETER_ORDER

    p_init = initial.to_array()
    p_fit = fitted.to_array()

    comparison = {}

    for name, pi, pf in zip(names, p_init, p_fit):

        if np.abs(pi) > 0:

            rel = 100.0 * np.abs((pf - pi) / pi)

        else:

            rel = np.nan

        comparison[name] = {
            "initial": pi,
            "fitted": pf,
            "relative_error_percent": rel
        }

    return comparison


# ==========================================================
# Print comparison table
# ==========================================================

def print_parameter_comparison(
    initial: OrbitalElements,
    fitted: OrbitalElements
):
    """
    Print formatted parameter comparison table.
    """

    comparison = parameter_comparison(
        initial,
        fitted
    )

    print("\n=== PARAMETER COMPARISON ===\n")

    print(
        f"{'Param':<10}"
        f"{'Initial':<15}"
        f"{'Fitted':<15}"
        f"{'Rel. Error (%)':<20}"
    )

    for key, values in comparison.items():

        print(
            f"{key:<10}"
            f"{values['initial']:<15.6f}"
            f"{values['fitted']:<15.6f}"
            f"{values['relative_error_percent']:<20.6f}"
        )


# ==========================================================
# Fit summary
# ==========================================================

def fit_summary(
    residuals,
    elements: OrbitalElements
):
    """
    Compute standard fit diagnostics.

    Returns
    -------
    dict
    """

    chi2 = chi_square(residuals)

    redchi2 = reduced_chi_square(
        residuals,
        np.sum(elements.fitted_mask())
    )

    rms = rms_residual(residuals)

    return {
        "chi_square": chi2,
        "reduced_chi_square": redchi2,
        "rms": rms
    }

# ==========================================================
# Combined modality diagnostics
# ==========================================================

def combined_fit_summary(
    visual_residuals,
    rv_residuals=None,
    rv_components=None,
    n_params=0
):
    """
    Compute separated chi-square diagnostics
    for visual + RV fitting.

    Parameters
    ----------
    visual_residuals : ndarray

    rv_residuals : ndarray or None

    rv_components : ndarray or None

        0 = primary
        1 = secondary

    n_params : int

    Returns
    -------
    dict
    """

    # ------------------------------------------------------
    # Visual chi-square
    # ------------------------------------------------------

    chi2_visual = np.sum(
        visual_residuals**2
    )

    n_visual = len(visual_residuals)

    # ------------------------------------------------------
    # No RV data
    # ------------------------------------------------------

    if rv_residuals is None:

        chi2_total = chi2_visual

        dof = n_visual - n_params

        if dof <= 0:
            raise ValueError(
                "Degrees of freedom must be positive. "
                f"Got {dof} (n_obs={n_visual}, "
                f"n_params={n_params})."
            )

        rms = np.sqrt(
            chi2_total / n_visual
        )

        return {

            "chi2_visual": chi2_visual,

            "chi2_total": chi2_total,

            "reduced_chi2": (
                chi2_total / dof
            ),

            "rms": rms
        }

    # ------------------------------------------------------
    # RV masks
    # ------------------------------------------------------

    mask_rv1 = (rv_components == 0)

    mask_rv2 = (rv_components == 1)

    # ------------------------------------------------------
    # RV chi-square
    # ------------------------------------------------------

    chi2_rv1 = np.sum(
        rv_residuals[mask_rv1]**2
    )

    chi2_rv2 = np.sum(
        rv_residuals[mask_rv2]**2
    )

    # ------------------------------------------------------
    # Totals
    # ------------------------------------------------------

    chi2_total = (
        chi2_visual
        +
        chi2_rv1
        +
        chi2_rv2
    )

    n_total = (
        len(visual_residuals)
        +
        len(rv_residuals)
    )

    dof = n_total - n_params

    if dof <= 0:
        raise ValueError(
            "Degrees of freedom must be positive. "
            f"Got {dof} (n_obs={n_total}, "
            f"n_params={n_params})."
        )

    reduced = (
        chi2_total / dof
    )

    rms = np.sqrt(
        chi2_total / n_total
    )

    return {

        "chi2_visual": chi2_visual,

        "chi2_rv1": chi2_rv1,

        "chi2_rv2": chi2_rv2,

        "chi2_total": chi2_total,

        "reduced_chi2": reduced,

        "rms": rms
    }

# ==========================================================
# Print fit summary
# ==========================================================

def print_fit_summary(
    residuals,
    elements: OrbitalElements
):
    """
    Print formatted fit diagnostics.
    """

    stats = fit_summary(
        residuals,
        elements
    )

    print("\n=== FIT STATISTICS ===\n")

    print(f"Chi-square        = {stats['chi_square']:.6f}")
    print(f"Reduced Chi-square= {stats['reduced_chi_square']:.6f}")
    print(f"RMS (normalized)  = {stats['rms']:.6f}")