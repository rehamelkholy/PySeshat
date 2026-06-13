import numpy as np

from copy import deepcopy

from pyseshat.inference.likelihood import (
    combined_residuals
)

from pyseshat.orbit.orbital_elements import (
    OrbitalElements
)


# ==========================================================
# Node flip
# ==========================================================

def node_flip(
    elements: OrbitalElements
):
    """
    Generate the alternative visual-binary
    node assignment.

    For visual binaries:

        (Omega, omega)

    and

        (Omega+180, omega+180)

    produce the same apparent orbit.

    Radial velocities break this degeneracy.

    Parameters
    ----------
    elements : OrbitalElements

    Returns
    -------
    OrbitalElements
        New orbital-elements object with
        flipped node assignment.
    """

    flipped = deepcopy(
        elements
    )

    flipped.Omega = (
        flipped.Omega + 180.0
    ) % 360.0

    flipped.omega = (
        flipped.omega + 180.0
    ) % 360.0

    return flipped

# ==========================================================
# Compare node assignments
# ==========================================================

def node_assignment_chi2(
    elements,
    t,
    X_obs,
    Y_obs,
    sx,
    sy,
    rv_data
):
    """
    Evaluate both node assignments without fitting.

    Returns
    -------
    dict
    """

    # ------------------------------------------------------
    # Original node assignment
    # ------------------------------------------------------

    r_original = combined_residuals(
        elements.fitted_parameters(),
        elements,
        t,
        X_obs,
        Y_obs,
        sx,
        sy,
        rv_data
    )

    chi2_original = np.sum(
        r_original**2
    )

    # ------------------------------------------------------
    # Flipped node assignment
    # ------------------------------------------------------

    flipped = node_flip(
        elements
    )

    r_flipped = combined_residuals(
        flipped.fitted_parameters(),
        flipped,
        t,
        X_obs,
        Y_obs,
        sx,
        sy,
        rv_data
    )

    chi2_flipped = np.sum(
        r_flipped**2
    )

    return {
        "chi2_original": chi2_original,
        "chi2_flipped": chi2_flipped,
        "prefer_flipped":
            chi2_flipped < chi2_original
    }

# ==========================================================
# Select best node assignment
# ==========================================================

def select_best_node_assignment(
    elements,
    t,
    X_obs,
    Y_obs,
    sx,
    sy,
    rv_data
):
    """
    Select the better node assignment
    before optimization.

    Returns
    -------
    OrbitalElements
    """

    comparison = node_assignment_chi2(
        elements,
        t,
        X_obs,
        Y_obs,
        sx,
        sy,
        rv_data
    )

    if comparison["prefer_flipped"]:

        return node_flip(
            elements
        )

    return elements