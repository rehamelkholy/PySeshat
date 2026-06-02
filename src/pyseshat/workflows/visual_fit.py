import numpy as np

from pyseshat.io.orbit_parser import parse_orbit_file

from pyseshat.inference.optimizer import fit_visual_orbit

from pyseshat.inference.likelihood import (
    visual_xy_residuals
)

from pyseshat.orbit.orbit_model import (
    compute_visual_orbit
)

from pyseshat.analysis.statistics import (
    fit_summary
)


# ==========================================================
# Polar -> Cartesian conversion
# ==========================================================

def polar_to_cartesian(data):
    """
    Convert visual binary observations from:

        (theta, rho)

    to:

        (X, Y)

    Parameters
    ----------
    data : ndarray

    Returns
    -------
    t, X, Y, sx, sy
    """

    t = data[:,0]

    theta = np.deg2rad(data[:,1])

    rho = data[:,2]

    sigma = data[:,3]

    X = rho * np.cos(theta)
    Y = rho * np.sin(theta)

    # isotropic approximation
    sx = sigma
    sy = sigma

    return t, X, Y, sx, sy


# ==========================================================
# Main deterministic visual fit workflow
# ==========================================================

def run_visual_fit(filepath):
    """
    Execute deterministic visual binary fitting workflow.

    Parameters
    ----------
    filepath : str

    Returns
    -------
    dict
    """

    # ------------------------------------------------------
    # Parse ORBIT-format file
    # ------------------------------------------------------

    parsed = parse_orbit_file(filepath)

    initial_elements = parsed["elements"]

    visual_data = parsed["visual_data"]

    metadata = parsed["metadata"]

    # ------------------------------------------------------
    # Convert observations
    # ------------------------------------------------------

    t, X, Y, sx, sy = polar_to_cartesian(
        visual_data
    )

    # ------------------------------------------------------
    # Fit orbit
    # ------------------------------------------------------

    fitted_elements, optimizer_result = fit_visual_orbit(
        initial_elements,
        t,
        X,
        Y,
        sx,
        sy
    )

    # ------------------------------------------------------
    # Orbit evaluated at observation epochs
    # ------------------------------------------------------

    orbit_observed = compute_visual_orbit(
        t,
        fitted_elements
    )

    # ------------------------------------------------------
    # Residuals
    # ------------------------------------------------------

    residuals = visual_xy_residuals(
        fitted_elements.fitted_parameters(),
        fitted_elements,
        t,
        X,
        Y,
        sx,
        sy
    )

    # ------------------------------------------------------
    # Statistics
    # ------------------------------------------------------

    statistics = fit_summary(
        residuals,
        fitted_elements
    )

    return {

        "metadata": metadata,

        "initial_elements": initial_elements,

        "fitted_elements": fitted_elements,

        "visual_data": visual_data,

        "cartesian_data": {
            "t": t,
            "X": X,
            "Y": Y,
            "sx": sx,
            "sy": sy
        },

        "orbit_observed": orbit_observed,

        "residuals": residuals,

        "statistics": statistics,

        "optimizer_result": optimizer_result
    }