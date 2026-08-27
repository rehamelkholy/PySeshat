"""
Bounds for the modern ORBIT3 hierarchical-triple
parameterization.

The ORBIT3 representation contains 20 parameters:

    P
    T0
    e
    a
    Omega
    omega
    i
    K1
    K2
    f
    P2
    T02
    e2
    a2
    Omega2
    omega2
    i2
    K3
    K4
    V0

This module is separate from the legacy ORBITT2
parameterization in ``triple_bounds.py``.
"""

import numpy as np

from pyseshat.triple.triple_elements import (
    ORBIT3_PARAMETER_ORDER
)

# ORBIT3_PARAMETER_ORDER is imported from triple_elements.py,
# the single canonical source of this ordering, and re-exported
# here for backward compatibility with existing imports.


# ==========================================================
# Complete ORBIT3 bounds
# ==========================================================

def orbit3_full_bounds():
    """
    Return lower and upper bounds for the complete
    20-parameter ORBIT3 parameter vector.

    Parameter order
    ---------------

    P, T0, e, a, Omega, omega, i,
    K1, K2, f,
    P2, T02, e2, a2, Omega2, omega2, i2,
    K3, K4, V0

    The bounds on f are deliberately unrestricted.

    The reference ORBIT3 implementation does not impose
    explicit optimizer bounds on f, and the official
    HIP111805 benchmark contains

        f = -0.33.

    Therefore f must not be restricted to [0, 1].

    The remaining bounds provide numerical/physical
    safeguards for the SciPy optimizer. They should not be
    interpreted as bounds imposed by the original IDL
    implementation.
    """

    lower = np.array([

        # --------------------------------------------------
        # Inner orbit
        # --------------------------------------------------

        1e-12,       # P
        -np.inf,     # T0
        0.0,         # e
        1e-12,       # a
        0.0,         # Omega
        0.0,         # omega
        0.0,         # i

        # --------------------------------------------------
        # Inner spectroscopy / wobble
        # --------------------------------------------------

        0.0,         # K1
        0.0,         # K2
        -np.inf,     # f

        # --------------------------------------------------
        # Outer orbit
        # --------------------------------------------------

        1e-12,       # P2
        -np.inf,     # T02
        0.0,         # e2
        1e-12,       # a2
        0.0,         # Omega2
        0.0,         # omega2
        0.0,         # i2

        # --------------------------------------------------
        # Outer spectroscopy
        # --------------------------------------------------

        0.0,         # K3
        0.0,         # K4
        -np.inf,     # V0

    ], dtype=float)


    upper = np.array([

        # --------------------------------------------------
        # Inner orbit
        # --------------------------------------------------

        np.inf,      # P
        np.inf,      # T0
        0.999999,    # e
        np.inf,      # a
        360.0,       # Omega
        360.0,       # omega
        180.0,       # i

        # --------------------------------------------------
        # Inner spectroscopy / wobble
        # --------------------------------------------------

        np.inf,      # K1
        np.inf,      # K2
        np.inf,      # f

        # --------------------------------------------------
        # Outer orbit
        # --------------------------------------------------

        np.inf,      # P2
        np.inf,      # T02
        0.999999,    # e2
        np.inf,      # a2
        360.0,       # Omega2
        360.0,       # omega2
        180.0,       # i2

        # --------------------------------------------------
        # Outer spectroscopy
        # --------------------------------------------------

        np.inf,      # K3
        np.inf,      # K4
        np.inf,      # V0

    ], dtype=float)


    if lower.size != 20:

        raise RuntimeError(

            "Internal ORBIT3 lower-bound vector must "
            f"contain exactly 20 parameters; "
            f"received {lower.size}."

        )


    if upper.size != 20:

        raise RuntimeError(

            "Internal ORBIT3 upper-bound vector must "
            f"contain exactly 20 parameters; "
            f"received {upper.size}."

        )


    return lower, upper


# ==========================================================
# Fitted-parameter bounds
# ==========================================================

def build_orbit3_bounds(
    elements
):
    """
    Construct optimizer bounds for the fitted ORBIT3
    parameters only.

    Parameters
    ----------
    elements : TripleOrbitalElements
        ORBIT3 element object containing the fit mask.

    Returns
    -------
    tuple(ndarray, ndarray)
        Lower and upper bounds corresponding exactly to
        ``elements.fitted_parameter_names()``.
    """

    lower_full, upper_full = (
        orbit3_full_bounds()
    )


    mask = elements.fitted_mask()


    if mask.size != 20:

        raise ValueError(

            "ORBIT3 fit mask must contain exactly "
            f"20 entries; received {mask.size}."

        )


    if lower_full.size != 20:

        raise RuntimeError(

            "Internal ORBIT3 lower-bound vector "
            "does not contain 20 parameters."

        )


    if upper_full.size != 20:

        raise RuntimeError(

            "Internal ORBIT3 upper-bound vector "
            "does not contain 20 parameters."

        )


    return (

        lower_full[mask],

        upper_full[mask]

    )