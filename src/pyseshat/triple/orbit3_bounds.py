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

This module is deliberately separate from the legacy
ORBITT2 bounds implementation in ``triple_bounds.py``.
"""

import numpy as np


# ==========================================================
# ORBIT3 parameter order
# ==========================================================

ORBIT3_PARAMETER_ORDER = [

    # ------------------------------------------------------
    # Inner orbit
    # ------------------------------------------------------

    "P",
    "T0",
    "e",
    "a",
    "Omega",
    "omega",
    "i",

    # ------------------------------------------------------
    # Inner spectroscopic orbit / wobble
    # ------------------------------------------------------

    "K1",
    "K2",
    "f",

    # ------------------------------------------------------
    # Outer orbit
    # ------------------------------------------------------

    "P2",
    "T02",
    "e2",
    "a2",
    "Omega2",
    "omega2",
    "i2",

    # ------------------------------------------------------
    # Outer spectroscopic orbit
    # ------------------------------------------------------

    "K3",
    "K4",
    "V0",

]


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

    These are physical parameter-domain bounds.

    Orbital eccentricities satisfy

        0 <= e < 1

    and

        0 <= e2 < 1.

    The astrometric wobble factor satisfies

        0 <= f <= 1.

    This follows from the ORBIT3 definition

        f = q_in / (1 + q_in).

    Periods and semi-major axes are positive.

    Angular elements use the conventional ranges

        0 <= Omega < 360 deg
        0 <= omega < 360 deg
        0 <= i <= 180 deg.

    Radial-velocity semi-amplitudes are non-negative.

    Epochs and systemic velocity are otherwise unrestricted.
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
        0.0,         # f

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
        1.0,         # f

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
# Fitted ORBIT3 bounds
# ==========================================================

def build_orbit3_bounds(
    elements
):
    """
    Construct optimizer bounds for the fitted parameters
    of a TripleOrbitalElements object using the modern
    20-parameter ORBIT3 representation.

    Only parameters marked True in the ORBIT3 fit mask are
    returned.
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

            "Internal ORBIT3 lower-bound vector must "
            f"contain exactly 20 parameters; "
            f"received {lower_full.size}."

        )


    if upper_full.size != 20:

        raise RuntimeError(

            "Internal ORBIT3 upper-bound vector must "
            f"contain exactly 20 parameters; "
            f"received {upper_full.size}."

        )


    return (

        lower_full[mask],

        upper_full[mask]

    )