import numpy as np


# ==========================================================
# Legacy ORBITT2 fitted-parameter mask
# ==========================================================

def _legacy_fitted_mask(
    elements
):
    """
    Return the fitted-parameter mask for the legacy
    15-parameter ORBITT2 triple representation.

    Parameter order:

        P, T0, e, a, Omega, omega, i,
        P2, T02, e2, a2, Omega2, omega2, i2,
        B

    This must remain separate from
    TripleOrbitalElements.fitted_mask(), which represents
    the expanded ORBIT3 parameter space.
    """

    return np.array([

        # --------------------------------------------------
        # Outer orbit
        # --------------------------------------------------

        elements.fit_flags["P"],
        elements.fit_flags["T0"],
        elements.fit_flags["e"],
        elements.fit_flags["a"],
        elements.fit_flags["Omega"],
        elements.fit_flags["omega"],
        elements.fit_flags["i"],

        # --------------------------------------------------
        # Inner orbit
        # --------------------------------------------------

        elements.fit_flags["P2"],
        elements.fit_flags["T02"],
        elements.fit_flags["e2"],
        elements.fit_flags["a2"],
        elements.fit_flags["Omega2"],
        elements.fit_flags["omega2"],
        elements.fit_flags["i2"],

        # --------------------------------------------------
        # Legacy scale parameter
        # --------------------------------------------------

        elements.fit_flags["B"]

    ], dtype=bool)


# ==========================================================
# Legacy triple optimizer bounds
# ==========================================================

def build_triple_bounds(
    elements
):
    """
    Construct optimizer bounds for fitted parameters
    in the legacy ORBITT2 15-parameter representation.

    The parameter order is:

        P, T0, e, a, Omega, omega, i,
        P2, T02, e2, a2, Omega2, omega2, i2,
        B

    This function belongs to the legacy visual-orbit
    fitting pathway. ORBIT3 spectroscopic parameters
    (K1, K2, f, K3, K4, V0) are deliberately not included.
    """

    # ------------------------------------------------------
    # Full lower bounds.
    # ------------------------------------------------------

    lower_full = np.array([

        # --------------------------------------------------
        # Outer orbit
        # --------------------------------------------------

        1e-6,       # P
        -np.inf,    # T0
        0.0,        # e
        0.0,        # a
        0.0,        # Omega
        0.0,        # omega
        0.0,        # i

        # --------------------------------------------------
        # Inner orbit
        # --------------------------------------------------

        1e-6,       # P2
        -np.inf,    # T02
        0.0,        # e2
        0.0,        # a2
        0.0,        # Omega2
        0.0,        # omega2
        0.0,        # i2

        # --------------------------------------------------
        # Legacy scale
        # --------------------------------------------------

        0.0         # B

    ], dtype=float)


    # ------------------------------------------------------
    # Full upper bounds.
    # ------------------------------------------------------

    upper_full = np.array([

        # --------------------------------------------------
        # Outer orbit
        # --------------------------------------------------

        np.inf,     # P
        np.inf,     # T0
        0.99,       # e
        np.inf,     # a
        360.0,      # Omega
        360.0,      # omega
        180.0,      # i

        # --------------------------------------------------
        # Inner orbit
        # --------------------------------------------------

        np.inf,     # P2
        np.inf,     # T02
        0.99,       # e2
        np.inf,     # a2
        360.0,      # Omega2
        360.0,      # omega2
        180.0,      # i2

        # --------------------------------------------------
        # Legacy scale
        # --------------------------------------------------

        np.inf      # B

    ], dtype=float)


    # ------------------------------------------------------
    # IMPORTANT:
    #
    # triple_to_array() in the legacy fitter has 15
    # parameters, so the mask must also have 15 entries.
    #
    # Do NOT use elements.fitted_mask() here because that
    # now represents the 20-parameter ORBIT3 model.
    # ------------------------------------------------------

    mask = _legacy_fitted_mask(
        elements
    )


    return (

        lower_full[mask],

        upper_full[mask]

    )