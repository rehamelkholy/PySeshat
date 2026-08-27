import numpy as np


from pyseshat.orbit.orbit_model import (
    compute_visual_orbit
)


from pyseshat.triple.triple_elements import (
    TripleOrbitalElements
)


# ==========================================================
# Legacy hierarchical triple orbit
# ==========================================================

def compute_triple_orbit(
    epochs,
    elements: TripleOrbitalElements
):
    """
    Compute the legacy hierarchical triple visual orbit.

    This function is retained for the existing ORBITT2
    implementation.

    Parameters
    ----------
    epochs : array-like
        Observation epochs.

    elements : TripleOrbitalElements
        Triple orbital elements.

    Returns
    -------
    dict
        Cartesian and polar coordinates of the combined
        legacy triple trajectory.
    """

    outer = compute_visual_orbit(

        epochs,

        elements.outer

    )


    inner = compute_visual_orbit(

        epochs,

        elements.inner

    )


    X = (

        outer["X"]

        +

        elements.scale_inner
        * inner["X"]

    )


    Y = (

        outer["Y"]

        +

        elements.scale_inner
        * inner["Y"]

    )


    rho = np.sqrt(

        X**2

        +

        Y**2

    )


    theta = (

        np.degrees(

            np.arctan2(

                Y,

                X

            )

        )

        %

        360.0

    )


    return {

        "X": X,

        "Y": Y,

        "rho": rho,

        "theta": theta,

        "outer": outer,

        "inner": inner

    }


# ==========================================================
# ORBIT3 visual orbit
# ==========================================================

def compute_orbit3_visual(
    epochs,
    elements: TripleOrbitalElements
):
    """
    Compute the two visual observables defined by ORBIT3.

    ORBIT3 defines:

        POS1 = inner orbit

        POS2 = outer orbit + f * inner orbit

    where ``f`` is the astrometric wobble factor.

    Parameters
    ----------
    epochs : array-like
        Observation epochs.

    elements : TripleOrbitalElements
        ORBIT3 triple orbital elements.

    Returns
    -------
    dict
        Dictionary containing the individual inner and outer
        visual orbits together with the two ORBIT3 visual
        observables.

    Notes
    -----
    The formulation follows the ``eph`` and ``eph2`` routines
    in Tokovinin's ORBIT3 implementation.
    """

    epochs = np.asarray(
        epochs,
        dtype=float
    )


    # ------------------------------------------------------
    # ORBIT3 first orbit:
    #
    #   el[0:6]
    #
    # This is the inner visual orbit.
    # ------------------------------------------------------

    inner = compute_visual_orbit(

        epochs,

        elements.inner

    )


    # ------------------------------------------------------
    # ORBIT3 second orbit:
    #
    #   el[10:16]
    #
    # This is the outer visual orbit.
    # ------------------------------------------------------

    outer = compute_visual_orbit(

        epochs,

        elements.outer

    )


    # ------------------------------------------------------
    # POS1
    #
    # ORBIT3:
    #
    #   eph(el[0:6], t)
    #
    # Therefore POS1 is simply the inner orbit.
    # ------------------------------------------------------

    pos1_X = inner["X"]

    pos1_Y = inner["Y"]


    # ------------------------------------------------------
    # POS2
    #
    # ORBIT3:
    #
    #   eph(el[10:16], t)
    #       +
    #   eph(el[0:6], t) * el[9]
    #
    # where el[9] = f.
    # ------------------------------------------------------

    pos2_X = (

        outer["X"]

        +

        elements.f
        * inner["X"]

    )


    pos2_Y = (

        outer["Y"]

        +

        elements.f
        * inner["Y"]

    )


    # ------------------------------------------------------
    # Polar coordinates
    # ------------------------------------------------------

    pos1_rho = np.sqrt(

        pos1_X**2

        +

        pos1_Y**2

    )


    pos1_theta = (

        np.degrees(

            np.arctan2(

                pos1_Y,

                pos1_X

            )

        )

        %

        360.0

    )


    pos2_rho = np.sqrt(

        pos2_X**2

        +

        pos2_Y**2

    )


    pos2_theta = (

        np.degrees(

            np.arctan2(

                pos2_Y,

                pos2_X

            )

        )

        %

        360.0

    )


    return {

        "inner": inner,

        "outer": outer,

        "POS1": {

            "X": pos1_X,

            "Y": pos1_Y,

            "rho": pos1_rho,

            "theta": pos1_theta,

        },

        "POS2": {

            "X": pos2_X,

            "Y": pos2_Y,

            "rho": pos2_rho,

            "theta": pos2_theta,

        },

    }


# ==========================================================
# ORBIT3 radial-velocity orbit
# ==========================================================

def compute_orbit3_radial_velocities(
    epochs,
    elements: TripleOrbitalElements
):
    """
    Compute the three radial-velocity observables defined
    by ORBIT3.

    ORBIT3 constructs:

        RV_Aa = RV_inner_primary
                + RV_outer_primary

        RV_Ab = RV_inner_secondary
                + RV_outer_primary

        RV_B  = RV_outer_secondary

    The ORBIT3 input omega is already in the spectroscopic
    convention for combined visual/spectroscopic orbits.

    Therefore, unlike the generic visual-binary RV model,
    this function must NOT apply an additional 180-degree
    omega transformation.

    The implementation follows Tokovinin's orbit3.pro:

        A1 = e*cos(omega) + cos(nu + omega)

        RV1 = V0 + K1*A1
        RV2 = V0 - K2*A1

    The inner systemic velocity is explicitly zero in
    orbit3.pro, while the outer orbit carries V0.
    """

    epochs = np.asarray(
        epochs,
        dtype=float
    )


    # ------------------------------------------------------
    # Obtain the orbital anomalies directly from the
    # corresponding visual-orbit calculations.
    #
    # compute_visual_orbit provides E and nu from P, T0,
    # and e. Its nu calculation is independent of the
    # visual/spectroscopic omega convention.
    # ------------------------------------------------------

    inner_orbit = compute_visual_orbit(

        epochs,

        elements.inner

    )


    outer_orbit = compute_visual_orbit(

        epochs,

        elements.outer

    )


    # ------------------------------------------------------
    # ORBIT3 spectroscopic convention
    #
    # IMPORTANT:
    #
    # orbit3.pro uses the supplied omega directly:
    #
    #     U  = V + W
    #     A1 = e*CW + cos(U)
    #
    # There is no omega -> omega + 180 transformation.
    # ------------------------------------------------------

    inner_omega = np.deg2rad(
        elements.inner.omega
    )


    outer_omega = np.deg2rad(
        elements.outer.omega
    )


    inner_common = (

        np.cos(

            inner_orbit["nu"]
            +
            inner_omega

        )

        +

        elements.inner.e
        *
        np.cos(
            inner_omega
        )

    )


    outer_common = (

        np.cos(

            outer_orbit["nu"]
            +
            outer_omega

        )

        +

        elements.outer.e
        *
        np.cos(
            outer_omega
        )

    )


    # ------------------------------------------------------
    # Inner spectroscopic orbit
    #
    # orbit3.pro:
    #
    #   elin = [el[0:8], 0.]
    #
    #   RV_Aa_inner = K1 * A1
    #   RV_Ab_inner = -K2 * A1
    #
    # The inner systemic velocity is zero.
    # ------------------------------------------------------

    inner_primary = (

        elements.K1
        *
        inner_common

    )


    inner_secondary = (

        -

        elements.K2
        *
        inner_common

    )


    # ------------------------------------------------------
    # Outer spectroscopic orbit
    #
    # orbit3.pro:
    #
    #   RV_A_outer = V0 + K3*A1
    #   RV_B_outer = V0 - K4*A1
    # ------------------------------------------------------

    outer_primary = (

        elements.V0

        +

        elements.K3
        *
        outer_common

    )


    outer_secondary = (

        elements.V0

        -

        elements.K4
        *
        outer_common

    )


    # ------------------------------------------------------
    # ORBIT3 observable velocities
    #
    # From orbit3.pro:
    #
    #   Aa = inner primary + outer primary
    #   Ab = inner secondary + outer primary
    #   B  = outer secondary
    # ------------------------------------------------------

    rv_Aa = (

        inner_primary

        +

        outer_primary

    )


    rv_Ab = (

        inner_secondary

        +

        outer_primary

    )


    rv_B = outer_secondary


    return {

        "inner_primary": inner_primary,

        "inner_secondary": inner_secondary,

        "outer_primary": outer_primary,

        "outer_secondary": outer_secondary,

        "Aa": rv_Aa,

        "Ab": rv_Ab,

        "B": rv_B,

    }


# ==========================================================
# Complete ORBIT3 forward model
# ==========================================================

def compute_orbit3_model(
    epochs,
    elements: TripleOrbitalElements
):
    """
    Compute all ORBIT3 model observables.

    This is the complete deterministic forward model before
    construction of observational residuals.

    Returns
    -------
    dict
        Contains:

        visual
            POS1 and POS2 predictions.

        radial_velocity
            Aa, Ab, and B predictions.

    Notes
    -----
    ORBIT3 combines the two visual orbits and three
    spectroscopic observables simultaneously. The original
    program subsequently forms weighted residuals in

        X1, Y1, X2, Y2, RV1, RV2, RV3.

    This function deliberately performs only the forward
    calculation. It does not apply observational weights.
    """

    visual = compute_orbit3_visual(

        epochs,

        elements

    )


    radial_velocity = (
        compute_orbit3_radial_velocities(

            epochs,

            elements

        )
    )


    return {

        "visual": visual,

        "radial_velocity": radial_velocity

    }