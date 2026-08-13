import numpy as np


from pyseshat.orbit.orbit_model import (
    compute_visual_orbit
)


from pyseshat.orbit.orbital_elements import (
    OrbitalElements
)


from pyseshat.orbit.rv_model import (
    compute_rv_orbit
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

    The inner systemic velocity is explicitly set to zero
    in the original ORBIT3 implementation. The outer orbit
    carries the fitted systemic velocity V0.

    Parameters
    ----------
    epochs : array-like
        Observation epochs.

    elements : TripleOrbitalElements
        ORBIT3 triple orbital elements.

    Returns
    -------
    dict
        Arrays for ``Aa``, ``Ab``, and ``B`` radial
        velocities, together with the individual inner and
        outer binary RV curves.
    """

    epochs = np.asarray(
        epochs,
        dtype=float
    )


    # ------------------------------------------------------
    # Construct temporary binary-element objects containing
    # the spectroscopic parameters belonging to each orbit.
    #
    # The orbital geometry itself is taken directly from the
    # corresponding TripleOrbitalElements orbit.
    # ------------------------------------------------------

    inner_rv_elements = OrbitalElements(

        P=elements.inner.P,

        T0=elements.inner.T0,

        e=elements.inner.e,

        a=elements.inner.a,

        Omega=elements.inner.Omega,

        omega=elements.inner.omega,

        i=elements.inner.i,

        K1=elements.K1,

        K2=elements.K2,

        V0=0.0

    )


    outer_rv_elements = OrbitalElements(

        P=elements.outer.P,

        T0=elements.outer.T0,

        e=elements.outer.e,

        a=elements.outer.a,

        Omega=elements.outer.Omega,

        omega=elements.outer.omega,

        i=elements.outer.i,

        K1=elements.K3,

        K2=elements.K4,

        V0=elements.V0

    )


    # ------------------------------------------------------
    # Inner spectroscopic orbit
    #
    # component 0 -> Aa
    # component 1 -> Ab
    # ------------------------------------------------------

    inner_primary = compute_rv_orbit(

        epochs,

        inner_rv_elements,

        component=0

    )


    inner_secondary = compute_rv_orbit(

        epochs,

        inner_rv_elements,

        component=1

    )


    # ------------------------------------------------------
    # Outer spectroscopic orbit
    #
    # component 0 -> A
    # component 1 -> B
    # ------------------------------------------------------

    outer_primary = compute_rv_orbit(

        epochs,

        outer_rv_elements,

        component=0

    )


    outer_secondary = compute_rv_orbit(

        epochs,

        outer_rv_elements,

        component=1

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