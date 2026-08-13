import numpy as np


from pyseshat.triple.triple_elements import (
    TripleOrbitalElements
)


from pyseshat.orbit.orbit_model import (
    compute_visual_orbit
)


from pyseshat.triple.triple_orbit import (
    compute_orbit3_visual,
    compute_orbit3_radial_velocities,
)


# ==========================================================
# Legacy ORBITT2 visual residuals
# ==========================================================

def triple_visual_residuals(
    observations,
    elements: TripleOrbitalElements
):
    """
    Compute weighted visual residuals for the legacy
    ORBITT2 hierarchical-triple formulation.

    The legacy ORBITT2 observation convention is:

        I1
            Outer-pair observation.

            The model is the vector sum of the
            outer orbit and the inner orbit:

                r_I1 = r_outer + r_inner

        I2
            Resolved inner-subsystem observation.

            The model is the inner orbit scaled
            by the astrometric wobble factor B:

                r_I2 = B * r_inner

    Parameters
    ----------
    observations : list[TripleObservation]
        Astrometric observations.

    elements : TripleOrbitalElements
        Triple orbital elements.

    Returns
    -------
    numpy.ndarray
        Flattened weighted Cartesian residual vector.
    """

    residuals = []


    for obs in observations:

        # --------------------------------------------------
        # Observation epoch
        # --------------------------------------------------

        t = np.array([
            obs.epoch
        ])


        # --------------------------------------------------
        # Observation in Cartesian coordinates
        #
        # ORBIT convention:
        #
        #     X = rho cos(theta)
        #     Y = rho sin(theta)
        # --------------------------------------------------

        theta = np.deg2rad(
            obs.theta
        )


        X_obs = (
            obs.rho
            *
            np.cos(theta)
        )


        Y_obs = (
            obs.rho
            *
            np.sin(theta)
        )


        # --------------------------------------------------
        # Wide-orbit observation
        #
        # Legacy ORBITT2:
        #
        #     I1 = outer + inner
        # --------------------------------------------------

        if obs.orbit_id == "I1":

            outer_orbit = compute_visual_orbit(
                t,
                elements.outer
            )


            inner_orbit = compute_visual_orbit(
                t,
                elements.inner
            )


            X_model = (
                outer_orbit["X"][0]
                +
                inner_orbit["X"][0]
            )


            Y_model = (
                outer_orbit["Y"][0]
                +
                inner_orbit["Y"][0]
            )


        # --------------------------------------------------
        # Inner-subsystem observation
        #
        # Legacy ORBITT2:
        #
        #     I2 = B * inner
        # --------------------------------------------------

        elif obs.orbit_id == "I2":

            inner_orbit = compute_visual_orbit(
                t,
                elements.inner
            )


            X_model = (
                elements.B
                *
                inner_orbit["X"][0]
            )


            Y_model = (
                elements.B
                *
                inner_orbit["Y"][0]
            )


        else:

            raise ValueError(
                "Unknown legacy triple "
                f"observation type: {obs.orbit_id}"
            )


        # --------------------------------------------------
        # Weighted Cartesian residuals
        # --------------------------------------------------

        resX = (
            X_obs
            -
            X_model
        ) / obs.sigma


        resY = (
            Y_obs
            -
            Y_model
        ) / obs.sigma


        residuals.extend([
            resX,
            resY
        ])


    return np.asarray(
        residuals,
        dtype=float
    )


# ==========================================================
# ORBIT3 visual residuals
# ==========================================================

def orbit3_visual_residuals(
    observations,
    elements: TripleOrbitalElements
):
    """
    Compute weighted visual residuals using the ORBIT3
    hierarchical-triple formulation.

    ORBIT3 defines:

        I1:
            POS1 = inner visual orbit

        I2:
            POS2 = outer visual orbit
                   + f * inner visual orbit

    Parameters
    ----------
    observations : list[TripleObservation]
        Astrometric observations containing I1 and/or I2
        records.

    elements : TripleOrbitalElements
        ORBIT3 triple orbital elements.

    Returns
    -------
    numpy.ndarray
        Flattened weighted Cartesian residual vector.

    Notes
    -----
    The residuals are expressed in Cartesian coordinates,
    consistent with the existing binary and legacy-triple
    fitting infrastructure.
    """

    residuals = []


    for obs in observations:

        if obs.orbit_id not in (
            "I1",
            "I2"
        ):

            raise ValueError(
                "ORBIT3 visual residuals received "
                f"non-visual observation: "
                f"{obs.orbit_id}"
            )


        # --------------------------------------------------
        # Observation in Cartesian coordinates
        # --------------------------------------------------

        theta = np.deg2rad(
            obs.theta
        )


        X_obs = (
            obs.rho
            *
            np.cos(theta)
        )


        Y_obs = (
            obs.rho
            *
            np.sin(theta)
        )


        # --------------------------------------------------
        # Evaluate the ORBIT3 visual model at this epoch.
        # --------------------------------------------------

        model = compute_orbit3_visual(

            np.array([
                obs.epoch
            ]),

            elements

        )


        if obs.orbit_id == "I1":

            X_model = (
                model["POS1"]["X"][0]
            )


            Y_model = (
                model["POS1"]["Y"][0]
            )


        else:

            X_model = (
                model["POS2"]["X"][0]
            )


            Y_model = (
                model["POS2"]["Y"][0]
            )


        # --------------------------------------------------
        # Weighted Cartesian residuals
        # --------------------------------------------------

        resX = (
            X_obs
            -
            X_model
        ) / obs.sigma


        resY = (
            Y_obs
            -
            Y_model
        ) / obs.sigma


        residuals.extend([
            resX,
            resY
        ])


    return np.asarray(
        residuals,
        dtype=float
    )


# ==========================================================
# ORBIT3 radial-velocity residuals
# ==========================================================

def orbit3_radial_velocity_residuals(
    observations,
    elements: TripleOrbitalElements
):
    """
    Compute weighted radial-velocity residuals for ORBIT3.

    The ORBIT3 radial-velocity observables are:

        Va:
            RV(Aa) =
                RV(inner primary)
                +
                RV(outer primary)

        Vb:
            RV(Ab) =
                RV(inner secondary)
                +
                RV(outer primary)

        Vc:
            RV(B) =
                RV(outer secondary)

    Parameters
    ----------
    observations : list[TripleObservation]
        Radial-velocity observations with orbit_id equal
        to Va, Vb, or Vc.

    elements : TripleOrbitalElements
        ORBIT3 triple orbital elements.

    Returns
    -------
    numpy.ndarray
        Flattened weighted radial-velocity residual vector.
    """

    residuals = []


    for obs in observations:

        if obs.orbit_id not in (
            "Va",
            "Vb",
            "Vc"
        ):

            raise ValueError(
                "ORBIT3 radial-velocity residuals "
                "received non-RV observation: "
                f"{obs.orbit_id}"
            )


        if obs.velocity is None:

            raise ValueError(
                "Radial-velocity observation "
                f"{obs.orbit_id} has no velocity value."
            )


        model = compute_orbit3_radial_velocities(

            np.array([
                obs.epoch
            ]),

            elements

        )


        if obs.orbit_id == "Va":

            model_velocity = (
                model["Aa"][0]
            )


        elif obs.orbit_id == "Vb":

            model_velocity = (
                model["Ab"][0]
            )


        else:

            model_velocity = (
                model["B"][0]
            )


        residual = (

            obs.velocity
            -
            model_velocity

        ) / obs.sigma


        residuals.append(
            residual
        )


    return np.asarray(
        residuals,
        dtype=float
    )


# ==========================================================
# Complete ORBIT3 residual vector
# ==========================================================

def orbit3_residuals(
    observations,
    rv_observations,
    elements: TripleOrbitalElements
):
    """
    Compute the complete weighted ORBIT3 residual vector.

    Parameters
    ----------
    observations : list[TripleObservation]
        Visual observations containing I1 and/or I2 records.

    rv_observations : list[TripleObservation]
        Radial-velocity observations containing Va, Vb,
        and/or Vc records.

    elements : TripleOrbitalElements
        ORBIT3 triple orbital elements.

    Returns
    -------
    numpy.ndarray
        Combined weighted residual vector.

    Notes
    -----
    The vector contains:

        visual residuals
            X1, Y1, X2, Y2 contributions

        followed by:

            RV1, RV2, RV3 contributions

    The exact ordering of individual observations is
    inherited from the supplied observation lists.

    The ordering of residuals does not alter chi-square,
    which is the sum of squared normalized residuals.
    """

    visual_residuals = (
        orbit3_visual_residuals(

            observations,

            elements

        )
    )


    radial_velocity_residuals = (
        orbit3_radial_velocity_residuals(

            rv_observations,

            elements

        )
    )


    return np.concatenate([

        visual_residuals,

        radial_velocity_residuals

    ])


# ==========================================================
# ORBIT3 chi-square
# ==========================================================

def orbit3_chi_square(
    observations,
    rv_observations,
    elements: TripleOrbitalElements
):
    """
    Compute the ORBIT3 weighted chi-square.

    """

    residuals = orbit3_residuals(

        observations,

        rv_observations,

        elements

    )


    return np.sum(
        residuals**2
    )