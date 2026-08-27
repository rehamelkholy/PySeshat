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
    The public function retains the existing observation-
    level ordering for backward compatibility.

    The complete ORBIT3 optimizer residual vector is assembled
    separately by ``orbit3_residuals()`` in explicit ORBIT3
    component blocks.
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

    Notes
    -----
    The public function retains the supplied observation
    ordering. The complete ORBIT3 residual vector is assembled
    separately by ``orbit3_residuals()`` in explicit RV1,
    RV2, and RV3 blocks.
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
# ORBIT3 complete visual residual blocks
# ==========================================================

def _orbit3_visual_residual_blocks(
    observations,
    elements: TripleOrbitalElements
):
    """
    Construct the ORBIT3 visual residual blocks.

    The ORBIT3 ordering is:

        X1(all I1)
        Y1(all I1)
        X2(all I2)
        Y2(all I2)

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray]
        X1, Y1, X2, Y2 weighted residual blocks.
    """

    I1_observations = [
        obs
        for obs in observations
        if obs.orbit_id == "I1"
    ]

    I2_observations = [
        obs
        for obs in observations
        if obs.orbit_id == "I2"
    ]

    invalid = [
        obs.orbit_id
        for obs in observations
        if obs.orbit_id not in ("I1", "I2")
    ]

    if invalid:

        raise ValueError(
            "ORBIT3 visual residuals received "
            f"non-visual observation types: {invalid}"
        )

    X1 = []
    Y1 = []
    X2 = []
    Y2 = []

    # ------------------------------------------------------
    # I1 -> X1, Y1
    # ------------------------------------------------------

    for obs in I1_observations:

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

        model = compute_orbit3_visual(

            np.array([
                obs.epoch
            ]),

            elements

        )

        X_model = (
            model["POS1"]["X"][0]
        )

        Y_model = (
            model["POS1"]["Y"][0]
        )

        X1.append(
            (
                X_obs
                -
                X_model
            )
            /
            obs.sigma
        )

        Y1.append(
            (
                Y_obs
                -
                Y_model
            )
            /
            obs.sigma
        )

    # ------------------------------------------------------
    # I2 -> X2, Y2
    # ------------------------------------------------------

    for obs in I2_observations:

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

        model = compute_orbit3_visual(

            np.array([
                obs.epoch
            ]),

            elements

        )

        X_model = (
            model["POS2"]["X"][0]
        )

        Y_model = (
            model["POS2"]["Y"][0]
        )

        X2.append(
            (
                X_obs
                -
                X_model
            )
            /
            obs.sigma
        )

        Y2.append(
            (
                Y_obs
                -
                Y_model
            )
            /
            obs.sigma
        )

    return (
        np.asarray(X1, dtype=float),
        np.asarray(Y1, dtype=float),
        np.asarray(X2, dtype=float),
        np.asarray(Y2, dtype=float),
    )


# ==========================================================
# ORBIT3 complete RV residual blocks
# ==========================================================

def _orbit3_rv_residual_blocks(
    rv_observations,
    elements: TripleOrbitalElements
):
    """
    Construct the ORBIT3 radial-velocity residual blocks.

    The ORBIT3 ordering is:

        RV1 = Va
        RV2 = Vb
        RV3 = Vc
    """

    Va_observations = [
        obs
        for obs in rv_observations
        if obs.orbit_id == "Va"
    ]

    Vb_observations = [
        obs
        for obs in rv_observations
        if obs.orbit_id == "Vb"
    ]

    Vc_observations = [
        obs
        for obs in rv_observations
        if obs.orbit_id == "Vc"
    ]

    invalid = [
        obs.orbit_id
        for obs in rv_observations
        if obs.orbit_id not in (
            "Va",
            "Vb",
            "Vc"
        )
    ]

    if invalid:

        raise ValueError(
            "ORBIT3 radial-velocity residuals "
            f"received invalid observation types: {invalid}"
        )

    RV1 = []
    RV2 = []
    RV3 = []

    # ------------------------------------------------------
    # Va -> RV1
    # ------------------------------------------------------

    for obs in Va_observations:

        if obs.velocity is None:

            raise ValueError(
                "Radial-velocity observation "
                "Va has no velocity value."
            )

        model = compute_orbit3_radial_velocities(

            np.array([
                obs.epoch
            ]),

            elements

        )

        RV1.append(
            (
                obs.velocity
                -
                model["Aa"][0]
            )
            /
            obs.sigma
        )

    # ------------------------------------------------------
    # Vb -> RV2
    # ------------------------------------------------------

    for obs in Vb_observations:

        if obs.velocity is None:

            raise ValueError(
                "Radial-velocity observation "
                "Vb has no velocity value."
            )

        model = compute_orbit3_radial_velocities(

            np.array([
                obs.epoch
            ]),

            elements

        )

        RV2.append(
            (
                obs.velocity
                -
                model["Ab"][0]
            )
            /
            obs.sigma
        )

    # ------------------------------------------------------
    # Vc -> RV3
    # ------------------------------------------------------

    for obs in Vc_observations:

        if obs.velocity is None:

            raise ValueError(
                "Radial-velocity observation "
                "Vc has no velocity value."
            )

        model = compute_orbit3_radial_velocities(

            np.array([
                obs.epoch
            ]),

            elements

        )

        RV3.append(
            (
                obs.velocity
                -
                model["B"][0]
            )
            /
            obs.sigma
        )

    return (
        np.asarray(RV1, dtype=float),
        np.asarray(RV2, dtype=float),
        np.asarray(RV3, dtype=float),
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

    ORBIT3 residual ordering:

        X1(all I1)
        Y1(all I1)
        X2(all I2)
        Y2(all I2)
        RV1(all Va)
        RV2(all Vb)
        RV3(all Vc)

    This explicit block ordering matches the ORBIT3
    component-level residual construction.

    The public ``orbit3_visual_residuals()`` and
    ``orbit3_radial_velocity_residuals()`` functions retain
    their existing observation-level behavior. This function
    is the optimizer-facing ORBIT3 residual vector.
    """

    X1, Y1, X2, Y2 = (
        _orbit3_visual_residual_blocks(
            observations,
            elements
        )
    )

    RV1, RV2, RV3 = (
        _orbit3_rv_residual_blocks(
            rv_observations,
            elements
        )
    )

    return np.concatenate([

        X1,
        Y1,
        X2,
        Y2,
        RV1,
        RV2,
        RV3,

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