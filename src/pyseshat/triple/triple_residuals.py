import numpy as np


from pyseshat.triple.triple_elements import (
    TripleOrbitalElements
)


from pyseshat.orbit.orbit_model import (
    compute_visual_orbit
)


# ==========================================================
# Triple visual residuals
# ==========================================================

def triple_visual_residuals(
    observations,
    elements: TripleOrbitalElements
):
    """
    Compute residual vector for a hierarchical
    triple system.

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
        # ORBIT uses:
        #
        #     X = rho cos(theta)
        #     Y = rho sin(theta)
        #
        # with X toward North and Y toward East.
        # --------------------------------------------------

        theta = np.deg2rad(
            obs.theta
        )


        X_obs = (
            obs.rho
            * np.cos(theta)
        )


        Y_obs = (
            obs.rho
            * np.sin(theta)
        )


        # --------------------------------------------------
        # Wide-orbit observation
        #
        # ORBITT2 convention:
        #
        #     I1 = outer measurement
        #
        # The apparent outer-pair position includes the
        # displacement of the inner subsystem. Therefore:
        #
        #     r_I1 =
        #         r_outer
        #         +
        #         r_inner
        #
        # This corresponds to the eph2() construction in
        # the original ORBITT2 implementation.
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
        # ORBITT2 convention:
        #
        #     I2 = resolved inner subsystem
        #
        # The observed astrometric displacement is scaled
        # by the wobble factor B:
        #
        #     r_I2 = B * r_inner
        #
        # The original ORBITT2 source applies this factor
        # to both Cartesian coordinates.
        # --------------------------------------------------

        elif obs.orbit_id == "I2":

            inner_orbit = compute_visual_orbit(
                t,
                elements.inner
            )


            X_model = (
                elements.B
                * inner_orbit["X"][0]
            )


            Y_model = (
                elements.B
                * inner_orbit["Y"][0]
            )


        # --------------------------------------------------
        # Unknown observation type
        # --------------------------------------------------

        else:

            raise ValueError(
                f"Unknown orbit type: "
                f"{obs.orbit_id}"
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