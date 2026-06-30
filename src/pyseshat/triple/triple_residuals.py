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
    Compute residual vector for hierarchical
    triple system.

    Parameters
    ----------
    observations : list[TripleObservation]

    elements : TripleOrbitalElements

    Returns
    -------
    residual vector
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
        # Wide orbit observation
        # --------------------------------------------------

        if obs.orbit_id == "I1":

            orbit = compute_visual_orbit(
                t,
                elements.outer
            )

            X_model = orbit["X"][0]
            Y_model = orbit["Y"][0]

        # --------------------------------------------------
        # Inner subsystem observation
        # --------------------------------------------------

        elif obs.orbit_id == "I2":

            orbit = compute_visual_orbit(
                t,
                elements.inner
            )

            X_model = (
                elements.B
                * orbit["X"][0]
            )

            Y_model = (
                elements.B
                * orbit["Y"][0]
            )

        else:

            raise ValueError(
                f"Unknown orbit type: "
                f"{obs.orbit_id}"
            )

        # --------------------------------------------------
        # Weighted residuals
        # --------------------------------------------------

        resX = (
            X_obs - X_model
        ) / obs.sigma

        resY = (
            Y_obs - Y_model
        ) / obs.sigma

        residuals.extend([
            resX,
            resY
        ])

    return np.asarray(
        residuals
    )