"""
Posterior probability for joint orbital and
photometric Bayesian inference.
"""

import numpy as np

from pyseshat.inference.joint_problem import (
    JointInferenceProblem,
)

from pyseshat.inference.posterior import (
    log_posterior_from_problem,
)

from pyseshat.inference.photometric_posterior import (
    photometric_log_posterior_from_problem,
)


def joint_log_posterior(
    orbit_fitted_values,
    joint_problem: JointInferenceProblem,
):
    """
    Evaluate the joint Bayesian posterior for a binary
    system using orbital and photometric constraints.

    Parameters
    ----------
    orbit_fitted_values : array_like
        Current fitted orbital parameter vector.

    joint_problem : JointInferenceProblem
        Canonical joint inference problem.

    Returns
    -------
    float
        Joint log-posterior

            logP =
                logP_orbit
                +
                logP_photometry
    """

    if not isinstance(

        joint_problem,

        JointInferenceProblem

    ):

        raise TypeError(

            "joint_problem must be a "
            "JointInferenceProblem instance."

        )

    orbit_log_posterior = (

        log_posterior_from_problem(

            fitted_values=orbit_fitted_values,

            problem=joint_problem.orbit

        )

    )

    photometric_log_posterior = (

        photometric_log_posterior_from_problem(

            joint_problem.photometry

        )

    )

    if (

        not np.isfinite(

            orbit_log_posterior

        )

        or

        not np.isfinite(

            photometric_log_posterior

        )

    ):

        return -np.inf

    return (

        orbit_log_posterior

        +

        photometric_log_posterior

    )