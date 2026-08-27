from dataclasses import replace

from pyseshat.inference.joint_parameters import (
    JointParameters,
)

from pyseshat.photometry.parameters import (
    update_parameters,
)

from pyseshat.inference.joint_problem import (
    JointInferenceProblem,
)

from pyseshat.inference.likelihood import (
    log_likelihood_from_problem,
)

from pyseshat.inference.photometric_likelihood import (
    photometric_log_likelihood_from_problem,
)

# NOTE: `pyseshat.inference.evolution_likelihood.evolution_log_likelihood`
# is a real, working Gaussian likelihood for a single stellar-evolution
# sample (see that module). It is deliberately NOT called from here yet:
# `JointInferenceProblem.evolution_context` is currently typed `Any` with
# no defined contract for how per-component EvolutionSample/
# StellarObservables pairs would be attached, so there is nothing well
# -defined to pass it. Wire it in once that contract exists -- don't
# reach for this function just to silence the NotImplementedError below.


def joint_log_likelihood(
    orbit_fitted_values,
    joint_problem: JointInferenceProblem,
):
    """
    Evaluate the joint Gaussian log-likelihood for
    orbital and photometric inference.

    Parameters
    ----------
    orbit_fitted_values : array_like
        Current fitted orbital parameter vector.

    joint_problem : JointInferenceProblem
        Canonical joint inference problem.

    Returns
    -------
    float
        Joint Gaussian log-likelihood

            logL =
                logL_orbit
                +
                logL_photometry
    """

    if not isinstance(

        joint_problem,

        JointInferenceProblem

    ):

        raise TypeError(

            "joint_problem must be a "
            "JointInferenceProblem instance."

        )

    orbit_log_likelihood = (

        log_likelihood_from_problem(

            fitted_values=orbit_fitted_values,

            problem=joint_problem.orbit

        )

    )

    photometric_log_likelihood = (

        photometric_log_likelihood_from_problem(

            joint_problem.photometry

        )

    )

    evolution_log_like = 0.0

    if joint_problem.has_evolution_constraints:

        #
        # Silently contributing 0.0 here would let a real
        # evolution constraint be dropped from the likelihood
        # with no signal at all -- raise instead, consistent
        # with joint_log_likelihood_from_parameters below.
        #
        raise NotImplementedError(
            "Evolution likelihood integration "
            "is not yet implemented."
        )

    return (

        orbit_log_likelihood

        +

        photometric_log_likelihood

        +

        evolution_log_like

    )

def joint_log_likelihood_from_parameters(
    parameters: JointParameters,
    joint_problem,
):
    """
    Evaluate the joint log-likelihood from a canonical
    JointParameters instance.

    Parameters
    ----------
    parameters : JointParameters
        Current optimizer state.

    joint_problem : JointInferenceProblem
        Canonical joint inference problem.

    Returns
    -------
    float
        Joint Gaussian log-likelihood.
    """

    #
    # Update the orbital problem.
    #
    orbit_problem = replace(
        joint_problem.orbit_problem,
        elements=parameters.orbital,
    )

    #
    # Update the photometric problem.
    #
    photometric_problem = replace(
        joint_problem.photometric_problem,
        parameters=parameters.photometric,
    )

    updated_problem = replace(
        joint_problem,
        orbit_problem=orbit_problem,
        photometric_problem=photometric_problem,
    )

    evolution_log_like = 0.0

    if updated_problem.has_evolution_constraints:

        #
        # The evolution likelihood will be evaluated
        # from the current photometric solution.
        #
        raise NotImplementedError(
            "Evolution likelihood integration "
            "is not yet implemented."
        )

    #
    # Reuse the existing implementation.
    #
    return (

        joint_log_likelihood(

            parameters.orbital.fitted_parameters(),

            updated_problem,

        )

        +

        evolution_log_like

    )