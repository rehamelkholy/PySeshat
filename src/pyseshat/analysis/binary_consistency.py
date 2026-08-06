from __future__ import annotations

import math

from dataclasses import dataclass

from pyseshat.evolution.solution import (
    BinaryEvolutionSolution,
)

from pyseshat.evolution.interpolate import EvolutionSample

from pyseshat.evolution.observables import StellarObservables


@dataclass(frozen=True)
class BinaryConsistencyResult:
    """
    Physical consistency diagnostics for a binary system.
    """

    orbital_mass: float

    evolutionary_mass: float

    fractional_mass_difference: float

    primary_luminosity_sigma: float

    secondary_luminosity_sigma: float

def compute_mass_consistency(
    orbital_mass: float,
    evolutionary_mass: float,
) -> float:
    """
    Compute the fractional difference between
    orbital and evolutionary total masses.

    Parameters
    ----------
    orbital_mass
        Total mass from the orbital solution.

    evolutionary_mass
        Total mass inferred from stellar evolution.

    Returns
    -------
    float
        Fractional mass difference

            |M_orb - M_evol| / M_orb
    """

    if orbital_mass <= 0.0:
        raise ValueError(
            "orbital_mass must be positive."
        )

    if evolutionary_mass <= 0.0:
        raise ValueError(
            "evolutionary_mass must be positive."
        )

    return abs(
        orbital_mass - evolutionary_mass
    ) / orbital_mass

def normalized_difference(
    value1: float,
    sigma1: float,
    value2: float,
    sigma2: float,
) -> float:
    """
    Compute the normalized difference between two
    independent measurements.

    Parameters
    ----------
    value1, value2
        Measured values.

    sigma1, sigma2
        One-sigma uncertainties.

    Returns
    -------
    float
        Normalized discrepancy

            (x1 - x2) /
            sqrt(s1² + s2²)
    """

    if sigma1 < 0.0:
        raise ValueError(
            "sigma1 must be non-negative."
        )

    if sigma2 < 0.0:
        raise ValueError(
            "sigma2 must be non-negative."
        )

    variance = sigma1**2 + sigma2**2

    if variance == 0.0:
        raise ValueError(
            "Combined uncertainty must be positive."
        )

    return (
        value1 - value2
    ) / math.sqrt(variance)

def evolutionary_total_mass(
    solution: BinaryEvolutionSolution,
) -> float:
    """
    Return the total stellar mass implied by an
    evolutionary solution.

    Returns
    -------
    float
        Total stellar mass.
    """

    return (
        solution.primary.stellar_mass
        + solution.secondary.stellar_mass
    )

def compute_luminosity_consistency(
    observed: StellarObservables,
    model: EvolutionSample,
) -> float:
    """
    Compute the normalized discrepancy between the
    observed and evolutionary stellar luminosities.

    Parameters
    ----------
    observed
        Observed stellar parameters.

    model
        Best-fitting evolutionary model.

    Returns
    -------
    float
        Normalized luminosity discrepancy.
    """

    if observed.log_luminosity is None:
        raise ValueError(
            "Observed log_luminosity is unavailable."
        )

    if observed.sigma_log_luminosity is None:
        raise ValueError(
            "Observed sigma_log_luminosity is unavailable."
        )

    if model.log_luminosity is None:
        raise ValueError(
            "Evolutionary log_luminosity is unavailable."
        )

    return normalized_difference(
        observed.log_luminosity,
        observed.sigma_log_luminosity,
        model.log_luminosity,
        0.0,
    )

def build_binary_consistency_result(
    *,
    orbital_mass: float,
    evolution_solution: BinaryEvolutionSolution,
    primary_observables: StellarObservables,
    secondary_observables: StellarObservables,
) -> BinaryConsistencyResult:
    """
    Construct the physical consistency diagnostics for
    a binary system.

    Parameters
    ----------
    orbital_mass
        Total system mass derived from orbital dynamics.

    evolution_solution
        Best-fitting binary evolutionary solution.

    primary_observables
        Observed properties of the primary component.

    secondary_observables
        Observed properties of the secondary component.

    Returns
    -------
    BinaryConsistencyResult
        Summary of the physical consistency diagnostics.
    """

    evolutionary_mass = evolutionary_total_mass(
        evolution_solution
    )

    return BinaryConsistencyResult(

        orbital_mass=orbital_mass,

        evolutionary_mass=evolutionary_mass,

        fractional_mass_difference=(
            compute_mass_consistency(
                orbital_mass,
                evolutionary_mass,
            )
        ),

        primary_luminosity_sigma=(
            compute_luminosity_consistency(
                primary_observables,
                evolution_solution.primary,
            )
        ),

        secondary_luminosity_sigma=(
            compute_luminosity_consistency(
                secondary_observables,
                evolution_solution.secondary,
            )
        ),
    )