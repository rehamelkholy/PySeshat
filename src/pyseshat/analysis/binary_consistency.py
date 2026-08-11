from __future__ import annotations

import math

from dataclasses import dataclass

from pyseshat.analysis.masses import (
    total_mass_from_elements,
    total_mass_uncertainty_from_elements,
)

from pyseshat.evolution.binary import (
    fit_binary_evolution_from_orbit,
)

from pyseshat.evolution.interpolate import (
    EvolutionSample,
)

from pyseshat.evolution.models import (
    EvolutionModel,
)

from pyseshat.evolution.observables import (
    StellarObservables,
)

from pyseshat.evolution.solution import (
    BinaryEvolutionSolution,
)

from pyseshat.inference.base_fit_result import (
    BaseFitResult,
)

from pyseshat.orbit.orbital_elements import (
    OrbitalElements,
)

from pyseshat.photometry.evolution_adapter import (
    stellar_observables_from_solution,
)


@dataclass(frozen=True)
class BinaryConsistencyResult:
    """
    Physical consistency diagnostics for a binary system.

    Luminosities are expressed as
    log10(L / L_sun).
    """

    orbital_mass: float

    evolutionary_mass: float

    fractional_mass_difference: float

    primary_observed_log_luminosity: float | None = None

    primary_evolutionary_log_luminosity: float | None = None

    primary_luminosity_sigma: float | None = None

    secondary_observed_log_luminosity: float | None = None

    secondary_evolutionary_log_luminosity: float | None = None

    secondary_luminosity_sigma: float | None = None

    orbital_mass_sigma: float | None = None


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
        orbital_mass
        - evolutionary_mass
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

    variance = (
        sigma1**2
        +
        sigma2**2
    )

    if variance == 0.0:

        raise ValueError(
            "Combined uncertainty must be positive."
        )

    return (
        value1
        - value2
    ) / math.sqrt(
        variance
    )


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
        +
        solution.secondary.stellar_mass
    )


def compute_luminosity_consistency(
    observed: StellarObservables,
    model: EvolutionSample,
) -> float:
    """
    Compute the normalized discrepancy between the
    observed and evolutionary stellar luminosities.

    The comparison is performed in
    log10(L / L_sun).
    """

    if observed.log_luminosity is None:

        raise ValueError(
            "Observed log_luminosity is unavailable."
        )

    if observed.sigma_log_luminosity is None:

        raise ValueError(
            "Observed sigma_log_luminosity "
            "is unavailable."
        )

    if model.log_luminosity is None:

        raise ValueError(
            "Evolutionary log_luminosity "
            "is unavailable."
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
    orbital_mass_sigma: float | None = None,
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

    orbital_mass_sigma
        Optional one-sigma uncertainty in the orbital
        dynamical mass.

    Returns
    -------
    BinaryConsistencyResult
        Summary of the physical consistency diagnostics.
    """

    if (
        orbital_mass_sigma is not None
        and orbital_mass_sigma < 0.0
    ):

        raise ValueError(
            "orbital_mass_sigma must be "
            "non-negative."
        )

    if primary_observables.log_luminosity is None:

        raise ValueError(
            "Primary observed log_luminosity "
            "is unavailable."
        )

    if secondary_observables.log_luminosity is None:

        raise ValueError(
            "Secondary observed log_luminosity "
            "is unavailable."
        )

    if evolution_solution.primary.log_luminosity is None:

        raise ValueError(
            "Primary evolutionary log_luminosity "
            "is unavailable."
        )

    if evolution_solution.secondary.log_luminosity is None:

        raise ValueError(
            "Secondary evolutionary log_luminosity "
            "is unavailable."
        )

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

        primary_observed_log_luminosity=(
            primary_observables.log_luminosity
        ),

        primary_evolutionary_log_luminosity=(
            evolution_solution.primary.log_luminosity
        ),

        primary_luminosity_sigma=(
            compute_luminosity_consistency(

                primary_observables,

                evolution_solution.primary,
            )
        ),

        secondary_observed_log_luminosity=(
            secondary_observables.log_luminosity
        ),

        secondary_evolutionary_log_luminosity=(
            evolution_solution.secondary.log_luminosity
        ),

        secondary_luminosity_sigma=(
            compute_luminosity_consistency(

                secondary_observables,

                evolution_solution.secondary,
            )
        ),

        orbital_mass_sigma=(
            orbital_mass_sigma
        ),
    )


def build_binary_consistency_from_elements(
    *,
    orbital_elements: OrbitalElements,
    parallax_mas: float,
    evolution_solution: BinaryEvolutionSolution,
    primary_observables: StellarObservables,
    secondary_observables: StellarObservables,
    orbital_mass_sigma: float | None = None,
) -> BinaryConsistencyResult:
    """
    Construct binary physical-consistency diagnostics
    directly from an orbital solution.
    """

    orbital_mass = total_mass_from_elements(

        orbital_elements,

        parallax_mas=parallax_mas,
    )

    return build_binary_consistency_result(

        orbital_mass=orbital_mass,

        evolution_solution=evolution_solution,

        primary_observables=primary_observables,

        secondary_observables=secondary_observables,

        orbital_mass_sigma=(
            orbital_mass_sigma
        ),
    )


def build_binary_consistency_from_fit_results(
    *,
    orbital_fit_result: BaseFitResult,
    photometric_fit_result: BaseFitResult,
    parallax_mas: float,
    sigma_parallax_mas: float | None = None,
    evolution_model: EvolutionModel,
    metallicity: float,
) -> BinaryConsistencyResult:
    """
    Build the complete binary consistency result
    from fitted orbital and photometric solutions.

    The orbital fit provides the orbital elements and
    their uncertainties. The photometric fit provides
    the stellar parameters and their uncertainties.

    If ``sigma_parallax_mas`` is supplied, the uncertainty
    in the dynamical mass is propagated analytically.
    Otherwise ``orbital_mass_sigma`` is ``None``.
    """

    orbital_elements = (
        orbital_fit_result.elements
    )

    if not isinstance(
        orbital_elements,
        OrbitalElements,
    ):

        raise TypeError(
            "orbital_fit_result.elements must be "
            "an OrbitalElements instance."
        )

    (
        primary_observables,
        secondary_observables,
    ) = stellar_observables_from_solution(

        parameters=(
            photometric_fit_result.elements
        ),

        uncertainties=(
            photometric_fit_result.uncertainties
        ),
    )

    evolution_solution = (
        fit_binary_evolution_from_orbit(

            orbital_elements=(
                orbital_elements
            ),

            evolution_model=(
                evolution_model
            ),

            metallicity=metallicity,

            primary_observables=(
                primary_observables
            ),

            secondary_observables=(
                secondary_observables
            ),
        )
    )

    orbital_mass_sigma = None

    if sigma_parallax_mas is not None:

        orbital_mass_sigma = (
            total_mass_uncertainty_from_elements(

                elements=(
                    orbital_elements
                ),

                uncertainties=(
                    orbital_fit_result.uncertainties
                ),

                parallax_mas=parallax_mas,

                sigma_parallax_mas=(
                    sigma_parallax_mas
                ),
            )
        )

    return build_binary_consistency_from_elements(

        orbital_elements=(
            orbital_elements
        ),

        parallax_mas=parallax_mas,

        evolution_solution=(
            evolution_solution
        ),

        primary_observables=(
            primary_observables
        ),

        secondary_observables=(
            secondary_observables
        ),

        orbital_mass_sigma=(
            orbital_mass_sigma
        ),
    )