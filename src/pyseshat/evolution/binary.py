from __future__ import annotations

from dataclasses import dataclass

from pyseshat.analysis.masses import (
    component_masses_from_elements,
)

from pyseshat.evolution.age import (
    CommonAgeFitResult,
    TrackFitResult,
    fit_common_age,
    fit_track,
)

from pyseshat.evolution.models import (
    EvolutionModel,
    EvolutionTrack,
)

from pyseshat.evolution.observables import (
    StellarObservables,
)

from pyseshat.evolution.search import (
    nearest_track,
)

from pyseshat.evolution.solution import (
    BinaryEvolutionSolution,
)

from pyseshat.orbit.orbital_elements import (
    OrbitalElements,
)


@dataclass(frozen=True)
class BinaryTrackFitResult:
    """
    Best-fitting evolutionary solution for a binary system.
    """

    primary: TrackFitResult
    secondary: TrackFitResult

    age_difference: float


def fit_binary_tracks(
    *,
    primary_track: EvolutionTrack,
    secondary_track: EvolutionTrack,
    primary_log_teff: float,
    primary_log_luminosity: float,
    primary_sigma_log_teff: float,
    primary_sigma_log_luminosity: float,
    secondary_log_teff: float,
    secondary_log_luminosity: float,
    secondary_sigma_log_teff: float,
    secondary_sigma_log_luminosity: float,
) -> BinaryTrackFitResult:
    """
    Fit both components of a binary independently.

    This is the first stage of the binary evolutionary fitting.
    """

    primary = fit_track(
        primary_track,
        observables=StellarObservables(
            log_teff=primary_log_teff,
            log_luminosity=primary_log_luminosity,
            sigma_log_teff=primary_sigma_log_teff,
            sigma_log_luminosity=primary_sigma_log_luminosity,
        ),
    )

    secondary = fit_track(
        secondary_track,
        observables=StellarObservables(
            log_teff=secondary_log_teff,
            log_luminosity=secondary_log_luminosity,
            sigma_log_teff=secondary_sigma_log_teff,
            sigma_log_luminosity=secondary_sigma_log_luminosity,
        ),
    )

    return BinaryTrackFitResult(
        primary=primary,
        secondary=secondary,
        age_difference=abs(
            primary.sample.age
            - secondary.sample.age
        ),
    )


def fit_binary_evolution_from_orbit(
    *,
    orbital_elements: OrbitalElements,
    evolution_model: EvolutionModel,
    metallicity: float,
    primary_observables: StellarObservables,
    secondary_observables: StellarObservables,
) -> BinaryEvolutionSolution:
    """
    Fit a binary evolutionary solution using component masses
    obtained from an SB2 orbital solution.

    The workflow is:

        OrbitalElements
            |
            v
        component masses
            |
            v
        nearest evolutionary tracks
            |
            v
        common-age fit
            |
            v
        BinaryEvolutionSolution

    Parameters
    ----------
    orbital_elements
        Fitted SB2 orbital elements. The component-mass
        calculation requires P, e, i, K1, and K2.

    evolution_model
        Evolutionary model containing the available stellar
        evolution tracks.

    metallicity
        Metallicity used when selecting the evolutionary tracks.

    primary_observables
        Observed stellar parameters for the primary.

    secondary_observables
        Observed stellar parameters for the secondary.

    Returns
    -------
    BinaryEvolutionSolution
        Best-fitting common-age evolutionary solution.
    """

    component_masses = (
        component_masses_from_elements(
            orbital_elements
        )
    )

    primary_track = nearest_track(
        evolution_model,
        initial_mass=component_masses["M1"],
        metallicity=metallicity,
    )

    secondary_track = nearest_track(
        evolution_model,
        initial_mass=component_masses["M2"],
        metallicity=metallicity,
    )

    common_age_result = fit_common_age(
        primary_track=primary_track,
        secondary_track=secondary_track,
        primary_observables=primary_observables,
        secondary_observables=secondary_observables,
    )

    return BinaryEvolutionSolution.from_common_age_fit(
        common_age_result,
        metallicity=metallicity,
    )