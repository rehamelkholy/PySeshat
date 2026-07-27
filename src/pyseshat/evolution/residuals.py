from __future__ import annotations

from dataclasses import dataclass

from pyseshat.evolution.interpolate import EvolutionSample
from pyseshat.evolution.observables import StellarObservables


@dataclass(frozen=True)
class StellarResiduals:
    """
    Residuals between observed and predicted stellar parameters.
    """

    log_teff: float | None = None
    log_luminosity: float | None = None
    log_radius: float | None = None
    log_g: float | None = None

    @property
    def available(self) -> tuple[str, ...]:
        fields = []

        if self.log_teff is not None:
            fields.append("log_teff")

        if self.log_luminosity is not None:
            fields.append("log_luminosity")

        if self.log_radius is not None:
            fields.append("log_radius")

        if self.log_g is not None:
            fields.append("log_g")

        return tuple(fields)


def compute_residuals(
    sample: EvolutionSample,
    observables: StellarObservables,
) -> StellarResiduals:
    """
    Compute observed minus predicted residuals for all available
    stellar observables.

    Any observable not present in the observations is omitted
    (represented by None).
    """

    return StellarResiduals(
        log_teff=(
            observables.log_teff - sample.log_teff
            if observables.log_teff is not None
            else None
        ),
        log_luminosity=(
            observables.log_luminosity - sample.log_luminosity
            if observables.log_luminosity is not None
            else None
        ),
        log_radius=(
            observables.log_radius - sample.log_radius
            if observables.log_radius is not None
            else None
        ),
        log_g=(
            observables.log_g - sample.log_g
            if observables.log_g is not None
            else None
        ),
    )