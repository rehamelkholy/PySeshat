from __future__ import annotations

from dataclasses import dataclass


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