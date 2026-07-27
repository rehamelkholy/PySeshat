from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class StellarObservables:
    """
    Observed stellar parameters with associated uncertainties.

    Any observable left as None is ignored during χ² evaluation.
    """

    log_teff: float | None = None
    sigma_log_teff: float | None = None

    log_luminosity: float | None = None
    sigma_log_luminosity: float | None = None

    log_radius: float | None = None
    sigma_log_radius: float | None = None

    log_g: float | None = None
    sigma_log_g: float | None = None