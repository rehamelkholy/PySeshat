from dataclasses import dataclass


@dataclass
class TripleObservation:

    epoch: float

    theta: float

    rho: float

    sigma: float

    orbit_id: str