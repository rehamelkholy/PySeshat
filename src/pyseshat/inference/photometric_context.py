from dataclasses import dataclass


@dataclass(frozen=True)
class PhotometricContext:
    """
    Immutable observational context for unresolved-binary
    photometric inference.

    Parameters
    ----------
    atmosphere_grid
        Stellar atmosphere grid used to synthesize spectra.

    passbands : mapping
        Mapping of passband names to
        PhotometricPassband objects.

    zero_points : mapping
        Mapping of passband names to photometric
        zero points.

    measurements : iterable
        Observed photometric measurements.

    primary_logg : float
        Fixed surface gravity of the primary component.

    secondary_logg : float
        Fixed surface gravity of the secondary component.

    metallicity : float
        Fixed stellar metallicity [M/H].

    distance : float
        Distance to the binary system in parsecs.

    rv : float, default=3.1
        Ratio of total-to-selective extinction.
    magnitude_differences : tuple, default=()
        Observed component magnitude differences
        (photometry.observations.MagnitudeDifference); each adds one
        residual (model - observed) / sigma.
    """

    atmosphere_grid: object

    passbands: object

    zero_points: object

    measurements: object

    primary_logg: float

    secondary_logg: float

    metallicity: float

    distance: float

    rv: float = 3.1
    magnitude_differences: tuple = ()