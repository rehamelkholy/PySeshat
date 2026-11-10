from dataclasses import dataclass

import numpy as np


@dataclass(
    frozen=True
)
class PhotometricMeasurement:
    """
    Represent one observed photometric magnitude.

    Parameters
    ----------
    passband : str
        Name or identifier of the photometric
        passband.

    magnitude : float
        Observed apparent magnitude.

    uncertainty : float
        One-standard-deviation uncertainty in
        magnitude.

    Notes
    -----
    This class represents one independent
    magnitude measurement,

        m_obs +/- sigma_m.

    The passband name should identify the
    calibration associated with the observation,
    for example ``"Johnson B"`` or
    ``"Johnson V"``.
    """

    passband: str

    magnitude: float

    uncertainty: float


    def __post_init__(
        self
    ):
        """
        Validate and normalize the measurement.
        """

        # ==================================================
        # Validate passband name
        # ==================================================

        if not isinstance(

            self.passband,

            str

        ):

            raise TypeError(

                "passband must be a string."

            )


        normalized_passband = (

            self.passband.strip()

        )


        if not normalized_passband:

            raise ValueError(

                "passband must not be empty."

            )


        # ==================================================
        # Normalize numerical values
        # ==================================================

        try:

            normalized_magnitude = float(

                self.magnitude

            )


        except (

            TypeError,

            ValueError

        ) as error:

            raise TypeError(

                "magnitude must be a real number."

            ) from error


        try:

            normalized_uncertainty = float(

                self.uncertainty

            )


        except (

            TypeError,

            ValueError

        ) as error:

            raise TypeError(

                "uncertainty must be a real number."

            ) from error


        # ==================================================
        # Validate magnitude
        # ==================================================

        if not np.isfinite(

            normalized_magnitude

        ):

            raise ValueError(

                "magnitude must be finite."

            )


        # ==================================================
        # Validate uncertainty
        # ==================================================

        if not np.isfinite(

            normalized_uncertainty

        ):

            raise ValueError(

                "uncertainty must be finite."

            )


        if normalized_uncertainty <= 0.0:

            raise ValueError(

                "uncertainty must be positive."

            )


        # ==================================================
        # Store normalized values
        # ==================================================

        object.__setattr__(

            self,

            "passband",

            normalized_passband

        )


        object.__setattr__(

            self,

            "magnitude",

            normalized_magnitude

        )


        object.__setattr__(

            self,

            "uncertainty",

            normalized_uncertainty

        )


@dataclass(frozen=True)
class MagnitudeDifference:
    """
    Observed magnitude difference between the two components of
    an unresolved binary, e.g. from speckle interferometry.

    Parameters
    ----------
    passband : str
        Passband in which the difference was measured.
    value : float
        m_secondary - m_primary, in magnitudes.
    uncertainty : float
        One-standard-deviation uncertainty in magnitudes (> 0).

    Notes
    -----
    Combined-light photometry constrains only the sum of the two
    components' fluxes; for components of similar temperature it
    cannot separate the individual temperatures. A measured
    magnitude difference supplies the flux ratio, as in the
    synthetic-photometry approach of Al-Wardat (2002, 2012).
    """

    passband: str
    value: float
    uncertainty: float

    def __post_init__(self):
        if not isinstance(self.passband, str) or not self.passband.strip():
            raise TypeError("passband must be a non-empty string.")
        object.__setattr__(self, "passband", self.passband.strip())
        for name in ("value", "uncertainty"):
            v = float(getattr(self, name))
            if not np.isfinite(v):
                raise ValueError(f"{name} must be finite.")
            object.__setattr__(self, name, v)
        if self.uncertainty <= 0.0:
            raise ValueError("uncertainty must be positive.")
