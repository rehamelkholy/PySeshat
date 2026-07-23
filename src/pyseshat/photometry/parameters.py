from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PhotometricParameters:
    """
    Immutable container for the free parameters of the
    unresolved-binary photometric model.

    Parameters
    ----------
    primary_temperature : float
        Effective temperature of the primary star [K].

    secondary_temperature : float
        Effective temperature of the secondary star [K].

    primary_radius : float
        Radius of the primary star [R_sun].

    secondary_radius : float
        Radius of the secondary star [R_sun].

    ebv : float
        Color excess E(B-V).
    """

    primary_temperature: float
    secondary_temperature: float
    primary_radius: float
    secondary_radius: float
    ebv: float

    # ======================================================
    # Parameter vector
    # ======================================================

    def fitted_parameters(
        self
    ):
        """
        Return the parameter vector used by the optimizer.
        """

        return np.asarray(

            [

                self.primary_temperature,

                self.secondary_temperature,

                self.primary_radius,

                self.secondary_radius,

                self.ebv

            ],

            dtype=float

        )

    # ======================================================
    # Parameter names
    # ======================================================

    @staticmethod
    def fitted_parameter_names():
        """
        Return the optimizer parameter names.
        """

        return [

            "primary_temperature",

            "secondary_temperature",

            "primary_radius",

            "secondary_radius",

            "ebv"

        ]


# ==========================================================
# Parameter update
# ==========================================================

def update_parameters(
    parameters,
    values
):
    """
    Construct a new PhotometricParameters object from an
    optimized parameter vector.

    Parameters
    ----------
    parameters : PhotometricParameters
        Template parameter object.

    values : array_like
        Optimized parameter vector.

    Returns
    -------
    PhotometricParameters
    """

    values = np.asarray(

        values,

        dtype=float

    )


    if values.shape != (5,):

        raise ValueError(

            "Photometric parameter vector must "
            "contain exactly five values."

        )


    return PhotometricParameters(

        primary_temperature=float(

            values[0]

        ),

        secondary_temperature=float(

            values[1]

        ),

        primary_radius=float(

            values[2]

        ),

        secondary_radius=float(

            values[3]

        ),

        ebv=float(

            values[4]

        )

    )