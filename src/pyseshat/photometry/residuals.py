import numpy as np

from pyseshat.photometry.observations import (
    PhotometricMeasurement
)


def photometric_residual(
    model_magnitude,
    measurement
):
    """
    Calculate the standardized residual for one
    photometric magnitude measurement.

    Parameters
    ----------
    model_magnitude : float
        Synthetic model magnitude.

    measurement : PhotometricMeasurement
        Observed magnitude and its one-standard-
        deviation uncertainty.

    Returns
    -------
    float
        Standardized photometric residual,

            residual
            =
            (model - observed) / uncertainty.

    Raises
    ------
    TypeError
        If measurement is not a
        PhotometricMeasurement.

    ValueError
        If model_magnitude is not finite.
    """

    # ======================================================
    # Validate measurement
    # ======================================================

    if not isinstance(

        measurement,

        PhotometricMeasurement

    ):

        raise TypeError(

            "measurement must be a "
            "PhotometricMeasurement."

        )


    # ======================================================
    # Normalize model magnitude
    # ======================================================

    try:

        model_magnitude = float(

            model_magnitude

        )


    except (

        TypeError,

        ValueError

    ) as error:

        raise TypeError(

            "model_magnitude must be a real number."

        ) from error


    # ======================================================
    # Validate model magnitude
    # ======================================================

    if not np.isfinite(

        model_magnitude

    ):

        raise ValueError(

            "model_magnitude must be finite."

        )


    # ======================================================
    # Calculate standardized residual
    # ======================================================

    return float(

        (

            model_magnitude

            -

            measurement.magnitude

        )

        /

        measurement.uncertainty

    )


def photometric_chi_square(
    model_magnitudes,
    measurements
):
    """
    Calculate the photometric chi-square statistic
    for independent magnitude measurements.

    Parameters
    ----------
    model_magnitudes : mapping
        Mapping from passband identifiers to
        synthetic model magnitudes.

        For example,

            {
                "Johnson B": 7.15,
                "Johnson V": 6.28
            }

    measurements : iterable of PhotometricMeasurement
        Observed independent photometric
        measurements.

    Returns
    -------
    float
        Photometric chi-square statistic,

            chi_square
            =
            sum(residual**2).

    Raises
    ------
    TypeError
        If model_magnitudes is not a mapping or if
        a measurement has an invalid type.

    ValueError
        If no measurements are supplied, if a
        passband is duplicated, or if a required
        model magnitude is missing.
    """

    # ======================================================
    # Validate model-magnitude mapping
    # ======================================================

    if not hasattr(

        model_magnitudes,

        "keys"

    ):

        raise TypeError(

            "model_magnitudes must be a mapping."

        )


    # ======================================================
    # Materialize measurements
    # ======================================================

    measurements = tuple(

        measurements

    )


    if len(

        measurements

    ) == 0:

        raise ValueError(

            "At least one photometric measurement "
            "is required."

        )


    # ======================================================
    # Calculate residuals
    # ======================================================

    residuals = []


    used_passbands = set()


    for measurement in measurements:

        if not isinstance(

            measurement,

            PhotometricMeasurement

        ):

            raise TypeError(

                "Each measurement must be a "
                "PhotometricMeasurement."

            )


        if measurement.passband in used_passbands:

            raise ValueError(

                "Duplicate photometric measurement "
                "for passband: "

                f"{measurement.passband}"

            )


        used_passbands.add(

            measurement.passband

        )


        if measurement.passband not in (

            model_magnitudes

        ):

            raise ValueError(

                "Missing model magnitude for "
                "passband: "

                f"{measurement.passband}"

            )


        residuals.append(

            photometric_residual(

                model_magnitude=(

                    model_magnitudes[

                        measurement.passband

                    ]

                ),

                measurement=measurement

            )

        )


    # ======================================================
    # Calculate chi-square
    # ======================================================

    residual_array = np.asarray(

        residuals,

        dtype=float

    )


    return float(

        np.sum(

            residual_array**2

        )

    )