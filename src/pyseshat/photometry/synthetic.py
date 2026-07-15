import numpy as np

from pyseshat.photometry.filters import (
    PhotometricPassband
)


def synthetic_band_flux(
    wavelength,
    flux,
    passband
):
    """
    Calculate the normalized throughput-weighted flux
    of a spectral energy distribution through a
    photometric passband.

    Parameters
    ----------
    wavelength : array_like
        SED wavelength grid in Angstrom.

    flux : array_like
        Monochromatic SED flux density in

            erg s^-1 cm^-2 Angstrom^-1.

    passband : PhotometricPassband
        Photometric transmission curve.

    Returns
    -------
    float
        Normalized passband-averaged monochromatic flux
        density in

            erg s^-1 cm^-2 Angstrom^-1.

    Raises
    ------
    TypeError
        If passband is not a PhotometricPassband.

    ValueError
        If the SED arrays are invalid, the SED and
        passband have no wavelength overlap, or the
        integrated throughput is zero.

    Notes
    -----
    The synthetic band flux is calculated as

                    integral f_lambda S d lambda
        <f_lambda> = ----------------------------
                       integral S d lambda

    where S is the dimensionless passband throughput.

    This function does not apply extinction or convert
    fluxes to magnitudes.
    """

    # ======================================================
    # Validate passband
    # ======================================================

    if not isinstance(

        passband,

        PhotometricPassband

    ):

        raise TypeError(

            "passband must be a "
            "PhotometricPassband instance."

        )

    # ======================================================
    # Normalize SED arrays
    # ======================================================

    wavelength = np.asarray(

        wavelength,

        dtype=float

    )

    flux = np.asarray(

        flux,

        dtype=float

    )

    # ======================================================
    # Validate dimensions
    # ======================================================

    if wavelength.ndim != 1:

        raise ValueError(

            "wavelength must be a "
            "one-dimensional array."

        )

    if flux.ndim != 1:

        raise ValueError(

            "flux must be a "
            "one-dimensional array."

        )

    # ======================================================
    # Validate array lengths
    # ======================================================

    if (

        len(
            wavelength
        )

        !=

        len(
            flux
        )

    ):

        raise ValueError(

            "wavelength and flux must "
            "have the same length."

        )

    if len(
        wavelength
    ) < 2:

        raise ValueError(

            "The SED must contain at least "
            "two wavelength samples."

        )

    # ======================================================
    # Validate finite values
    # ======================================================

    if not np.all(

        np.isfinite(
            wavelength
        )

    ):

        raise ValueError(

            "wavelength must contain only "
            "finite values."

        )

    if not np.all(

        np.isfinite(
            flux
        )

    ):

        raise ValueError(

            "flux must contain only "
            "finite values."

        )

    # ======================================================
    # Validate wavelength grid
    # ======================================================

    if np.any(

        wavelength <= 0.0

    ):

        raise ValueError(

            "All wavelengths must be positive."

        )

    if np.any(

        np.diff(
            wavelength
        )

        <= 0.0

    ):

        raise ValueError(

            "wavelength must be strictly increasing."

        )

    # ======================================================
    # Validate flux
    # ======================================================

    if np.any(

        flux < 0.0

    ):

        raise ValueError(

            "flux must not contain "
            "negative values."

        )

    # ======================================================
    # Restrict integration to wavelength overlap
    # ======================================================

    overlap = (

        (

            wavelength

            >=

            passband.minimum_wavelength

        )

        &

        (

            wavelength

            <=

            passband.maximum_wavelength

        )

    )

    if np.count_nonzero(

        overlap

    ) < 2:

        raise ValueError(

            "The SED and passband do not have "
            "sufficient wavelength overlap."

        )

    integration_wavelength = wavelength[

        overlap

    ]

    integration_flux = flux[

        overlap

    ]

    # ======================================================
    # Interpolate throughput onto the SED grid
    # ======================================================

    throughput = passband.throughput_at(

        integration_wavelength

    )

    # ======================================================
    # Integrate passband throughput
    # ======================================================

    throughput_integral = np.trapezoid(

        throughput,

        integration_wavelength

    )

    if (

        not np.isfinite(
            throughput_integral
        )

        or

        throughput_integral <= 0.0

    ):

        raise ValueError(

            "The integrated passband throughput "
            "must be positive and finite."

        )

    # ======================================================
    # Integrate throughput-weighted SED
    # ======================================================

    weighted_flux_integral = np.trapezoid(

        (

            integration_flux

            *

            throughput

        ),

        integration_wavelength

    )

    band_flux = (

        weighted_flux_integral

        /

        throughput_integral

    )

    if not np.isfinite(

        band_flux

    ):

        raise ValueError(

            "The calculated synthetic band flux "
            "is not finite."

        )

    return float(

        band_flux

    )