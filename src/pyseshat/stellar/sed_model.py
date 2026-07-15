import numpy as np

from pyseshat.stellar.atmospheres import (
    AtmosphereModel
)


# ==========================================================
# Physical constants
# ==========================================================

SOLAR_RADIUS_CM = 6.957e10

PARSEC_CM = 3.0856775814913673e18


# ==========================================================
# Scale stellar surface flux
# ==========================================================

def scale_surface_flux(
    atmosphere,
    radius,
    distance
):
    """
    Scale a stellar-atmosphere surface flux to the
    monochromatic flux received by an observer.

    Parameters
    ----------
    atmosphere : AtmosphereModel
        Stellar-atmosphere spectrum. Its flux is interpreted
        as the monochromatic stellar-surface flux

            F_lambda

        in erg s^-1 cm^-2 Angstrom^-1.

    radius : float
        Stellar radius in solar radii.

    distance : float
        Distance to the star in parsecs.

    Returns
    -------
    AtmosphereModel
        New atmosphere model containing the observer-frame
        monochromatic flux

            f_lambda = F_lambda (R / d)^2.

        The wavelength grid and atmosphere parameters are
        preserved.

    Raises
    ------
    TypeError
        If atmosphere is not an AtmosphereModel.

    ValueError
        If radius or distance is not positive and finite.

    Notes
    -----
    This function applies geometric dilution only. It does
    not apply interstellar extinction.
    """

    # ======================================================
    # Validate atmosphere
    # ======================================================

    if not isinstance(

        atmosphere,

        AtmosphereModel

    ):

        raise TypeError(

            "atmosphere must be an "
            "AtmosphereModel instance."

        )

    # ======================================================
    # Normalize physical parameters
    # ======================================================

    radius = float(

        radius

    )

    distance = float(

        distance

    )

    # ======================================================
    # Validate physical parameters
    # ======================================================

    if (

        not np.isfinite(
            radius
        )

        or

        radius <= 0.0

    ):

        raise ValueError(

            "radius must be a positive "
            "finite value."

        )

    if (

        not np.isfinite(
            distance
        )

        or

        distance <= 0.0

    ):

        raise ValueError(

            "distance must be a positive "
            "finite value."

        )

    # ======================================================
    # Convert to cgs units
    # ======================================================

    radius_cm = (

        radius

        *

        SOLAR_RADIUS_CM

    )

    distance_cm = (

        distance

        *

        PARSEC_CM

    )

    # ======================================================
    # Geometric dilution
    # ======================================================

    scale_factor = (

        radius_cm

        /

        distance_cm

    ) ** 2

    observed_flux = (

        atmosphere.flux

        *

        scale_factor

    )

    # ======================================================
    # Construct scaled spectrum
    # ======================================================

    return AtmosphereModel(

        wavelength=(

            atmosphere
            .wavelength
            .copy()

        ),

        flux=observed_flux,

        teff=atmosphere.teff,

        logg=atmosphere.logg,

        metallicity=atmosphere.metallicity

    )

def composite_binary_sed(
    primary,
    secondary,
    primary_radius,
    secondary_radius,
    distance
):
    """
    Construct the unresolved spectral energy distribution
    of a binary stellar system.

    Parameters
    ----------
    primary : AtmosphereModel
        Stellar-atmosphere spectrum of the primary
        component.

    secondary : AtmosphereModel
        Stellar-atmosphere spectrum of the secondary
        component.

    primary_radius : float
        Radius of the primary component in solar radii.

    secondary_radius : float
        Radius of the secondary component in solar radii.

    distance : float
        Distance to the binary system in parsecs.

    Returns
    -------
    tuple of numpy.ndarray
        Two arrays containing:

        wavelength
            Common wavelength grid in Angstrom.

        flux
            Composite monochromatic flux received by the
            observer in

                erg s^-1 cm^-2 Angstrom^-1.

    Raises
    ------
    TypeError
        If either component is not an AtmosphereModel.

    ValueError
        If the two atmosphere models do not use identical
        wavelength grids.

    Notes
    -----
    The unresolved binary flux is

        f_lambda =

            F_lambda,1 (R_1 / d)^2

            +

            F_lambda,2 (R_2 / d)^2.

    Interstellar extinction is not applied.
    """

    # ======================================================
    # Validate atmosphere models
    # ======================================================

    if not isinstance(

        primary,

        AtmosphereModel

    ):

        raise TypeError(

            "primary must be an "
            "AtmosphereModel instance."

        )

    if not isinstance(

        secondary,

        AtmosphereModel

    ):

        raise TypeError(

            "secondary must be an "
            "AtmosphereModel instance."

        )

    # ======================================================
    # Validate wavelength grids
    # ======================================================

    same_shape = (

        primary.wavelength.shape

        ==

        secondary.wavelength.shape

    )

    same_values = (

        same_shape

        and

        np.array_equal(

            primary.wavelength,

            secondary.wavelength

        )

    )

    if not same_values:

        raise ValueError(

            "primary and secondary atmosphere models "
            "must use identical wavelength grids."

        )

    # ======================================================
    # Scale component spectra
    # ======================================================

    primary_observed = scale_surface_flux(

        atmosphere=primary,

        radius=primary_radius,

        distance=distance

    )

    secondary_observed = scale_surface_flux(

        atmosphere=secondary,

        radius=secondary_radius,

        distance=distance

    )

    # ======================================================
    # Construct unresolved composite spectrum
    # ======================================================

    composite_flux = (

        primary_observed.flux

        +

        secondary_observed.flux

    )

    return (

        primary.wavelength.copy(),

        composite_flux

    )