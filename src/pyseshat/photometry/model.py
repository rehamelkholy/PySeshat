import numpy as np

from pyseshat.photometry.extinction import (
    apply_extinction,
    ccm89_optical_extinction
)

from pyseshat.photometry.synthetic import (
    pogson_magnitude,
    synthetic_band_flux
)

from pyseshat.stellar.sed_model import (
    composite_binary_sed
)

from pyseshat.photometry.residuals import (
    photometric_chi_square
)


def synthetic_binary_magnitudes(
    atmosphere_grid,
    passbands,
    zero_points,
    primary_temperature,
    secondary_temperature,
    primary_logg,
    secondary_logg,
    metallicity,
    primary_radius,
    secondary_radius,
    distance,
    ebv=0.0,
    rv=3.1
):
    """
    Calculate extincted synthetic magnitudes for an
    unresolved binary system.

    Parameters
    ----------
    atmosphere_grid
        Stellar-atmosphere grid providing an
        ``interpolate()`` method.

    passbands : mapping
        Mapping from passband identifiers to
        photometric-passband objects.

    zero_points : mapping
        Mapping from passband identifiers to
        positive flux-density zero points.

        Each zero point must use the same physical
        units as the synthetic band flux.

    primary_temperature : float
        Primary effective temperature in kelvin.

    secondary_temperature : float
        Secondary effective temperature in kelvin.

    primary_logg : float
        Primary surface gravity, log10(g).

    secondary_logg : float
        Secondary surface gravity, log10(g).

    metallicity : float
        Atmosphere-grid metallicity.

    primary_radius : float
        Primary radius in solar radii.

    secondary_radius : float
        Secondary radius in solar radii.

    distance : float
        Distance in parsecs.

    ebv : float, default=0.0
        Color excess E(B-V), in magnitudes.

    rv : float, default=3.1
        Ratio of total-to-selective extinction.

    Returns
    -------
    dict
        Mapping from passband identifiers to
        synthetic apparent magnitudes.

    Raises
    ------
    TypeError
        If passbands or zero_points is not a mapping.

    ValueError
        If no passbands are supplied, if a zero point
        is missing or invalid, or if a passband lies
        outside the implemented optical CCM89 domain.

    Notes
    -----
    The current implementation uses only the optical
    CCM89 extinction branch,

        1.1 <= x < 3.3 inverse micrometres.

    Therefore, every requested passband must lie
    entirely inside the corresponding wavelength
    interval.
    """

    # ======================================================
    # Validate mappings
    # ======================================================

    if not hasattr(

        passbands,

        "items"

    ):

        raise TypeError(

            "passbands must be a mapping."

        )


    if not hasattr(

        zero_points,

        "keys"

    ):

        raise TypeError(

            "zero_points must be a mapping."

        )


    if len(

        passbands

    ) == 0:

        raise ValueError(

            "At least one passband is required."

        )


    # ======================================================
    # Validate zero-point coverage
    # ======================================================

    normalized_zero_points = {}


    for name in passbands:

        if name not in zero_points:

            raise ValueError(

                "Missing zero point for passband: "

                f"{name}"

            )


        try:

            zero_point = float(

                zero_points[

                    name

                ]

            )


        except (

            TypeError,

            ValueError

        ) as error:

            raise TypeError(

                "Photometric zero points must be "
                "real numbers."

            ) from error


        if (

            not np.isfinite(

                zero_point

            )

            or

            zero_point <= 0.0

        ):

            raise ValueError(

                "Photometric zero points must be "
                "finite and positive."

            )


        normalized_zero_points[

            name

        ] = zero_point


    # ======================================================
    # Validate color excess
    # ======================================================

    try:

        ebv = float(

            ebv

        )


    except (

        TypeError,

        ValueError

    ) as error:

        raise TypeError(

            "ebv must be a real number."

        ) from error


    if not np.isfinite(

        ebv

    ):

        raise ValueError(

            "ebv must be finite."

        )


    if ebv < 0.0:

        raise ValueError(

            "ebv must not be negative."

        )

    # ======================================================
    # Interpolate component atmospheres
    # ======================================================

    primary = atmosphere_grid.interpolate(

        teff=primary_temperature,

        logg=primary_logg,

        metallicity=metallicity

    )


    secondary = atmosphere_grid.interpolate(

        teff=secondary_temperature,

        logg=secondary_logg,

        metallicity=metallicity

    )


    # ======================================================
    # Construct intrinsic unresolved binary SED
    # ======================================================

    wavelength, intrinsic_flux = (

        composite_binary_sed(

            primary=primary,

            secondary=secondary,

            primary_radius=primary_radius,

            secondary_radius=secondary_radius,

            distance=distance

        )

    )


    # ======================================================
    # Determine required wavelength interval
    # ======================================================

    minimum_required_wavelength = min(

        passband.minimum_wavelength

        for passband in passbands.values()

    )


    maximum_required_wavelength = max(

        passband.maximum_wavelength

        for passband in passbands.values()

    )


    # ======================================================
    # Restrict SED to the required wavelength interval
    # ======================================================

    wavelength_mask = (

        (

            wavelength

            >=

            minimum_required_wavelength

        )

        &

        (

            wavelength

            <=

            maximum_required_wavelength

        )

    )


    model_wavelength = wavelength[

        wavelength_mask

    ]


    model_flux = intrinsic_flux[

        wavelength_mask

    ]


    if model_wavelength.size < 2:

        raise ValueError(

            "The atmosphere wavelength grid does not "
            "cover the requested passbands."

        )


    # ======================================================
    # Apply extinction when requested
    # ======================================================

    if ebv == 0.0:

        extincted_flux = model_flux.copy()


    else:

        # ==================================================
        # Validate the optical CCM89 domain
        # ==================================================

        minimum_optical_wavelength = (

            10000.0

            /

            3.3

        )


        maximum_optical_wavelength = (

            10000.0

            /

            1.1

        )


        if (

            minimum_required_wavelength

            <=

            minimum_optical_wavelength

            or

            maximum_required_wavelength

            >

            maximum_optical_wavelength

        ):

            raise ValueError(

                "All passbands must lie inside the "
                "implemented optical CCM89 domain "
                "when E(B-V) is greater than zero."

            )


        # ==================================================
        # Calculate optical CCM89 extinction
        # ==================================================

        extinction_curve = (

            ccm89_optical_extinction(

                wavelength=model_wavelength,

                ebv=ebv,

                rv=rv

            )

        )


        # ==================================================
        # Apply extinction to the model SED
        # ==================================================

        extincted_flux = apply_extinction(

            flux=model_flux,

            extinction=extinction_curve

        )


    # ======================================================
    # Calculate synthetic magnitudes
    # ======================================================

    magnitudes = {}


    for name, passband in (

        passbands.items()

    ):

        band_flux = synthetic_band_flux(

            wavelength=model_wavelength,

            flux=extincted_flux,

            passband=passband

        )


        magnitudes[

            name

        ] = pogson_magnitude(

            flux=band_flux,

            zero_point=(

                normalized_zero_points[

                    name

                ]

            )

        )


    return magnitudes

def binary_photometric_chi_square(
    parameters,
    atmosphere_grid,
    passbands,
    zero_points,
    measurements,
    primary_logg,
    secondary_logg,
    metallicity,
    distance,
    rv=3.1
):
    """
    Evaluate the photometric chi-square of a binary
    stellar model.

    Parameters
    ----------
    parameters : array_like
        Physical model parameters in the fixed order

            [
                primary_temperature,
                secondary_temperature,
                primary_radius,
                secondary_radius,
                ebv
            ]

        Temperatures are expressed in kelvin, radii
        in solar radii, and E(B-V) in magnitudes.

    atmosphere_grid
        Stellar-atmosphere grid providing an
        ``interpolate()`` method.

    passbands : mapping
        Mapping from passband identifiers to
        photometric-passband objects.

    zero_points : mapping
        Mapping from passband identifiers to
        photometric flux-density zero points.

    measurements : iterable of PhotometricMeasurement
        Observed photometric magnitude measurements.

    primary_logg : float
        Fixed primary surface gravity.

    secondary_logg : float
        Fixed secondary surface gravity.

    metallicity : float
        Fixed atmosphere-grid metallicity.

    distance : float
        Fixed system distance in parsecs.

    rv : float, default=3.1
        Fixed ratio of total-to-selective
        extinction.

    Returns
    -------
    float
        Photometric chi-square statistic,

            chi_square
            =
            sum(
                (
                    model_magnitude
                    -
                    observed_magnitude
                )**2
                /
                uncertainty**2
            ).

    Raises
    ------
    ValueError
        If ``parameters`` does not contain exactly
        five finite values or if any physical
        parameter lies outside its allowed domain.

    Notes
    -----
    The fitted parameter vector is intentionally
    restricted to

        T_1, T_2, R_1, R_2, E(B-V).

    Surface gravities, metallicity, distance, and
    R_V remain fixed configuration values.
    """

    # ======================================================
    # Normalize parameter vector
    # ======================================================

    try:

        parameter_array = np.asarray(

            parameters,

            dtype=float

        )

    except (

        TypeError,

        ValueError

    ) as error:

        raise TypeError(

            "parameters must contain real numbers."

        ) from error


    # ======================================================
    # Validate parameter-vector structure
    # ======================================================

    if (

        parameter_array.ndim != 1

        or

        parameter_array.size != 5

    ):

        raise ValueError(

            "parameters must be a one-dimensional "
            "sequence containing exactly five "
            "values: T1, T2, R1, R2, and E(B-V)."

        )


    if not np.all(

        np.isfinite(
            parameter_array
        )

    ):

        raise ValueError(

            "parameters must contain only finite "
            "values."

        )


    # ======================================================
    # Unpack physical parameters
    # ======================================================

    (

        primary_temperature,

        secondary_temperature,

        primary_radius,

        secondary_radius,

        ebv

    ) = parameter_array


    # ======================================================
    # Validate physical parameter domains
    # ======================================================

    if primary_temperature <= 0.0:

        raise ValueError(

            "primary_temperature must be positive."

        )


    if secondary_temperature <= 0.0:

        raise ValueError(

            "secondary_temperature must be positive."

        )


    if primary_radius <= 0.0:

        raise ValueError(

            "primary_radius must be positive."

        )


    if secondary_radius <= 0.0:

        raise ValueError(

            "secondary_radius must be positive."

        )


    if ebv < 0.0:

        raise ValueError(

            "ebv must not be negative."

        )


    # ======================================================
    # Calculate synthetic model magnitudes
    # ======================================================

    model_magnitudes = (

        synthetic_binary_magnitudes(

            atmosphere_grid=atmosphere_grid,

            passbands=passbands,

            zero_points=zero_points,

            primary_temperature=(

                primary_temperature

            ),

            secondary_temperature=(

                secondary_temperature

            ),

            primary_logg=primary_logg,

            secondary_logg=secondary_logg,

            metallicity=metallicity,

            primary_radius=primary_radius,

            secondary_radius=secondary_radius,

            distance=distance,

            ebv=ebv,

            rv=rv

        )

    )


    # ======================================================
    # Calculate photometric chi-square
    # ======================================================

    return photometric_chi_square(

        model_magnitudes=model_magnitudes,

        measurements=measurements

    )