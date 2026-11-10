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
    composite_binary_sed,
    scale_surface_flux
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


def synthetic_binary_sed(
    atmosphere_grid,
    passbands,
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
    Calculate the extincted synthetic flux-density SED curve
    (wavelength vs. flux) of an unresolved binary, for plotting.

    This is a sibling of ``synthetic_binary_magnitudes()`` above,
    provided to support ``visualization.sed_plot.plot_sed_fit()``
    (see ``docs/guide/visualization.md``'s "SED fit" section). It performs
    EXACTLY the same physical steps as
    ``synthetic_binary_magnitudes()`` -- atmosphere interpolation
    (``atmosphere_grid.interpolate()``), unresolved-binary flux
    combination (``composite_binary_sed()``), wavelength
    restriction to the requested passbands' combined coverage, and
    CCM89 optical extinction (``ccm89_optical_extinction()``/
    ``apply_extinction()``) -- but returns the extincted flux
    CURVES (composite and each component individually) instead of
    collapsing them to per-passband magnitudes via
    ``synthetic_band_flux()``/``pogson_magnitude()``. Deliberately
    implemented as an independent function rather than a shared
    refactor of the validated, tested
    ``synthetic_binary_magnitudes()`` above, to avoid touching
    that function's tested code path; self-consistency between the
    two is verified directly in
    ``tests/test_sed_curve.py`` (integrating this function's
    curves through a passband and confirming the resulting
    magnitude matches ``synthetic_binary_magnitudes()``'s own
    output to numerical precision), not merely assumed from
    sharing the same helper calls.

    Parameters
    ----------
    atmosphere_grid
        Stellar-atmosphere grid providing an ``interpolate()``
        method.

    passbands : mapping
        Mapping from passband identifiers to photometric-passband
        objects. Only used to determine the wavelength interval
        the returned curves must cover (no photometric integration
        is performed here) -- the same restriction
        ``synthetic_binary_magnitudes()`` applies, so the returned
        curves are guaranteed valid over exactly the range a
        companion call to that function would need.

    primary_temperature, secondary_temperature : float
        Component effective temperatures in kelvin.

    primary_logg, secondary_logg : float
        Component surface gravities, log10(g).

    metallicity : float
        Atmosphere-grid metallicity.

    primary_radius, secondary_radius : float
        Component radii in solar radii.

    distance : float
        Distance in parsecs.

    ebv : float, default=0.0
        Color excess E(B-V), in magnitudes.

    rv : float, default=3.1
        Ratio of total-to-selective extinction.

    Returns
    -------
    dict
        ``{"wavelength": ndarray, "composite_flux": ndarray,
        "primary_flux": ndarray, "secondary_flux": ndarray}``, all
        flux arrays in the same units as
        ``AtmosphereModel.flux``/``composite_binary_sed()``
        (erg s^-1 cm^-2 Angstrom^-1), extincted identically to
        ``synthetic_binary_magnitudes()``'s internal model SED
        (``primary_flux + secondary_flux`` matches
        ``composite_flux`` to floating-point precision, since both
        components share the same E(B-V)/R_V, the standard
        assumption for two stars at the same distance behind the
        same foreground reddening).

    Raises
    ------
    ValueError
        Under the same conditions as
        ``synthetic_binary_magnitudes()``: no passbands supplied,
        insufficient wavelength coverage, or (when E(B-V) > 0) a
        passband outside the implemented optical CCM89 domain.

    Notes
    -----
    Individual-component curves are diluted by their own
    ``(radius / distance)^2`` factor (via ``scale_surface_flux()``,
    through ``composite_binary_sed()``'s own primary/secondary
    scaling) and extincted with the SAME wavelength-dependent
    extinction curve as the composite -- physically correct for
    two components of one physical binary at one distance behind
    one foreground dust screen, and the reason
    ``primary_flux + secondary_flux`` matches ``composite_flux`` to
    floating-point precision (extinction is a multiplicative factor
    applied identically to both terms of a sum -- mathematically
    exact, though not necessarily bit-for-bit identical once
    floating-point rounding is involved).
    """

    # ======================================================
    # Validate passbands
    # ======================================================

    if not hasattr(passbands, "items"):

        raise TypeError(
            "passbands must be a mapping."
        )

    if len(passbands) == 0:

        raise ValueError(
            "At least one passband is required."
        )

    # ======================================================
    # Validate color excess
    # ======================================================

    try:

        ebv = float(ebv)

    except (TypeError, ValueError) as error:

        raise TypeError(
            "ebv must be a real number."
        ) from error

    if not np.isfinite(ebv):

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
        metallicity=metallicity,
    )

    secondary = atmosphere_grid.interpolate(
        teff=secondary_temperature,
        logg=secondary_logg,
        metallicity=metallicity,
    )

    # ======================================================
    # Construct diluted (distance-scaled) component SEDs and
    # their sum, exactly as composite_binary_sed() does
    # internally for synthetic_binary_magnitudes() -- but here
    # each component's own diluted flux is kept, not just the
    # sum.
    # ======================================================

    from pyseshat.stellar.sed_model import scale_surface_flux

    primary_observed = scale_surface_flux(
        atmosphere=primary,
        radius=primary_radius,
        distance=distance,
    )

    secondary_observed = scale_surface_flux(
        atmosphere=secondary,
        radius=secondary_radius,
        distance=distance,
    )

    wavelength = primary_observed.wavelength.copy()

    intrinsic_composite_flux = (
        primary_observed.flux + secondary_observed.flux
    )

    # ======================================================
    # Determine required wavelength interval (identical logic
    # to synthetic_binary_magnitudes())
    # ======================================================

    minimum_required_wavelength = min(
        passband.minimum_wavelength for passband in passbands.values()
    )

    maximum_required_wavelength = max(
        passband.maximum_wavelength for passband in passbands.values()
    )

    wavelength_mask = (
        (wavelength >= minimum_required_wavelength)
        & (wavelength <= maximum_required_wavelength)
    )

    model_wavelength = wavelength[wavelength_mask]

    model_primary_flux = primary_observed.flux[wavelength_mask]

    model_secondary_flux = secondary_observed.flux[wavelength_mask]

    model_composite_flux = intrinsic_composite_flux[wavelength_mask]

    if model_wavelength.size < 2:

        raise ValueError(
            "The atmosphere wavelength grid does not "
            "cover the requested passbands."
        )

    # ======================================================
    # Apply extinction when requested (identical logic to
    # synthetic_binary_magnitudes(), applied to each component
    # and the composite alike)
    # ======================================================

    if ebv == 0.0:

        extincted_primary_flux = model_primary_flux.copy()
        extincted_secondary_flux = model_secondary_flux.copy()
        extincted_composite_flux = model_composite_flux.copy()

    else:

        minimum_optical_wavelength = 10000.0 / 3.3
        maximum_optical_wavelength = 10000.0 / 1.1

        if (
            minimum_required_wavelength <= minimum_optical_wavelength
            or maximum_required_wavelength > maximum_optical_wavelength
        ):

            raise ValueError(
                "All passbands must lie inside the "
                "implemented optical CCM89 domain "
                "when E(B-V) is greater than zero."
            )

        extinction_curve = ccm89_optical_extinction(
            wavelength=model_wavelength,
            ebv=ebv,
            rv=rv,
        )

        extincted_primary_flux = apply_extinction(
            flux=model_primary_flux,
            extinction=extinction_curve,
        )

        extincted_secondary_flux = apply_extinction(
            flux=model_secondary_flux,
            extinction=extinction_curve,
        )

        extincted_composite_flux = apply_extinction(
            flux=model_composite_flux,
            extinction=extinction_curve,
        )

    return {
        "wavelength": model_wavelength,
        "composite_flux": extincted_composite_flux,
        "primary_flux": extincted_primary_flux,
        "secondary_flux": extincted_secondary_flux,
    }


def synthetic_magnitude_differences(
    atmosphere_grid,
    passbands,
    primary_temperature,
    secondary_temperature,
    primary_logg,
    secondary_logg,
    metallicity,
    primary_radius,
    secondary_radius,
    distance
):
    """
    Synthetic magnitude difference m_secondary - m_primary in
    each passband, -2.5 log10(F_2 / F_1) of the band fluxes.

    Zero points cancel. Interstellar extinction is not applied:
    it dims both components equally, so its effect on the
    difference is only the second-order change of the effective
    wavelength between two stars of different temperature.

    Returns
    -------
    dict
        Mapping from passband identifiers to magnitude differences.
    """
    primary = scale_surface_flux(
        atmosphere_grid.interpolate(teff=primary_temperature, logg=primary_logg,
                                    metallicity=metallicity),
        primary_radius, distance)
    secondary = scale_surface_flux(
        atmosphere_grid.interpolate(teff=secondary_temperature, logg=secondary_logg,
                                    metallicity=metallicity),
        secondary_radius, distance)
    out = {}
    for name, passband in passbands.items():
        f1 = synthetic_band_flux(wavelength=primary.wavelength, flux=primary.flux,
                                 passband=passband)
        f2 = synthetic_band_flux(wavelength=secondary.wavelength, flux=secondary.flux,
                                 passband=passband)
        if not (f1 > 0.0 and f2 > 0.0):
            raise ValueError(f"Non-positive component band flux in passband {name}.")
        out[name] = -2.5 * np.log10(f2 / f1)
    return out
