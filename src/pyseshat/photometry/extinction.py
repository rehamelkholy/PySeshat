import numpy as np


def apply_extinction(
    flux,
    extinction
):
    """
    Apply wavelength-dependent extinction to a flux
    density array.

    Parameters
    ----------
    flux : float or array_like
        Intrinsic flux density.

    extinction : float or array_like
        Extinction in magnitudes.

        A scalar extinction value is applied uniformly
        to all flux values. An extinction array must be
        broadcast-compatible with ``flux``.

    Returns
    -------
    float or numpy.ndarray
        Extinguished flux density calculated as

            f_ext = f_intrinsic * 10**(-0.4 * A)

        A scalar flux input with scalar extinction
        produces a Python float. Array calculations
        produce a NumPy array.

    Raises
    ------
    ValueError
        If flux or extinction contains non-finite
        values, if flux is negative, if extinction is
        negative, or if the input shapes are not
        broadcast-compatible.

    Notes
    -----
    This function applies a supplied extinction value
    or extinction curve. It does not calculate
    extinction from E(B-V), A_V, or R_V.
    """

    # ======================================================
    # Normalize inputs
    # ======================================================

    flux_array = np.asarray(

        flux,

        dtype=float

    )


    extinction_array = np.asarray(

        extinction,

        dtype=float

    )


    # ======================================================
    # Validate flux
    # ======================================================

    if not np.all(

        np.isfinite(
            flux_array
        )

    ):

        raise ValueError(

            "flux must contain only finite values."

        )


    if np.any(

        flux_array < 0.0

    ):

        raise ValueError(

            "flux must not contain negative values."

        )


    # ======================================================
    # Validate extinction
    # ======================================================

    if not np.all(

        np.isfinite(
            extinction_array
        )

    ):

        raise ValueError(

            "extinction must contain only "
            "finite values."

        )


    if np.any(

        extinction_array < 0.0

    ):

        raise ValueError(

            "extinction must not contain "
            "negative values."

        )


    # ======================================================
    # Validate broadcasting
    # ======================================================

    try:

        attenuation = np.power(

            10.0,

            (

                -0.4

                *

                extinction_array

            )

        )


        extinguished_flux = (

            flux_array

            *

            attenuation

        )


    except ValueError as error:

        raise ValueError(

            "flux and extinction must have "
            "broadcast-compatible shapes."

        ) from error


    # ======================================================
    # Validate output
    # ======================================================

    if not np.all(

        np.isfinite(
            extinguished_flux
        )

    ):

        raise ValueError(

            "The calculated extinguished flux "
            "contains non-finite values."

        )


    # ======================================================
    # Preserve scalar return behavior
    # ======================================================

    if extinguished_flux.ndim == 0:

        return float(

            extinguished_flux

        )


    return extinguished_flux

def ccm89_optical_extinction(
    wavelength,
    ebv,
    rv=3.1
):
    """
    Calculate optical extinction using the CCM89 law.

    Parameters
    ----------
    wavelength : float or array_like
        Wavelength in Angstrom.

        The supported optical wavelength range is

            3030.303... < wavelength
            <= 9090.909... Angstrom,

        corresponding to

            1.1 <= x < 3.3 inverse micrometres.

    ebv : float
        Color excess E(B-V) in magnitudes.

    rv : float, default=3.1
        Ratio of total-to-selective extinction,

            R_V = A_V / E(B-V).

    Returns
    -------
    float or numpy.ndarray
        Monochromatic extinction A_lambda in magnitudes.

        A scalar wavelength input produces a Python
        float. An array input produces a NumPy array.

    Raises
    ------
    ValueError
        If wavelength, ebv, or rv is invalid, or if
        wavelength lies outside the supported optical
        CCM89 domain.

    Notes
    -----
    This function implements only the optical CCM89
    branch,

        1.1 <= x < 3.3 inverse micrometres,

    where

        x = 10000 / wavelength

    for wavelength expressed in Angstrom.

    The extinction is calculated from

        A_lambda
        =
        E(B-V) * [R_V * a(x) + b(x)].

    The infrared and ultraviolet CCM89 branches are
    intentionally not implemented.
    """

    # ======================================================
    # Normalize inputs
    # ======================================================

    wavelength_array = np.asarray(

        wavelength,

        dtype=float

    )


    ebv_array = np.asarray(

        ebv,

        dtype=float

    )


    rv_array = np.asarray(

        rv,

        dtype=float

    )


    # ======================================================
    # Validate wavelength
    # ======================================================

    if not np.all(

        np.isfinite(
            wavelength_array
        )

    ):

        raise ValueError(

            "wavelength must contain only "
            "finite values."

        )


    if np.any(

        wavelength_array <= 0.0

    ):

        raise ValueError(

            "wavelength must contain only "
            "positive values."

        )


    # ======================================================
    # Validate E(B-V)
    # ======================================================

    if not np.all(

        np.isfinite(
            ebv_array
        )

    ):

        raise ValueError(

            "ebv must contain only finite values."

        )


    if np.any(

        ebv_array < 0.0

    ):

        raise ValueError(

            "ebv must not contain negative values."

        )


    # ======================================================
    # Validate R_V
    # ======================================================

    if not np.all(

        np.isfinite(
            rv_array
        )

    ):

        raise ValueError(

            "rv must contain only finite values."

        )


    if np.any(

        rv_array <= 0.0

    ):

        raise ValueError(

            "rv must contain only positive values."

        )


    # ======================================================
    # Convert wavelength to inverse micrometres
    # ======================================================

    x = (

        10000.0

        /

        wavelength_array

    )


    # ======================================================
    # Validate optical CCM89 domain
    # ======================================================

    if np.any(

        (

            x < 1.1

        )

        |

        (

            x >= 3.3

        )

    ):

        raise ValueError(

            "wavelength lies outside the supported "
            "optical CCM89 domain: "
            "1.1 <= x < 3.3 inverse micrometres."

        )


    # ======================================================
    # Optical CCM89 polynomial coordinate
    # ======================================================

    y = (

        x

        -

        1.82

    )


    # ======================================================
    # Calculate a(x)
    # ======================================================

    a = (

        1.0

        +

        0.17699

        *

        y

        -

        0.50447

        *

        y**2

        -

        0.02427

        *

        y**3

        +

        0.72085

        *

        y**4

        +

        0.01979

        *

        y**5

        -

        0.77530

        *

        y**6

        +

        0.32999

        *

        y**7

    )


    # ======================================================
    # Calculate b(x)
    # ======================================================

    b = (

        1.41338

        *

        y

        +

        2.28305

        *

        y**2

        +

        1.07233

        *

        y**3

        -

        5.38434

        *

        y**4

        -

        0.62251

        *

        y**5

        +

        5.30260

        *

        y**6

        -

        2.09002

        *

        y**7

    )


    # ======================================================
    # Calculate A_lambda
    # ======================================================

    extinction = (

        ebv_array

        *

        (

            rv_array

            *

            a

            +

            b

        )

    )


    # ======================================================
    # Validate output
    # ======================================================

    if not np.all(

        np.isfinite(
            extinction
        )

    ):

        raise ValueError(

            "The calculated extinction contains "
            "non-finite values."

        )


    # ======================================================
    # Preserve scalar return behavior
    # ======================================================

    if extinction.ndim == 0:

        return float(

            extinction

        )


    return extinction