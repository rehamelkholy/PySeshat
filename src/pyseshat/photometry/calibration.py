"""
Vega-based calibration of synthetic magnitudes onto standard systems.

PySeshat follows the synthetic-magnitude convention of Maíz Apellániz
(2006, AJ, 131, 1184; 2007, ASP Conf. Ser., 364, 227):

    m_P = -2.5 log10( F_P / F_P,Vega ) + ZP_P,

where F_P and F_P,Vega are the target's and Vega's mean flux densities
through the same passband P, and ZP_P is Vega's magnitude in the
standard system of that passband (non-zero in general: Vega has
Strömgren u = 1.432). With ``pogson_magnitude(flux, zero_point)`` =
-2.5 log10(flux / zero_point), this corresponds to the zero point

    zero_point = F_P,Vega * 10**(0.4 * ZP_P).

Changed in 0.2.0: earlier versions used SVO Filter Profile Service
"Vega system" zero points directly, i.e. ZP_P = 0 in every band and
Vega fluxes integrated through SVO's passband curves rather than
PySeshat's own. Strömgren u and v were therefore about 1.43 and
0.18 mag too bright relative to the standard system, and Johnson U and
B were offset by about +0.07 and -0.07 mag because PySeshat's U and B
passband files differ from the curves the SVO values belong to.
"""

import math
from pathlib import Path

import numpy as np

from pyseshat.photometry.synthetic import synthetic_band_flux


# Vega's magnitude ZP_P in the standard systems, from Table 4 of
# Maíz Apellániz (2007), "A Uniform Set of Optical/NIR Photometric
# Zero Points to be Used with CHORIZOS", ASP Conf. Ser., 364, 227
# (arXiv:astro-ph/0609430), preferred values; also listed in Appendix
# B of the CHORIZOS handbook v2.1.4. Each value belongs to a specific
# sensitivity curve (Maiz Apellaniz 2007, Table 4, column 3): Maiz
# Apellaniz (2006) Tables 1-3 for Johnson UBV, Stromgren uvby and Tycho,
# and Bessell (1983) for Cousins R (see validation/photometry/passbands/).
# Changed in 0.2.0: the R band is "Cousins R" (previously "Johnson R",
# which used the Cousins R value with a curve of unknown origin).
# These values are defined for the Vega spectrum of Bohlin (2007),
# CALSPEC alpha_lyr_stis_003 (see VEGA_SPECTRUM_FILE).
VEGA_STANDARD_MAGNITUDES = {
    "Johnson U": 0.055,
    "Johnson B": 0.034,
    "Johnson V": 0.026,
    "Cousins R": 0.030,
    "Stromgren u": 1.432,
    "Stromgren v": 0.179,
    "Stromgren b": 0.018,
    "Stromgren y": 0.014,
    "Tycho BT": 0.069,
    "Tycho VT": 0.032,
}


# CALSPEC Vega spectrum alpha_lyr_stis_003 (Bohlin 2007, ASP Conf.
# Ser., 364, 315), shipped with the package (see data/README.md).
# Columns WAVELENGTH (Angstrom, vacuum) and FLUX
# (erg s^-1 cm^-2 Angstrom^-1).
VEGA_SPECTRUM_FILE = (
    Path(__file__).resolve().parent / "data" / "alpha_lyr_stis_003.fits"
)


def load_vega_spectrum(
    path=None
):
    """
    Read a CALSPEC Vega spectrum.

    Parameters
    ----------
    path : str or pathlib.Path, optional
        CALSPEC FITS file with WAVELENGTH (Angstrom) and FLUX
        (erg s^-1 cm^-2 Angstrom^-1) columns in extension 1.
        Defaults to ``VEGA_SPECTRUM_FILE``.

    Returns
    -------
    tuple of numpy.ndarray
        Wavelength and flux density arrays.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    """

    from astropy.io import fits

    path = Path(VEGA_SPECTRUM_FILE if path is None else path)

    if not path.is_file():

        raise FileNotFoundError(
            f"Vega spectrum not found: {path}"
        )

    with fits.open(path) as hdul:

        data = hdul[1].data

        wavelength = np.asarray(data["WAVELENGTH"], dtype=float)

        flux = np.asarray(data["FLUX"], dtype=float)

    return wavelength, flux


def standard_system_zero_point(
    vega_zero_point,
    vega_magnitude
):
    """
    Convert Vega's own mean flux density in a passband (a "Vega = 0"
    zero point) into the zero point of a standard system on which
    Vega has magnitude ``vega_magnitude``.

    Parameters
    ----------
    vega_zero_point : float
        Vega's mean flux density through the passband.

    vega_magnitude : float
        Vega's magnitude in the standard system
        (see ``VEGA_STANDARD_MAGNITUDES``).

    Returns
    -------
    float
        ``vega_zero_point * 10**(0.4 * vega_magnitude)``, in the
        units of ``vega_zero_point``.

    Raises
    ------
    ValueError
        If ``vega_zero_point`` is not finite and positive, or
        ``vega_magnitude`` is not finite.
    """

    zero_point = float(vega_zero_point)

    magnitude = float(vega_magnitude)

    if not math.isfinite(zero_point) or zero_point <= 0.0:

        raise ValueError(
            "vega_zero_point must be finite and positive."
        )

    if not math.isfinite(magnitude):

        raise ValueError(
            "vega_magnitude must be finite."
        )

    return zero_point * 10.0 ** (0.4 * magnitude)


def standard_system_zero_points(
    vega_zero_points,
    vega_magnitudes=None
):
    """
    Apply ``standard_system_zero_point`` to a mapping of passband
    names to Vega mean flux densities.

    Parameters
    ----------
    vega_zero_points : mapping
        Passband name -> Vega's mean flux density (Vega = 0 mag).

    vega_magnitudes : mapping, optional
        Passband name -> Vega's standard-system magnitude.
        Defaults to ``VEGA_STANDARD_MAGNITUDES``.

    Returns
    -------
    dict
        Passband name -> standard-system zero point.

    Raises
    ------
    KeyError
        If a passband has no entry in ``vega_magnitudes``.
    """

    if vega_magnitudes is None:

        vega_magnitudes = VEGA_STANDARD_MAGNITUDES

    missing = [
        name for name in vega_zero_points
        if name not in vega_magnitudes
    ]

    if missing:

        raise KeyError(
            "No standard-system Vega magnitude for passband(s): "
            + ", ".join(sorted(missing))
        )

    return {
        name: standard_system_zero_point(
            zero_point,
            vega_magnitudes[name]
        )
        for name, zero_point in vega_zero_points.items()
    }


def vega_zero_points(
    passbands,
    vega_wavelength=None,
    vega_flux=None,
    vega_magnitudes=None
):
    """
    Standard-system zero points for a set of passbands, computed by
    integrating a Vega spectrum through the same passbands that are
    used for the target (Maíz Apellániz 2006, 2007).

    Parameters
    ----------
    passbands : mapping
        Passband name -> passband object accepted by
        ``synthetic_band_flux``.

    vega_wavelength, vega_flux : array_like, optional
        Vega spectrum (Angstrom; erg s^-1 cm^-2 Angstrom^-1).
        Defaults to ``load_vega_spectrum()``.

    vega_magnitudes : mapping, optional
        Passband name -> Vega's standard-system magnitude.
        Defaults to ``VEGA_STANDARD_MAGNITUDES``.

    Returns
    -------
    dict
        Passband name -> zero point (erg s^-1 cm^-2 Angstrom^-1),
        for use as ``zero_points`` in the photometric model.

    Raises
    ------
    ValueError
        If only one of ``vega_wavelength`` and ``vega_flux`` is given.
    KeyError
        If a passband has no entry in ``vega_magnitudes``.
    """

    if (vega_wavelength is None) != (vega_flux is None):

        raise ValueError(
            "Give both vega_wavelength and vega_flux, or neither."
        )

    if vega_wavelength is None:

        vega_wavelength, vega_flux = load_vega_spectrum()

    vega_band_fluxes = {
        name: synthetic_band_flux(vega_wavelength, vega_flux, passband)
        for name, passband in passbands.items()
    }

    return standard_system_zero_points(
        vega_band_fluxes,
        vega_magnitudes
    )
