"""
Independent, literature-anchored sanity check for the whole
synthetic-photometry chain used by the binary SED-fitting
pipeline: CK04 atmosphere interpolation -> (R/d)^2 geometric
dilution -> passband integration -> standard-system zero point ->
Pogson magnitude.

A Sun-like model (Teff = 5772 K, log g = 4.4377, R = 1 R_sun, solar
metallicity) placed at 10 pc must reproduce

* the Sun's absolute magnitudes in Johnson U, B, V and Tycho BT, VT
  (Vega system) of Willmer (2018), ApJS, 236, 47,
  doi:10.3847/1538-4365/aabfdf, Table 3: 5.61, 5.44, 4.81, 5.58,
  4.88 mag. Cousins R is not tested here.
* the Sun's Stromgren indices of Melendez et al. (2010), A&A, 522,
  A98, doi:10.1051/0004-6361/201014741: (b-y) = 0.4105, m1 = 0.2122,
  c1 = 0.3319.

Zero points follow Maiz Apellaniz (2006, 2007): the CALSPEC Vega
spectrum shipped with PySeshat, integrated through the same passband
files, with Vega's standard-system magnitudes
(pyseshat.photometry.calibration.vega_zero_points; Vega u = 1.432).
The Stromgren-index test fails by more than 1 mag in c1 with Vega = 0
zero points, which is the error found and fixed in 0.2.0 (it had been
mistaken for a band-specific pipeline difference, and Stromgren u had
been excluded from the SED benchmarks because of it).

Tolerances are coarse (0.2 mag for magnitudes other than V, 0.05 mag
for V and b-y, 0.08 mag for m1 and c1; see STROMGREN_INDEX_TOLERANCE)
because the CK04 solar model is not a perfect match to the Sun; this is
a calibration cross-check, not a precision photometric calibration. The
tests with the observed solar spectrum (below) are tighter.

Passbands are the photon-counting curves of Maiz Apellaniz (2006),
Tables 1-3 (``# detector: photon`` in each file), integrated with the
lambda weighting of his Eq. (A1). With these curves the observed solar
spectrum gives U-B = 0.15 (Willmer: 0.17); the previous curves, which
were also integrated in energy form, gave 0.06.
"""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pyseshat.stellar.atmospheres import load_ck04_grid
from pyseshat.photometry.filters import load_passband
from pyseshat.photometry.synthetic import (
    synthetic_band_flux,
    pogson_magnitude,
)
from pyseshat.photometry.calibration import (
    vega_zero_points,
)
from pyseshat.stellar.sed_model import scale_surface_flux


ATMOSPHERE_DIR = ROOT / "validation" / "atmosphere_models" / "ckp00"
PASSBAND_DIR = ROOT / "validation" / "photometry" / "passbands"

# Willmer (2018), Table 3, Vega system.
LITERATURE_ABSOLUTE_MAGNITUDE = {
    "Johnson U": 5.61,
    "Johnson B": 5.44,
    "Johnson V": 4.81,
    "Tycho BT": 5.58,
    "Tycho VT": 4.88,
}

TOLERANCE_MAG = {
    "Johnson V": 0.05,
}
DEFAULT_TOLERANCE_MAG = 0.2

# Melendez et al. (2010), solar colours.
LITERATURE_SOLAR_STROMGREN = {
    "b-y": 0.4105,
    "m1": 0.2122,
    "c1": 0.3319,
}
# Per-index tolerances for the CK04 solar MODEL. b-y is held to 0.05.
# m1 and c1 are held to 0.08: with the Maiz Apellaniz (2006) curves the
# CK04 model gives m1 = 0.256 and c1 = 0.271, i.e. +0.065 and -0.036 mag
# from the OBSERVED CALSPEC solar spectrum through the same curves
# (0.191, 0.307), and +0.043 and -0.061 from Melendez et al. (2010). The
# difference is in the model atmosphere's violet/UV flux, not in the
# calibration, which test_observed_sun_stromgren_indices_match_literature
# checks at 0.04 mag.
STROMGREN_INDEX_TOLERANCE = {"b-y": 0.05, "m1": 0.08, "c1": 0.08}


PASSBAND_FILES = {
    "Johnson U": "johnson_u.dat",
    "Johnson B": "johnson_b.dat",
    "Johnson V": "johnson_v.dat",
    "Tycho BT": "tycho_bt.dat",
    "Tycho VT": "tycho_vt.dat",
    "Stromgren u": "stromgren_u.dat",
    "Stromgren v": "stromgren_v.dat",
    "Stromgren b": "stromgren_b.dat",
    "Stromgren y": "stromgren_y.dat",
}


ZERO_POINTS = vega_zero_points({
    name: load_passband(PASSBAND_DIR / fname, name=name)
    for name, fname in PASSBAND_FILES.items()
})


def _grid_files_available():
    return (ATMOSPHERE_DIR / "ckp00_5750.fits").is_file() and (
        ATMOSPHERE_DIR / "ckp00_6000.fits"
    ).is_file()


requires_grid = pytest.mark.skipif(
    not _grid_files_available(),
    reason="requires the local CK04 solar-metallicity 5750K/6000K FITS files",
)


def _sun_magnitude(passband_name):

    grid = load_ck04_grid(
        [
            ATMOSPHERE_DIR / "ckp00_5750.fits",
            ATMOSPHERE_DIR / "ckp00_6000.fits",
        ]
    )

    sun_atmosphere = grid.interpolate(teff=5772.0, logg=4.4377, metallicity=0.0)

    observed = scale_surface_flux(sun_atmosphere, radius=1.0, distance=10.0)

    passband = load_passband(
        PASSBAND_DIR / PASSBAND_FILES[passband_name],
        name=passband_name,
    )

    band_flux = synthetic_band_flux(observed.wavelength, observed.flux, passband)

    return pogson_magnitude(band_flux, ZERO_POINTS[passband_name])


@requires_grid
@pytest.mark.parametrize("passband_name", sorted(LITERATURE_ABSOLUTE_MAGNITUDE))
def test_sun_at_10pc_matches_literature_absolute_magnitude(passband_name):
    """
    A Sun-like model at 10 pc reproduces the Sun's absolute
    magnitude (Willmer 2018) in each Johnson/Tycho band.
    """

    magnitude = _sun_magnitude(passband_name)

    expected = LITERATURE_ABSOLUTE_MAGNITUDE[passband_name]

    tolerance = TOLERANCE_MAG.get(passband_name, DEFAULT_TOLERANCE_MAG)

    assert magnitude == pytest.approx(expected, abs=tolerance), (
        f"{passband_name}: predicted M={magnitude:.3f} vs literature "
        f"M={expected:.3f} (tolerance {tolerance} mag)"
    )


@requires_grid
def test_sun_stromgren_indices_match_literature():
    """
    A Sun-like model reproduces the solar Stromgren indices
    (b-y), m1 and c1 of Melendez et al. (2010) on the standard
    uvby system.
    """

    u, v, b, y = (
        _sun_magnitude(f"Stromgren {band}") for band in "uvby"
    )

    indices = {
        "b-y": b - y,
        "m1": (v - b) - (b - y),
        "c1": (u - v) - (v - b),
    }

    for name, expected in LITERATURE_SOLAR_STROMGREN.items():

        assert indices[name] == pytest.approx(
            expected, abs=STROMGREN_INDEX_TOLERANCE[name]
        ), (
            f"{name}: predicted {indices[name]:.4f} vs literature "
            f"{expected:.4f} (tolerance {STROMGREN_INDEX_TOLERANCE[name]})"
        )


# ----------------------------------------------------------------------
# Observed solar spectrum (CALSPEC sun_reference_stis_001, flux at 1 AU):
# tests the passband/zero-point chain without any model atmosphere.
# Tolerances: 0.02 mag for V, 0.1 mag for the other bands (Willmer 2018
# used different passband curves), 0.04 mag for the Stromgren indices.
# ----------------------------------------------------------------------

CALSPEC_SUN_FILE = ROOT / "tests" / "data" / "sun_reference_stis_001.fits"

AU_CM = 1.495978707e13
TEN_PC_CM = 10.0 * 3.0856775814913673e18


def _calspec_sun_magnitude(passband_name):

    import warnings

    import numpy as np
    from astropy.io import fits
    from astropy.io.fits.verify import VerifyWarning

    # The CALSPEC header has a non-integer TNULL keyword; harmless.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", VerifyWarning)
        data = fits.getdata(CALSPEC_SUN_FILE, 1)

    wavelength = np.asarray(data["WAVELENGTH"], dtype=float)

    flux = np.asarray(data["FLUX"], dtype=float) * (AU_CM / TEN_PC_CM) ** 2

    passband = load_passband(
        PASSBAND_DIR / PASSBAND_FILES[passband_name],
        name=passband_name,
    )

    return pogson_magnitude(
        synthetic_band_flux(wavelength, flux, passband),
        ZERO_POINTS[passband_name],
    )


@pytest.mark.parametrize("passband_name", sorted(LITERATURE_ABSOLUTE_MAGNITUDE))
def test_observed_sun_matches_literature_absolute_magnitude(passband_name):

    tolerance = 0.02 if passband_name == "Johnson V" else 0.1

    assert _calspec_sun_magnitude(passband_name) == pytest.approx(
        LITERATURE_ABSOLUTE_MAGNITUDE[passband_name], abs=tolerance
    )


def test_observed_sun_stromgren_indices_match_literature():

    u, v, b, y = (
        _calspec_sun_magnitude(f"Stromgren {band}") for band in "uvby"
    )

    indices = {
        "b-y": b - y,
        "m1": (v - b) - (b - y),
        "c1": (u - v) - (v - b),
    }

    for name, expected in LITERATURE_SOLAR_STROMGREN.items():

        assert indices[name] == pytest.approx(expected, abs=0.04), name
