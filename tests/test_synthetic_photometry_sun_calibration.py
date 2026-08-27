"""
Independent, literature-anchored sanity check for the whole
synthetic-photometry chain used by the Al-Wardat SED-fitting
pipeline: CK04 atmosphere interpolation -> (R/d)^2 geometric
dilution -> passband integration -> Vega zero point -> Pogson
magnitude.

Background
----------
While validating the HD214222 Al-Wardat benchmark (2026-08-27), the
model was found to predict entire-system magnitudes systematically
~0.25-0.5 mag brighter than the literature (Abushattal et al. 2025,
arXiv:2512.23652) Table 4 values, in every Johnson/Tycho band. Before
accepting that as a real astrophysical/methodological residual, this
test was written to independently check the pipeline's absolute flux
calibration against a completely different, well-established
reference: placing a Sun-like model (Teff=5772 K, logg=4.4377,
R=1 R_sun, solar metallicity) at 10 pc must reproduce the Sun's
well-known absolute Johnson V magnitude, M_V = 4.83 (a standard,
widely-cited literature value -- see e.g. Willmer 2018, ApJS 236, 47).

The pipeline reproduces M_V(Sun @ 10 pc) to within 0.04 mag, and
Johnson U/B/R and Tycho BT/VT all agree with their standard literature
zero-magnitude-flux-derived absolute magnitudes to within ~0.15 mag
(see docs/project_decisions.md for the full per-band table). This
rules out a code-level flux-scaling bug as the source of the
HD214222 residual and points instead to a genuine methodological
difference between this pipeline's (SVO Filter Profile Service,
Vega-system) zero-point convention and whatever internal calibration
Al-Wardat's own SED-fitting method uses (not specified in the paper).

One exception was found and is deliberately NOT covered by this test:
the SVO "Generic/Stromgren.u" zero point disagreed with both this Sun
check and the real HD214222 fit by ~1-1.7 mag, consistent with the
well-documented historical quirk that the classical Stromgren uvbybeta
system's u-band zero point (Crawford & Barnes 1970) is calibrated
against specific standard stars rather than a simple Vega-spectrum
convolution. Stromgren u is excluded from benchmark fitting for this
reason (see validation/expected_results/*.json "excluded_measurements").
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
from pyseshat.stellar.sed_model import scale_surface_flux


ATMOSPHERE_DIR = ROOT / "validation" / "atmosphere_models" / "ckp00"
PASSBAND_DIR = ROOT / "validation" / "photometry" / "passbands"

# Literature Sun absolute magnitudes (Vega system). Sources:
# Johnson V: Willmer (2018), ApJS 236, 47 (M_V,Sun = 4.83, the
# figure most commonly quoted in the stellar-astrophysics
# literature). Johnson U/B/R and Tycho BT/VT: derived the same way
# (synthetic photometry of the Sun) by multiple standard references
# and consistent with M_V=4.83 given the Sun's well-known standard
# colors (B-V=0.65, U-B=0.20, V-R=0.35 approx); tolerances below are
# deliberately loose (0.2 mag) because this is a coarse cross-check,
# not a precision photometric calibration.
LITERATURE_ABSOLUTE_MAGNITUDE = {
    "Johnson U": 5.61,
    "Johnson B": 5.44,
    "Johnson V": 4.83,
    "Johnson R": 4.42,
    "Tycho BT": 5.50,
    "Tycho VT": 4.86,
}

TOLERANCE_MAG = {
    # V is the best-established reference value and the pipeline's
    # own zero point is an exact SVO match for this passband curve,
    # so this one gets a tight tolerance.
    "Johnson V": 0.05,
}
DEFAULT_TOLERANCE_MAG = 0.2

ZERO_POINTS = {
    "Johnson U": 3.49719e-9,
    "Johnson B": 6.72553e-9,
    "Johnson V": 3.5833e-9,
    "Johnson R": 1.87529e-9,
    "Tycho BT": 6.50910e-9,
    "Tycho VT": 3.99504e-9,
}

PASSBAND_FILES = {
    "Johnson U": "johnson_u.dat",
    "Johnson B": "johnson_b.dat",
    "Johnson V": "johnson_v.dat",
    "Johnson R": "johnson_r.dat",
    "Tycho BT": "tycho_bt.dat",
    "Tycho VT": "tycho_vt.dat",
}


def _grid_files_available():
    return (ATMOSPHERE_DIR / "ckp00_5750.fits").is_file() and (
        ATMOSPHERE_DIR / "ckp00_6000.fits"
    ).is_file()


@pytest.mark.skipif(
    not _grid_files_available(),
    reason="requires the local CK04 solar-metallicity 5750K/6000K FITS files",
)
@pytest.mark.parametrize("passband_name", sorted(LITERATURE_ABSOLUTE_MAGNITUDE))
def test_sun_at_10pc_matches_literature_absolute_magnitude(passband_name):
    """
    A Sun-like (Teff=5772K, logg=4.4377, R=1 Rsun) model placed at
    10 pc should reproduce the Sun's known absolute magnitude in
    every band this pipeline currently supports zero points for,
    validating atmosphere interpolation, (R/d)^2 scaling, passband
    integration, and Vega zero points end to end.
    """

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

    magnitude = pogson_magnitude(band_flux, ZERO_POINTS[passband_name])

    expected = LITERATURE_ABSOLUTE_MAGNITUDE[passband_name]

    tolerance = TOLERANCE_MAG.get(passband_name, DEFAULT_TOLERANCE_MAG)

    assert magnitude == pytest.approx(expected, abs=tolerance), (
        f"{passband_name}: predicted M={magnitude:.3f} vs literature "
        f"M={expected:.3f} (tolerance {tolerance} mag)"
    )
