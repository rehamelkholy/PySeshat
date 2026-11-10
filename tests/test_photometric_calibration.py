"""Tests for pyseshat.photometry.calibration (standard-system zero points)."""

import math
from pathlib import Path

import numpy as np
import pytest

from pyseshat.photometry.calibration import (
    VEGA_SPECTRUM_FILE,
    VEGA_STANDARD_MAGNITUDES,
    load_vega_spectrum,
    standard_system_zero_point,
    standard_system_zero_points,
    vega_zero_points,
)
from pyseshat.photometry.filters import load_passband
from pyseshat.photometry.synthetic import pogson_magnitude, synthetic_band_flux


PASSBAND_DIR = Path(__file__).resolve().parents[1] / "validation" / "photometry" / "passbands"


def test_vega_gets_its_standard_magnitude():
    """Vega's own flux, measured against the converted zero point,
    gives Vega's standard-system magnitude."""

    vega_flux = 3.2441e-9

    zero_point = standard_system_zero_point(vega_flux, 1.432)

    assert pogson_magnitude(vega_flux, zero_point) == pytest.approx(1.432, abs=1e-12)


def test_zero_vega_magnitude_leaves_zero_point_unchanged():

    assert standard_system_zero_point(3.5833e-9, 0.0) == 3.5833e-9


def test_vega_magnitudes_are_maiz_apellaniz_2007_table_4():
    """Preferred values of Maiz Apellaniz (2007), ASP Conf. Ser., 364,
    227, Table 4."""

    assert VEGA_STANDARD_MAGNITUDES == {
        "Johnson U": 0.055, "Johnson B": 0.034, "Johnson V": 0.026,
        "Cousins R": 0.030,
        "Stromgren u": 1.432, "Stromgren v": 0.179,
        "Stromgren b": 0.018, "Stromgren y": 0.014,
        "Tycho BT": 0.069, "Tycho VT": 0.032,
    }


def test_vega_stromgren_indices_match_hauck_mermilliod():
    """The Stromgren values reproduce Vega's catalogue indices
    b-y = 0.004, m1 = 0.157, c1 = 1.088 (Hauck & Mermilliod 1998, as
    adopted by Bessell 2011, PASP, 123, 1442) to within 0.005 mag."""

    u, v, b, y = (
        VEGA_STANDARD_MAGNITUDES[f"Stromgren {band}"] for band in "uvby"
    )

    assert b - y == pytest.approx(0.004, abs=0.005)
    assert (v - b) - (b - y) == pytest.approx(0.157, abs=0.005)
    assert (u - v) - (v - b) == pytest.approx(1.088, abs=0.005)


def test_mapping_conversion_and_missing_band():

    converted = standard_system_zero_points({"Stromgren v": 1.0})

    assert converted["Stromgren v"] == pytest.approx(10 ** (0.4 * 0.179))

    with pytest.raises(KeyError, match="Unknown band"):

        standard_system_zero_points({"Unknown band": 1.0})


@pytest.mark.parametrize(
    "zero_point, magnitude",
    [(0.0, 0.0), (-1.0, 0.0), (math.nan, 0.0), (math.inf, 0.0), (1.0, math.nan)],
)
def test_invalid_input_raises(zero_point, magnitude):

    with pytest.raises(ValueError):

        standard_system_zero_point(zero_point, magnitude)


def test_shipped_vega_spectrum_is_calspec_stis_003():

    assert VEGA_SPECTRUM_FILE.name == "alpha_lyr_stis_003.fits"

    wavelength, flux = load_vega_spectrum()

    assert np.all(np.diff(wavelength) > 0)
    assert np.all(flux > 0)

    # Vega's flux density near 5556 A is about 3.4-3.5e-9 erg/s/cm2/A
    # (Bohlin 2007 scale).
    f5556 = np.interp(5556.0, wavelength, flux)
    assert 3.3e-9 < f5556 < 3.6e-9


def test_vega_zero_points_give_vega_its_standard_magnitudes():
    """Vega, observed through the same passbands, gets exactly
    VEGA_STANDARD_MAGNITUDES."""

    files = {
        "Johnson U": "johnson_u.dat", "Johnson B": "johnson_b.dat",
        "Johnson V": "johnson_v.dat", "Cousins R": "cousins_r.dat",
        "Stromgren u": "stromgren_u.dat", "Stromgren v": "stromgren_v.dat",
        "Stromgren b": "stromgren_b.dat", "Stromgren y": "stromgren_y.dat",
        "Tycho BT": "tycho_bt.dat", "Tycho VT": "tycho_vt.dat",
    }

    passbands = {
        name: load_passband(PASSBAND_DIR / fname, name=name)
        for name, fname in files.items()
    }

    zero_points = vega_zero_points(passbands)

    wavelength, flux = load_vega_spectrum()

    for name, passband in passbands.items():

        magnitude = pogson_magnitude(
            synthetic_band_flux(wavelength, flux, passband),
            zero_points[name],
        )

        assert magnitude == pytest.approx(
            VEGA_STANDARD_MAGNITUDES[name], abs=1e-10
        )


def test_vega_zero_points_requires_both_arrays():

    with pytest.raises(ValueError):

        vega_zero_points({}, vega_wavelength=np.array([1.0, 2.0]))
