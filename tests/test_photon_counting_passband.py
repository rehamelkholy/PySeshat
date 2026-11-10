"""
Photon-counting passbands (added in 0.2.0).

A photon-counting sensitivity curve P(lambda) is integrated with the
lambda weighting of Maiz Apellaniz (2006), AJ, 131, 1184, Eq. (A1),
which is the same as using S = P * lambda in the energy form. The
detector type is set explicitly or by a ``# detector:`` header line.
"""

from pathlib import Path

import numpy as np
import pytest

from pyseshat.photometry.filters import PhotometricPassband, load_passband
from pyseshat.photometry.synthetic import synthetic_band_flux


PASSBAND_DIR = (
    Path(__file__).resolve().parents[1] / "validation" / "photometry" / "passbands"
)


def _write(tmp_path, header):

    path = tmp_path / "band.dat"

    path.write_text(header + "4000 0.0\n4500 1.0\n5000 0.5\n5500 0.0\n")

    return path


@pytest.mark.parametrize(
    "header, expected",
    [
        ("# detector: photon\n", True),
        ("# Detector: Energy\n", False),
        ("# some comment\n", False),
        ("", False),
    ],
)
def test_detector_header(tmp_path, header, expected):

    assert load_passband(_write(tmp_path, header)).photon_counting is expected


def test_argument_overrides_header(tmp_path):

    path = _write(tmp_path, "# detector: photon\n")

    assert load_passband(path, photon_counting=False).photon_counting is False


@pytest.mark.parametrize(
    "header",
    ["# detector: photons\n", "# detector: photon\n# detector: energy\n"],
)
def test_invalid_detector_header(tmp_path, header):

    with pytest.raises(ValueError, match="detector"):

        load_passband(_write(tmp_path, header))


def test_photon_counting_must_be_bool():

    with pytest.raises(TypeError):

        PhotometricPassband(
            wavelength=[4000.0, 5000.0],
            throughput=[1.0, 1.0],
            photon_counting="yes",
        )


def test_photon_counting_equals_lambda_weighted_energy_curve():
    """Eq. (A1): a photon curve P gives the same band flux as the
    energy curve P * lambda."""

    wavelength = np.linspace(3000.0, 8000.0, 2001)

    flux = 1.0e-12 * (wavelength / 5000.0) ** -3

    curve_w = np.array([4000.0, 4500.0, 5000.0, 5500.0, 6000.0])

    curve_p = np.array([0.0, 0.6, 1.0, 0.4, 0.0])

    photon = PhotometricPassband(curve_w, curve_p, photon_counting=True)

    # Interpolating P*lambda linearly differs from (linear P)*lambda,
    # so compare on the SED grid itself.
    grid = wavelength[(wavelength >= 4000.0) & (wavelength <= 6000.0)]

    energy = PhotometricPassband(
        grid, np.interp(grid, curve_w, curve_p) * grid
    )

    ratio = (
        synthetic_band_flux(wavelength, flux, photon)
        / synthetic_band_flux(wavelength, flux, energy)
    )

    assert ratio == pytest.approx(1.0, abs=1e-12)

    # ... and differs from treating P as an energy curve for a
    # sloped spectrum.
    as_energy = PhotometricPassband(curve_w, curve_p)

    ratio_as_energy = (
        synthetic_band_flux(wavelength, flux, photon)
        / synthetic_band_flux(wavelength, flux, as_energy)
    )

    assert abs(ratio_as_energy - 1.0) > 1e-3


@pytest.mark.parametrize(
    "filename, photon",
    [
        ("johnson_u.dat", True), ("johnson_b.dat", True),
        ("johnson_v.dat", True), ("stromgren_u.dat", True),
        ("stromgren_v.dat", True), ("stromgren_b.dat", True),
        ("stromgren_y.dat", True), ("tycho_bt.dat", True),
        ("tycho_vt.dat", True), ("cousins_r.dat", False),
    ],
)
def test_shipped_passband_detector_types(filename, photon):
    """Maiz Apellaniz (2006) Tables 1-3 are photon-counting; the
    Bessell (1983) Cousins R curve is energy-integrating (Maiz
    Apellaniz 2007, Sect. 6)."""

    assert load_passband(PASSBAND_DIR / filename).photon_counting is photon


def test_cousins_r_is_bessell_1983_table_aii():

    band = load_passband(PASSBAND_DIR / "cousins_r.dat")

    assert len(band.wavelength) == 64

    assert band.minimum_wavelength == 5500.0

    assert band.maximum_wavelength == 8650.0

    assert np.allclose(np.diff(band.wavelength), 50.0)

    assert band.throughput.max() == 1.0

    assert band.throughput[-1] == pytest.approx(0.004)


def test_johnson_u_is_maiz_2006_table_3():

    band = load_passband(PASSBAND_DIR / "johnson_u.dat")

    assert len(band.wavelength) == 23

    assert (band.minimum_wavelength, band.maximum_wavelength) == (3050.0, 4150.0)

    assert band.throughput_at(np.array([3100.0]))[0] == pytest.approx(0.237)

    assert band.peak_wavelength == 3750.0
