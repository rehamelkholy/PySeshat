"""Component magnitude-difference constraint for unresolved-binary SED fits."""
from pathlib import Path

import numpy as np
import pytest

from pyseshat.photometry.filters import load_passband
from pyseshat.photometry.model import synthetic_magnitude_differences
from pyseshat.photometry.observations import MagnitudeDifference
from pyseshat.stellar.atmospheres import load_ck04_grid

ROOT = Path(__file__).resolve().parents[1]
GRID = load_ck04_grid([ROOT / "validation" / "atmosphere_models" / "ckp00" / f"ckp00_{t}.fits"
                       for t in (5000, 5250, 5500, 5750, 6000)])
V_BAND = {"Johnson V": load_passband(
    filename=ROOT / "validation" / "photometry" / "passbands" / "johnson_v.dat",
    name="Johnson V")}


def _dm(t1, t2, r1, r2):
    return synthetic_magnitude_differences(
        atmosphere_grid=GRID, passbands=V_BAND, primary_temperature=t1,
        secondary_temperature=t2, primary_logg=4.5, secondary_logg=4.5, metallicity=0.0,
        primary_radius=r1, secondary_radius=r2, distance=20.0)["Johnson V"]


def test_identical_components_give_zero():
    assert _dm(5500.0, 5500.0, 1.0, 1.0) == pytest.approx(0.0, abs=1e-12)


def test_radius_ratio_gives_five_log_ratio():
    assert _dm(5500.0, 5500.0, 1.0, 0.5) == pytest.approx(5 * np.log10(2.0), abs=1e-10)


def test_cooler_secondary_is_fainter():
    assert _dm(5900.0, 5100.0, 1.0, 1.0) > 0.0


def test_magnitude_difference_validation():
    with pytest.raises(ValueError):
        MagnitudeDifference("Johnson V", 0.1, 0.0)
    with pytest.raises(TypeError):
        MagnitudeDifference("", 0.1, 0.05)
    d = MagnitudeDifference(" Johnson V ", 0.09, 0.05)
    assert d.passband == "Johnson V" and d.value == 0.09
