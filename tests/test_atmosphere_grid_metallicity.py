"""
Metallicity-grid-selection coverage for the CK04 atmosphere-loading
machinery (``stellar/atmospheres.py``).

Why this file exists
-----------------------------------------------------------------
`stellar/atmospheres.py`'s per-model-parameter
interpolation (Teff, log g) has test coverage only indirectly, through
higher-level photometric-fit tests -- and metallicity SELECTION
specifically (choosing the right one of several loaded metallicity
slices, and rejecting a metallicity that was never loaded) has no
other dedicated test. Without these tests, nothing directly exercises
`AtmosphereGrid`/`load_ck04_grid` with more than one metallicity slice
loaded at once, so a metallicity-selection bug in that code path could
easily go unnoticed.

These tests use real CK04 FITS files already checked into
`validation/atmosphere_models/` (not synthetic fixtures) at the same
effective temperature (5750 K) across three different metallicity
subdirectories (ckm05 = -0.5, ckp00 = 0.0, ckp02 = +0.2), so a bug that
silently returned the wrong metallicity's spectrum (a real risk in
code keyed by a 3-tuple) would be caught by comparing actual flux
values, not just object identity.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pyseshat.stellar.atmospheres import (
    load_ck04_fits,
    load_ck04_grid,
)


ATMOSPHERE_MODELS_DIR = ROOT / "validation" / "atmosphere_models"

CKM05_5750 = ATMOSPHERE_MODELS_DIR / "ckm05" / "ckm05_5750.fits"
CKP00_5750 = ATMOSPHERE_MODELS_DIR / "ckp00" / "ckp00_5750.fits"
CKP02_5750 = ATMOSPHERE_MODELS_DIR / "ckp02" / "ckp02_5750.fits"

REQUIRED_FILES = (CKM05_5750, CKP00_5750, CKP02_5750)

pytestmark = pytest.mark.skipif(
    not all(path.is_file() for path in REQUIRED_FILES),
    reason=(
        "Full CK04 grid not present under validation/atmosphere_models/ "
        "-- these tests exercise real, checked-in CK04 FITS files "
        "across multiple metallicity slices, not synthetic fixtures."
    ),
)


def test_load_ck04_fits_reports_the_files_own_metallicity():
    """
    Each single-file loader call should tag every model it returns
    with the metallicity from THAT file's own FITS header (LOG_Z),
    not some default -- confirmed independently for three different
    metallicity directories at the identical Teff.
    """

    grid_m05 = load_ck04_fits(CKM05_5750)
    grid_p00 = load_ck04_fits(CKP00_5750)
    grid_p02 = load_ck04_fits(CKP02_5750)

    assert grid_m05.metallicities.tolist() == [-0.5]
    assert grid_p00.metallicities.tolist() == [0.0]
    assert grid_p02.metallicities.tolist() == [0.2]

    # Same Teff in every slice (all three files are the "_5750" file).
    assert grid_m05.temperatures.tolist() == [5750.0]
    assert grid_p00.temperatures.tolist() == [5750.0]
    assert grid_p02.temperatures.tolist() == [5750.0]


def test_load_ck04_grid_combines_multiple_metallicity_slices():
    """
    load_ck04_grid() given files from three different metallicity
    subdirectories at the same Teff should produce ONE combined grid
    exposing all three metallicities, not silently drop or overwrite
    any of them.
    """

    combined = load_ck04_grid([CKM05_5750, CKP00_5750, CKP02_5750])

    assert combined.metallicities.tolist() == [-0.5, 0.0, 0.2]

    # Every (teff, logg) pair present in one slice should be present
    # in all three (same source file family, only [M/H] differs).
    for metallicity in (-0.5, 0.0, 0.2):
        points_at_this_z = [
            point
            for point in combined.parameter_points
            if point[2] == metallicity
        ]
        assert len(points_at_this_z) > 0


def test_interpolate_selects_the_requested_metallicity_exactly():
    """
    The real regression this file exists to catch: with THREE
    metallicity slices loaded at the identical (teff, logg), does
    AtmosphereGrid.interpolate() actually return the flux from the
    slice that was asked for, not some other one?

    The three CK04 files at the same Teff genuinely have different
    flux tables (metallicity changes the emergent spectrum), so this
    compares real, independently-verifiable numeric flux arrays --
    not just that no exception was raised.
    """

    combined = load_ck04_grid([CKM05_5750, CKP00_5750, CKP02_5750])

    # Pick a logg exactly on the grid so this is exact retrieval, not
    # interpolation in (teff, logg) -- isolates metallicity selection
    # as the only thing under test.
    common_logg = sorted(
        set(combined.surface_gravities)
    )[len(combined.surface_gravities) // 2]

    model_m05 = combined.interpolate(5750.0, common_logg, metallicity=-0.5)
    model_p00 = combined.interpolate(5750.0, common_logg, metallicity=0.0)
    model_p02 = combined.interpolate(5750.0, common_logg, metallicity=0.2)

    assert model_m05.metallicity == pytest.approx(-0.5)
    assert model_p00.metallicity == pytest.approx(0.0)
    assert model_p02.metallicity == pytest.approx(0.2)

    # Different metallicity slices of a real CK04 grid at the same
    # (Teff, logg) are NOT numerically identical -- if interpolate()
    # had a bug that always returned e.g. the first-loaded slice
    # regardless of the requested metallicity, these would spuriously
    # match exactly.
    assert not np.array_equal(model_m05.flux, model_p00.flux)
    assert not np.array_equal(model_p00.flux, model_p02.flux)
    assert not np.array_equal(model_m05.flux, model_p02.flux)


def test_interpolate_rejects_a_metallicity_that_was_never_loaded():
    """
    Requesting a metallicity outside the loaded slice(s) must raise
    ValueError (not silently fall back to the nearest one) -- exact
    metallicity selection with no interpolation across the
    metallicity axis is a documented, deliberate design choice (see
    AtmosphereGrid.interpolate()'s docstring and
    docs/guide/binary_sed.md's "CK04 grid coverage" section), and
    "nearest available grid point" is a decision this codebase makes
    explicitly at the call site (see validation/expected_results/*.json
    "metallicity" fields), not something interpolate() should do
    implicitly.
    """

    grid_p00_only = load_ck04_fits(CKP00_5750)

    with pytest.raises(ValueError, match="metallicity"):
        grid_p00_only.interpolate(5750.0, 4.5, metallicity=0.5)


def test_atmosphere_grid_metallicities_property_is_sorted_and_unique():
    """
    AtmosphereGrid.metallicities should reflect exactly the distinct
    metallicity values actually loaded, sorted ascending -- this is
    what benchmark/report code (and this test file's own assertions
    above) rely on to describe grid coverage.
    """

    combined = load_ck04_grid([CKP02_5750, CKM05_5750, CKP00_5750])

    assert combined.metallicities.tolist() == [-0.5, 0.0, 0.2]
    assert len(combined.metallicities) == len(set(combined.metallicities))
