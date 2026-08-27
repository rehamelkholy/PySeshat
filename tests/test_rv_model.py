from pathlib import Path

import numpy as np

import pytest

from pyseshat.io.orbit_parser import (
    parse_orbit_file
)

from pyseshat.orbit.rv_model import (
    compute_rv_orbit,
    compute_rv_model
)

# ------------------------------------------------------
# Benchmark datasets
#
# NOTE: this only checks internal self-consistency of the
# RV model (finite output, non-trivial variation, and that
# compute_rv_model's per-component dispatch reproduces
# compute_rv_orbit called directly). It does NOT validate
# which physical star (primary/secondary) the K1 vs K2 sign
# convention actually corresponds to against a literature
# SB2 solution -- see docs/project_decisions.md, "RV K1/K2
# sign-convention audit finding", for why that is currently
# an open question rather than something asserted here.
# ------------------------------------------------------

ROOT = Path("validation/benchmark_systems")

benchmark_files = [

    ROOT / "FIN379" / "FIN379.inp",

    ROOT / "GL765" / "GL765.inp"
]


@pytest.mark.parametrize(
    "filepath",
    benchmark_files,
    ids=[f.name for f in benchmark_files]
)
def test_rv_model_self_consistency(filepath):

    parsed = parse_orbit_file(
        str(filepath)
    )

    elements = parsed["elements"]

    rv_data = parsed["rv_data"]

    if len(rv_data) == 0:

        pytest.skip(
            f"{filepath.name} has no RV data."
        )

    t = rv_data[:, 0]

    component = rv_data[:, 3].astype(int)

    # ----------------------------------------------
    # Primary/secondary RV, evaluated directly
    # ----------------------------------------------

    rv1 = compute_rv_orbit(
        t,
        elements,
        component=0
    )

    rv2 = compute_rv_orbit(
        t,
        elements,
        component=1
    )

    assert np.all(np.isfinite(rv1))

    assert np.all(np.isfinite(rv2))

    # A real spectroscopic orbit varies with phase; a
    # constant model output usually means K1/K2 == 0 or the
    # ephemeris collapsed (e.g. P == 0), either of which is a
    # parsing/model regression worth catching here.

    if elements.K1 != 0.0:

        assert np.ptp(rv1) > 0.0

    if elements.K2 != 0.0:

        assert np.ptp(rv2) > 0.0

    # ----------------------------------------------
    # compute_rv_model must reproduce compute_rv_orbit
    # component-by-component, for whichever component
    # encoding (0/1 or 1/2) this file actually uses.
    # ----------------------------------------------

    rv_model = compute_rv_model(
        rv_data,
        elements
    )

    assert rv_model.shape == t.shape

    assert np.all(np.isfinite(rv_model))

    has_zero = np.any(component == 0)

    primary_label, secondary_label = (
        (0, 1) if has_zero else (1, 2)
    )

    mask_primary = component == primary_label

    mask_secondary = component == secondary_label

    if np.any(mask_primary):

        np.testing.assert_allclose(

            rv_model[mask_primary],

            compute_rv_orbit(
                t[mask_primary],
                elements,
                component=0
            )

        )

    if np.any(mask_secondary):

        np.testing.assert_allclose(

            rv_model[mask_secondary],

            compute_rv_orbit(
                t[mask_secondary],
                elements,
                component=1
            )

        )
