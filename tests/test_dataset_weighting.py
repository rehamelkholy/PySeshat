import numpy as np
import pytest

from pyseshat.inference.dataset_weighting import (
    combine_error_floor,
    dataset_diagnostics,
)
from pyseshat.inference.optimizer import fit_visual_orbit
from pyseshat.orbit.orbit_model import compute_visual_orbit
from pyseshat.orbit.orbital_elements import OrbitalElements
from pyseshat.orbit.rv_model import compute_rv_model


def _truth():
    return OrbitalElements(
        P=10.0, T0=2.0, e=0.3, a=0.5, Omega=40.0, omega=70.0,
        i=60.0, K1=5.0, K2=7.0, V0=1.0,
    )


def _data(rng, vis_noise, rv_noise, claimed_vis, claimed_rv):
    el = _truth()
    t = np.linspace(0.0, 14.0, 30)
    orb = compute_visual_orbit(t, el)
    X = orb["X"] + rng.normal(0, vis_noise, t.size)
    Y = orb["Y"] + rng.normal(0, vis_noise, t.size)
    sx = np.full(t.size, claimed_vis)
    sy = np.full(t.size, claimed_vis)
    tr = np.linspace(0.0, 14.0, 25)
    rv = np.zeros((tr.size, 4))
    rv[:, 0] = tr
    rv[:, 2] = claimed_rv
    rv[:, 3] = 1
    rv[:, 1] = compute_rv_model(rv, el) + rng.normal(0, rv_noise, tr.size)
    return t, X, Y, sx, sy, rv


def _start():
    el = _truth()
    el.P, el.e, el.a = 10.2, 0.28, 0.52
    return el


def test_diagnostics_split():
    r = np.concatenate([np.ones(6), 2 * np.ones(4)])
    d = dataset_diagnostics(r, n_visual=3, n_rv=4, n_parameters=2)
    assert d["visual"]["n"] == 6 and d["rv"]["n"] == 4
    assert d["visual"]["chi2"] == pytest.approx(6.0)
    assert d["rv"]["chi2"] == pytest.approx(16.0)
    assert d["total"]["dof"] == pytest.approx(8.0)
    assert d["rv"]["rms"] == pytest.approx(2.0)


def test_error_floor_quadrature():
    out = combine_error_floor(np.array([3.0]), 4.0)
    assert out[0] == pytest.approx(5.0)


def test_dataset_weighting_rescales_underestimated_rv_errors():
    rng = np.random.default_rng(1)
    t, X, Y, sx, sy, rv = _data(rng, 0.005, 1.0, 0.005, 0.1)
    fit = fit_visual_orbit(
        _start(), t, X, Y, sx, sy, rv_data=rv, weighting="fixed"
    )
    base = fit.metadata["dataset_diagnostics"]
    assert base["rv"]["chi2_per_dof"] > 20

    fitw = fit_visual_orbit(
        _start(), t, X, Y, sx, sy, rv_data=rv, weighting="dataset"
    )
    dw = fitw.metadata["dataset_diagnostics"]
    assert dw["rv"]["chi2_per_dof"] == pytest.approx(1.0, abs=0.02)
    assert fitw.metadata["error_scales"]["rv"] > 5
    assert fitw.metadata["error_scales"]["visual"] >= 1.0


def test_auto_is_dataset_for_combined_fits():
    rng = np.random.default_rng(2)
    t, X, Y, sx, sy, rv = _data(rng, 0.005, 0.1, 0.005, 0.1)
    a = fit_visual_orbit(_start(), t, X, Y, sx, sy, rv_data=rv,
                         weighting="auto")
    b = fit_visual_orbit(
        _start(), t, X, Y, sx, sy, rv_data=rv, weighting="dataset"
    )
    np.testing.assert_allclose(a.residuals, b.residuals)
    assert a.metadata["weighting"] == "dataset"
    v = fit_visual_orbit(_start(), t, X, Y, sx, sy)
    assert v.metadata["weighting"] == "fixed"


def test_invalid_weighting_raises():
    rng = np.random.default_rng(3)
    t, X, Y, sx, sy, rv = _data(rng, 0.005, 0.1, 0.005, 0.1)
    with pytest.raises(ValueError):
        fit_visual_orbit(_start(), t, X, Y, sx, sy, rv_data=rv,
                         weighting="bogus")
