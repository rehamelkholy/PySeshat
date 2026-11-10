"""
T0 folding of binary MCMC samples (added in 0.2.0).

The orbit model depends on T0 only through (t - T0) / P, so T0 + k P is
the same orbit; on near-circular orbits emcee walkers drift across many
periods. ``_wrap_periodic_mcmc_samples`` folds T0 to within P/2 of the
least-squares T0.
"""

import numpy as np
import pytest

from pyseshat.inference.optimizer import _wrap_periodic_mcmc_samples
from pyseshat.orbit.orbit_model import compute_visual_orbit
from pyseshat.orbit.orbital_elements import OrbitalElements


def _elements(T0=2000.3, P=2.5):
    return OrbitalElements(P=P, T0=T0, e=0.3, a=0.5, Omega=40.0,
                           omega=120.0, i=60.0)


def test_model_is_periodic_in_t0():
    t = np.linspace(1990.0, 2010.0, 50)
    base = compute_visual_orbit(t, _elements())
    shifted = compute_visual_orbit(t, _elements(T0=2000.3 + 7 * 2.5))
    assert base.keys() == shifted.keys()
    for key in base:
        assert np.allclose(np.asarray(base[key]), np.asarray(shifted[key]), atol=1e-9), key


def test_t0_folded_within_half_period_using_sample_period():
    names = ["P", "T0", "omega"]
    rng = np.random.default_rng(1)
    P = 2.5 + 1e-4 * rng.standard_normal(200)
    k = rng.integers(-300, 300, size=200)
    t0_true = 2000.3 + 0.01 * rng.standard_normal(200)
    samples = np.column_stack([P, t0_true + k * P, 400.0 + rng.uniform(0, 10, 200)])

    out = _wrap_periodic_mcmc_samples(samples, names, reference_elements=_elements())

    assert np.allclose(out[:, 1], t0_true, atol=1e-9)
    assert np.all(np.abs(out[:, 1] - 2000.3) <= 0.5 * P + 1e-12)
    assert np.all((out[:, 2] >= 0.0) & (out[:, 2] < 360.0))
    assert np.array_equal(out[:, 0], P)          # P untouched
    assert samples[0, 1] == t0_true[0] + k[0] * P[0]  # input not modified


def test_t0_folded_with_fixed_period():
    names = ["T0", "e"]
    samples = np.array([[2000.3 + 5 * 2.5, 0.1], [2000.3 - 3 * 2.5 + 0.2, 0.2]])
    out = _wrap_periodic_mcmc_samples(samples, names, reference_elements=_elements())
    assert out[:, 0] == pytest.approx([2000.3, 2000.5])


def test_without_reference_t0_unchanged():
    names = ["T0", "omega"]
    samples = np.array([[2100.0, 370.0]])
    out = _wrap_periodic_mcmc_samples(samples, names)
    assert out[0, 0] == 2100.0 and out[0, 1] == pytest.approx(10.0)


def test_triple_names_and_mapping_reference():
    names = ["P", "T0", "omega", "P2", "T02", "Omega2", "omega2"]
    ref = {"P": 2.0, "T0": 2000.0, "P2": 30.0, "T02": 1990.0}
    samples = np.array([[2.0, 2000.1 + 4 * 2.0, -10.0, 30.0, 1990.5 - 3 * 30.0, 725.0, 365.0]])
    out = _wrap_periodic_mcmc_samples(
        samples, names, reference_elements=ref,
        angle_names=("Omega", "omega", "Omega2", "omega2"),
        epoch_period_pairs=(("T0", "P"), ("T02", "P2")))
    assert out[0] == pytest.approx([2.0, 2000.1, 350.0, 30.0, 1990.5, 5.0, 5.0])
