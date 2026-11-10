"""
The optional ``parameterization="sqrt_e"`` for binary MCMC (added in 0.2.0).

Checks the transformation (round trip, Jacobian, covariance), the
rejection rules, and that the sampled posterior agrees with the standard
parameterization on a well-constrained eccentric orbit.
"""

import math

import numpy as np
import pytest

from pyseshat.inference.optimizer import fit_visual_orbit, fit_visual_orbit_mcmc
from pyseshat.inference.reparameterization import (
    SqrtEParameterization,
    sqrt_e_log_probability,
)
from pyseshat.orbit.orbit_model import compute_visual_orbit
from pyseshat.orbit.orbital_elements import OrbitalElements


NAMES = ["P", "T0", "e", "a", "Omega", "omega", "i"]
VALUES = np.array([12.5, 2003.7, 0.31, 0.42, 63.0, 211.0, 48.0])


def _reparam():
    return SqrtEParameterization(NAMES, t_ref=2004.0, period=12.5)


def test_sampled_names_replace_in_place():
    assert _reparam().sampled_names() == ["P", "L", "h", "a", "Omega", "k", "i"]


@pytest.mark.parametrize("omega", [0.0, 10.0, 179.0, 211.0, 359.5])
@pytest.mark.parametrize("dT0", [-6.0, -0.3, 0.0, 4.1])
def test_round_trip(omega, dT0):
    r = _reparam()
    v = VALUES.copy()
    v[5] = omega
    v[1] = 2004.0 + dT0          # within P/2 of t_ref
    back = r.to_standard(r.to_sampled(v))
    assert back[1] == pytest.approx(v[1], abs=1e-9)
    assert back[2] == pytest.approx(v[2], rel=1e-12)
    assert ((back[5] - v[5] + 180) % 360 - 180) == pytest.approx(0.0, abs=1e-9)
    assert np.allclose(back[[0, 3, 4, 6]], v[[0, 3, 4, 6]])


def test_to_standard_folds_t0_within_half_period():
    r = _reparam()
    q = r.to_sampled(VALUES)
    q[1] += 360.0 * 7            # L is an angle: seven extra turns
    out = r.to_standard(q)
    assert abs(out[1] - 2004.0) <= 0.5 * 12.5
    assert out[1] == pytest.approx(VALUES[1], abs=1e-9)


def test_log_jacobian_matches_numerical_determinant():
    r = _reparam()
    q0 = r.to_sampled(VALUES)

    def jac_det(q):
        n = len(q)
        J = np.zeros((n, n))
        for j in range(n):
            h = 1e-6 * max(abs(q[j]), 1.0)
            up, dn = q.copy(), q.copy()
            up[j] += h
            dn[j] -= h
            J[:, j] = (r.to_standard(up) - r.to_standard(dn)) / (2 * h)
        return abs(np.linalg.det(J))

    # log|J| - log_jacobian must be the same constant at different points
    q1 = q0.copy()
    q1[0] = 30.0
    q1[2], q1[5] = 0.1, -0.5
    c0 = math.log(jac_det(q0)) - r.log_jacobian(q0)
    c1 = math.log(jac_det(q1)) - r.log_jacobian(q1)
    assert c0 == pytest.approx(c1, abs=1e-5)
    assert c0 == pytest.approx(math.log(1.0 / math.pi), abs=1e-5)


def test_fixed_period_has_zero_jacobian_term():
    names = ["T0", "e", "a", "omega"]
    r = SqrtEParameterization(names, t_ref=2004.0, period=12.5)
    q = r.to_sampled([2003.0, 0.2, 0.4, 30.0])
    assert r.log_jacobian(q) == 0.0
    assert r.to_standard(q)[0] == pytest.approx(2003.0, abs=1e-9)


def test_requires_t0_e_omega():
    with pytest.raises(ValueError, match="omega"):
        SqrtEParameterization(["P", "T0", "e", "a"], t_ref=0.0, period=1.0)


def test_log_probability_rejects_e_ge_1_and_adds_jacobian():
    r = _reparam()
    q = r.to_sampled(VALUES)
    seen = []

    def logp(v, offset):
        seen.append(np.array(v))
        return offset

    assert sqrt_e_log_probability(q, r, logp, (2.0,)) == pytest.approx(
        2.0 + math.log(12.5))
    assert np.allclose(seen[0][[0, 2, 3, 4, 6]], VALUES[[0, 2, 3, 4, 6]])
    bad = q.copy()
    bad[2], bad[5] = 0.8, 0.7
    assert sqrt_e_log_probability(bad, r, logp, (2.0,)) == -np.inf


def test_transform_covariance_linear_map():
    r = _reparam()
    C = np.diag((0.01 * np.abs(VALUES) + 1e-3) ** 2)
    out = r.transform_covariance(VALUES, C)
    assert out.shape == (7, 7)
    assert np.allclose(out, out.T)
    assert np.all(np.linalg.eigvalsh(out) > 0)
    assert r.transform_covariance(VALUES, None) is None


def test_invalid_parameterization_name():
    el = OrbitalElements(P=5.0, T0=2001.0, e=0.3, a=0.5, Omega=40.0,
                         omega=120.0, i=60.0)
    t = np.linspace(2000.0, 2010.0, 20)
    xy = compute_visual_orbit(t, el)
    s = np.full_like(t, 0.002)
    with pytest.raises(ValueError, match="parameterization"):
        fit_visual_orbit_mcmc(el, t, np.asarray(xy["X"]), np.asarray(xy["Y"]), s, s,
                              n_walkers=16, n_steps=10, burn_in=0,
                              parameterization="ecosw")


def test_posterior_agrees_with_standard_parameterization():
    truth = OrbitalElements(P=5.0, T0=2001.2, e=0.35, a=0.5, Omega=40.0,
                            omega=120.0, i=60.0)
    rng = np.random.default_rng(3)
    t = np.sort(rng.uniform(2000.0, 2012.0, 40))
    xy = compute_visual_orbit(t, truth)
    s = np.full_like(t, 0.004)
    X = np.asarray(xy["X"]) + s * rng.standard_normal(t.size)
    Y = np.asarray(xy["Y"]) + s * rng.standard_normal(t.size)

    def start():
        return OrbitalElements(P=5.0, T0=2001.2, e=0.35, a=0.5, Omega=40.0,
                               omega=120.0, i=60.0)

    ls = fit_visual_orbit(start(), t, X, Y, s, s)
    runs = {}
    for name in ("standard", "sqrt_e"):
        runs[name] = fit_visual_orbit_mcmc(
            start(), t, X, Y, s, s, n_walkers=32, n_steps=3000, burn_in=1000,
            rng=np.random.default_rng(7), parameterization=name)

    names = list(runs["sqrt_e"].parameter_names)
    assert names == list(runs["standard"].parameter_names)
    for j, n in enumerate(names):
        a = runs["standard"].samples[:, j]
        b = runs["sqrt_e"].samples[:, j]
        sd = ls.uncertainties[n]
        assert np.median(b) == pytest.approx(np.median(a), abs=0.25 * sd), n
        assert np.std(b) == pytest.approx(np.std(a), rel=0.25), n
    # log_probability is reported on the standard posterior
    lp_std = runs["standard"].log_probability
    lp_sq = runs["sqrt_e"].log_probability
    assert np.max(lp_sq) == pytest.approx(np.max(lp_std), abs=1.0)
