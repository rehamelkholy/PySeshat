"""
Convergence of the default ORBIT3 fit (fix of 0.2.0).

Until 0.2.0 fit_orbit3() defaulted to SciPy LM with the translated ORBIT3
forward-difference Jacobian (increments = 1% of each element, 1 deg for angles,
as in orbit3.pro). For HIP 111805 that fit stopped at the starting point
(chi2 = 502.10) and did not recover from displaced starts, while a numerical
Jacobian reaches chi2 = 385.44 (scripts/orbit3_convergence_check.py).
"""
import numpy as np
import pytest

from test_hip111805_orbit3 import load_hip111805

from pyseshat.triple.orbit3_fit import fit_orbit3
from pyseshat.triple.triple_fit import orbit3_to_array, update_orbit3_fitted_parameters
from pyseshat.triple.triple_residuals import orbit3_residuals

# Tokovinin & Latham (2017, ApJ 838, 54) Table 2, HIP 111805, ORBIT3 element order:
# P, T0, e, a, Omega, omega, i, K1, K2, f, P2, T02, e2, a2, Omega2, omega2, i2, K3, K4, V0
SIGMA_PUB = np.array([0.0004, 0.093, 0.011, 0.0010, 1.0, 22.3, 1.6, 0.25, 3.1, 0.015,
                      0.031, 0.073, 0.004, 0.0015, 0.09, 0.18, 0.10, 0.14, 0.23, 0.08])
CHI2_PUBLISHED = 502.1038
CHI2_MINIMUM = 385.4395


def _chi2(obs, rv, el):
    return float(np.sum(orbit3_residuals(obs, rv, el) ** 2))


@pytest.fixture(scope="module")
def hip():
    d = load_hip111805()
    return d["elements"], d["observations"], d["rv_observations"]


def test_fit_orbit3_rejects_unknown_jacobian(hip):
    el, obs, rv = hip
    with pytest.raises(ValueError):
        fit_orbit3(el, obs, rv, jac="analytic")


def test_default_fit_moves_from_published_start_to_minimum(hip):
    el, obs, rv = hip
    assert _chi2(obs, rv, el) == pytest.approx(CHI2_PUBLISHED, abs=1e-3)
    fit = fit_orbit3(el, obs, rv)
    assert fit.optimizer_result.success
    assert _chi2(obs, rv, fit.elements) == pytest.approx(CHI2_MINIMUM, abs=0.05)
    # bounded default: physical ranges, no mirrored solution
    assert 0.0 <= fit.elements.inner.e < 1.0 and 0.0 <= fit.elements.outer.e < 1.0
    assert fit.elements.K1 > 0.0 and fit.elements.K3 > 0.0


def test_default_fit_recovers_from_displaced_start(hip):
    el, obs, rv = hip
    mask = np.asarray(el.fitted_mask(), bool)
    p0 = orbit3_to_array(el)[mask] + 3.0 * SIGMA_PUB[mask]
    start = update_orbit3_fitted_parameters(el, p0)
    assert _chi2(obs, rv, start) > 1.5 * CHI2_PUBLISHED
    fit = fit_orbit3(start, obs, rv)
    assert _chi2(obs, rv, fit.elements) == pytest.approx(CHI2_MINIMUM, abs=0.1)


def test_legacy_orbit3_jacobian_path_still_available(hip):
    el, obs, rv = hip
    c0 = _chi2(obs, rv, el)
    fit = fit_orbit3(el, obs, rv, method="lm", jac="orbit3")
    c = _chi2(obs, rv, fit.elements)
    assert np.isfinite(c) and c <= c0 * (1 + 1e-9)
