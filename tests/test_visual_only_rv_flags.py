"""
Visual-only fits hold K1, K2 and V0 fixed (fixed in 0.2.0).

Before the fix the OrbitalElements defaults (all flags True) left the
RV-only parameters "fitted" without RV data: they were counted in the
degrees of freedom, reported with zero uncertainty, and drifted without
bound in MCMC.
"""

import numpy as np

from pyseshat.inference.optimizer import fit_visual_orbit, fit_visual_orbit_mcmc
from pyseshat.orbit.orbit_model import compute_visual_orbit
from pyseshat.orbit.orbital_elements import OrbitalElements


def _data():
    el = OrbitalElements(P=5.0, T0=2001.2, e=0.35, a=0.5, Omega=40.0,
                         omega=120.0, i=60.0, K1=10.0)
    t = np.linspace(2000.0, 2012.0, 40)
    xy = compute_visual_orbit(t, el)
    s = np.full_like(t, 0.004)
    return el, t, np.asarray(xy["X"]), np.asarray(xy["Y"]), s


def test_least_squares_drops_rv_parameters_without_rv_data():
    el, t, X, Y, s = _data()
    fit = fit_visual_orbit(el, t, X, Y, s, s)
    names = fit.elements.fitted_parameter_names()
    assert not {"K1", "K2", "V0"} & set(names)
    assert fit.metadata["dataset_diagnostics"]["total"]["dof"] == 2 * 40 - 7
    assert fit.elements.K1 == 10.0
    assert el.fit_flags["K1"] is True          # caller's elements untouched


def test_mcmc_does_not_sample_rv_parameters_without_rv_data():
    el, t, X, Y, s = _data()
    post = fit_visual_orbit_mcmc(el, t, X, Y, s, s, n_walkers=16, n_steps=50,
                                 burn_in=10, rng=np.random.default_rng(0))
    assert list(post.parameter_names) == ["P", "T0", "e", "a", "Omega", "omega", "i"]
    assert post.samples.shape[1] == 7
