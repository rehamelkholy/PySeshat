# Orbit fitting (visual + RV binaries)

For a simple (non-hierarchical) binary with visual astrometry, radial
velocities, or both, this is the whole workflow — see
`examples/01_visual_rv_binary_fit.py` for the full runnable version this
page walks through, and `examples/02_mcmc_posterior_exploration.py` for
the MCMC follow-up.

## Least-squares fit

```python
import numpy as np
from pyseshat.io.orbit_parser import parse_orbit_file
from pyseshat.inference.optimizer import fit_visual_orbit

parsed = parse_orbit_file("validation/benchmark_systems/FIN379/FIN379.inp")
elements, visual, rv = parsed["elements"], parsed["visual_data"], parsed["rv_data"]

t, theta, rho, sigma = visual[:, 0], np.deg2rad(visual[:, 1]), visual[:, 2], visual[:, 3]
X, Y = rho * np.cos(theta), rho * np.sin(theta)

fit_result = fit_visual_orbit(elements, t, X, Y, sigma, sigma, rv_data=rv)
fitted = fit_result.elements
```

Note the conversion from the parser's polar `(theta, rho)` columns to
Cartesian `(X, Y)` before fitting — residuals throughout this codebase
are always computed in Cartesian coordinates specifically to avoid the
0/360-degree position-angle discontinuity corrupting a residual near
that boundary (see `docs/project_decisions.md`'s design-choices
section). `rv_data=rv` is optional; omit it (or pass an empty array) for
a visual-only fit.

`elements.fit_flags` controls which parameters the optimizer actually
varies — any subset of `OrbitalElements`' fields can be held fixed at
its starting value (not merely bounds-pinned, genuinely absent from the
optimizer's parameter vector). This is how, for instance, a joint fit
can hold `V0` fixed while fitting everything else, or a triple fit can
freeze the well-known outer orbit while exploring the poorly-constrained
inner one.

## MCMC posterior exploration

Once you have a good least-squares starting point, explore the full
posterior (parameter uncertainties, correlations, multimodality):

```python
from pyseshat.inference.optimizer import fit_visual_orbit_mcmc

posterior = fit_visual_orbit_mcmc(
    elements, t, X, Y, sigma, sigma, rv_data=rv,
    n_walkers=32, n_steps=2000, burn_in=500, thin=10,
)

fig = posterior.corner_plot()
```

Check convergence before trusting the posterior — `mcmc_diagnostics.py`
and `MCMCResult`'s `effective_sample_size`/`has_autocorrelation_time`
give you the standard emcee convergence signals rather than assuming
`n_steps` was enough by construction.

## Reading a fit's quality

`inference/likelihood.py`'s `visual_xy_residuals`/`rv_residuals` give
you the same Cartesian residuals the optimizer minimized;
`analysis/statistics.py`'s `combined_fit_summary()` turns those into
chi-square, reduced chi-square, and RMS, normalized to the same key
schema (`chi_square`, `reduced_chi_square`, `rms`, `n_data`, `n_params`)
regardless of which fitting path (plain binary, ORBIT3 triple, Al-Wardat
SED) produced them — see `docs/project_decisions.md`'s "statistics
normalization" design-choice entry.

## Angle and sign conventions

Every internal convention (Thiele-Innes element signs,
`theta = atan2(Y, X)`, which RV semi-amplitude belongs to which
component) matches Tokovinin/ORBIT's own convention, not an
independently-derived textbook one — chosen deliberately, since a small
convention mismatch (an Omega definition, a position-angle direction, an
F/G sign) can silently produce a numerically plausible but mirrored or
sign-flipped solution. The RV `K1`/`K2` sign/component assignment in
particular was independently re-derived and empirically reconfirmed (not
merely assumed) against both FIN379 and GL765 — see
`tests/test_rv_sign_convention_validation.py` and
`docs/project_decisions.md`'s "RESOLVED: RV K1/K2 sign convention"
entry.

**The one convention this codebase does not yet verify**: whether
`plot_visual_orbit()`'s X/Y axes actually correspond to the standard
North-up/East-left astronomical convention. See `visualization.md` for
the full writeup — this affects how a plotted orbit *looks*, not the fit
itself, but matters if you're reproducing a published figure.
