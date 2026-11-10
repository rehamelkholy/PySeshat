"""
MCMC posterior and corner plot for the REAL Capella data (Torres, Claret &
Young 2009; subset B of scripts/capella_real_data.py), with the
eccentricity and argument of periastron free.

Torres et al. (2009, Sect. 4) found e = 0.00087 +- 0.00021 from these data
(driven by the Hummel et al. 1994 interferometry) but adopted a circular
orbit; Torres et al. (2015, Table 1) report e = 0.00089 +- 0.00011 with
newer velocities. The benchmark fit (capella_real_data.py) holds e = 0, as
Torres et al. (2009) did; this script frees e and omega to show the
near-circular-orbit e-omega correlation on real data.

Sampling uses parameterization="sqrt_e" (mean longitude L and
sqrt(e) sin/cos omega in place of T0, e, omega; PySeshat 0.2.0), because
in (T0, e, omega) the walkers mix very poorly on this e ~ 0.001 orbit.
Results are in the standard elements. T0 (periastron epoch) is defined
only modulo P and is folded to within +-P/2 of the least-squares value.
The autocorrelation time is computed in the sampled parameters, where
the chain actually moves.

Errors: the least-squares fit uses the default dataset weighting; the
converged per-dataset error scales are applied to the quoted errors, and
emcee then samples the posterior with those errors (the MCMC likelihood
uses the errors as given).

Writes
    validation/results/Capella_real_mcmc_samples.npz
    validation/results/Capella_real_mcmc_summary.json
    validation/results/plots/Capella_real/corner.pdf

Run:  MPLBACKEND=Agg PYTHONPATH=src python scripts/generate_capella_real_corner_plot.py
(add --rerun to discard a saved chain and sample again)
"""

import dataclasses
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import capella_real_data as cr  # noqa: E402
from pyseshat.inference.optimizer import fit_visual_orbit, fit_visual_orbit_mcmc  # noqa: E402
from pyseshat.inference.reparameterization import SqrtEParameterization  # noqa: E402

SAMPLES = ROOT / "validation/results/Capella_real_mcmc_samples.npz"
SUMMARY = ROOT / "validation/results/Capella_real_mcmc_summary.json"
PLOT = ROOT / "validation/results/plots/Capella_real/corner.pdf"

N_WALKERS, N_STEPS, BURN_IN, THIN, SEED = 32, 20000, 4000, 5, 42
PARAMETERIZATION = "sqrt_e"


def wrap_t0(samples, names, t_ref):
    """T0 (periastron epoch) is defined modulo P; fold it to within
    +-P/2 of t_ref (the least-squares value), with each sample's own P.
    fit_visual_orbit_mcmc already does this since 0.2.0, so this is a
    no-op on new chains; kept so that older saved chains plot correctly.
    Returns a copy."""
    s = np.array(samples, float, copy=True)
    i, ip = names.index("T0"), names.index("P")
    P = s[:, ip]
    s[:, i] = t_ref + np.mod(s[:, i] - t_ref + P / 2, P) - P / 2
    return s


def integrated_time(samples):
    """Autocorrelation time (in thinned steps) from the flat chain, which
    emcee stores step-major as (steps, walkers, parameters)."""
    import emcee
    chain = samples.reshape(-1, N_WALKERS, samples.shape[1])
    return emcee.autocorr.integrated_time(chain, quiet=True), chain.shape[0]


def start_eccentric():
    el = cr.start()
    el.e = 0.001
    el.omega = 340.0
    el.fit_flags["e"] = True
    el.fit_flags["omega"] = True
    return el


def scaled_data():
    t, X, Y, sx, sy = cr.astrometry(**cr.SUBSETS["B"])
    rv = cr.rv_cfa()
    ls = fit_visual_orbit(start_eccentric(), t, X, Y, sx, sy, rv_data=rv,
                          weighting="dataset")
    s = ls.metadata["error_scales"]
    rv = rv.copy()
    rv[:, 2] = rv[:, 2] * float(s["rv"])
    return ls, (t, X, Y, sx * float(s["visual"]), sy * float(s["visual"]), rv), s


def run():
    ls, (t, X, Y, sx, sy, rv), scales = scaled_data()
    post = fit_visual_orbit_mcmc(start_eccentric(), t, X, Y, sx, sy, rv_data=rv,
                                 n_walkers=N_WALKERS, n_steps=N_STEPS, burn_in=BURN_IN,
                                 thin=THIN, rng=np.random.default_rng(SEED),
                                 parameterization=PARAMETERIZATION)
    np.savez_compressed(SAMPLES, samples=post.samples,
                        parameter_names=np.array(post.parameter_names),
                        t0_ref=float(ls.elements.T0))
    names = list(post.parameter_names)
    t_ref = float(ls.elements.T0)
    wrapped = wrap_t0(post.samples, names, t_ref)
    reparam = SqrtEParameterization(names, t_ref=t_ref, period=float(ls.elements.P))
    sampled = np.array([reparam.to_sampled(v) for v in wrapped])
    iL = names.index("T0")            # L takes T0's slot; it is an angle
    L0 = float(reparam.to_sampled(ls.elements.fitted_parameters())[iL])
    sampled[:, iL] = L0 + ((sampled[:, iL] - L0 + 180.0) % 360.0) - 180.0
    tau, n_kept = integrated_time(sampled)
    summ = {"error_scales": {k: float(v) for k, v in scales.items()},
            "least_squares": {n: [float(v), float(ls.uncertainties[n])]
                              for n, v in zip(ls.elements.fitted_parameter_names(),
                                              ls.elements.fitted_parameters())},
            "acceptance_fraction": float(np.mean(post.acceptance_fraction))
            if post.acceptance_fraction is not None else None,
            "parameterization": PARAMETERIZATION,
            "autocorrelation_time_thinned_steps": dict(zip(reparam.sampled_names(),
                                                          map(float, tau))),
            "kept_steps_per_walker": int(n_kept),
            "kept_steps_over_max_tau": float(n_kept / np.max(tau)),
            "T0_folded_about": t_ref,
            "n_walkers": N_WALKERS, "n_steps": N_STEPS, "burn_in": BURN_IN, "thin": THIN,
            "n_samples": int(post.samples.shape[0]), "parameter_names": names,
            "percentiles_16_50_84": {n: [float(x) for x in
                                         np.percentile(wrapped[:, i], [16, 50, 84])]
                                     for i, n in enumerate(names)}}
    json.dump(summ, open(SUMMARY, "w"), indent=1)
    return post


def plot(samples, names):
    from pyseshat.inference.mcmc_result import MCMCResult
    s = np.array(samples, float, copy=True)
    if "omega" in names:   # display only: omega in (-180, 180]
        i = names.index("omega")
        s[:, i] = ((s[:, i] + 180.0) % 360.0) - 180.0
    res = MCMCResult(elements=None, optimizer_result=None, residuals=None, covariance=None,
                     uncertainties=None, samples=s, log_probability=np.zeros(len(s)),
                     parameter_names=names)
    PLOT.parent.mkdir(parents=True, exist_ok=True)
    fig = res.corner_plot(show_grid=False, output_format="pdf", save_path=PLOT)
    plt.close(fig)
    print("wrote", PLOT)


def main():
    if SAMPLES.exists() and "--rerun" not in sys.argv:
        d = np.load(SAMPLES, allow_pickle=True)
        names = list(d["parameter_names"])
        samples = wrap_t0(d["samples"], names, float(d["t0_ref"]))
        print("reusing", SAMPLES)
    else:
        post = run()
        names = list(post.parameter_names)
        samples = wrap_t0(post.samples, names, json.load(open(SUMMARY))["T0_folded_about"])
        print("autocorrelation:", json.load(open(SUMMARY))["kept_steps_over_max_tau"],
              "x max tau")
    plot(samples, names)
    for i, n in enumerate(names):
        p = np.percentile(samples[:, i], [16, 50, 84])
        print(f"{n:6s} {p[1]:.6g} (+{p[2]-p[1]:.3g} / -{p[1]-p[0]:.3g})")


if __name__ == "__main__":
    main()
