"""
Convergence diagnostic for the ORBIT3 path (HIP 111805 and zeta Aqr).

hip111805_recovery.py showed that the default ORBIT3 fit (SciPy LM with the
ORBIT3 forward-difference Jacobian, steps = 1% of each element, 1 deg for angles,
as in orbit3.pro `alleph`) stalls: from the published start it does not move,
and from displaced starts it stops far from the minimum, while method="trf"
(SciPy numerical Jacobian) lowers chi2 from 502.10 to 385.44.

For each system this script
  1. compares every column of the ORBIT3 Jacobian with a central-difference
     Jacobian (step = 1e-3 of the published sigma) at the published elements;
  2. fits from the published start with (a) LM + ORBIT3 Jacobian (current default),
     (b) LM + SciPy numerical Jacobian (same optimizer, only the Jacobian differs),
     (c) TRF + numerical Jacobian;
  3. refits from 10 starts displaced by 3 sigma and 10 by 10 sigma (random signs)
     with (b) and (c), and counts how many reach the lowest chi2 found.
Needs scipy: run in the project venv.  Writes validation/results/orbit3_convergence_check.json
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "validation"))
sys.path.insert(0, str(ROOT / "src"))

from benchmark_runner import _build_orbit3_elements, _flatten_orbit3_elements  # noqa: E402
from build_appendix_tables import ZETA  # noqa: E402  Tokovinin 2016 Table 5, (value, sigma)
from hip111805_recovery import PUB as HIP_PUB  # noqa: E402  Tokovinin & Latham 2017 Table 2
from triple_dataset_chi2 import _swap_legacy  # noqa: E402
from pyseshat.triple.orbit3_bounds import orbit3_full_bounds  # noqa: E402
from pyseshat.triple.orbit3_fit import fit_orbit3, orbit3_fit_flag_residuals, orbit3_jacobian  # noqa: E402
from pyseshat.triple.triple_fit import orbit3_to_array  # noqa: E402
from pyseshat.triple.triple_parser import parse_triple_file  # noqa: E402

V = ROOT / "validation"
ORDER = ["P", "T0", "e", "a", "Omega", "omega", "i", "K1", "K2", "f",
         "P2", "T02", "e2", "a2", "Omega2", "omega2", "i2", "K3", "K4", "V0"]
LO, HI = orbit3_full_bounds()


def setup(system):
    exp = json.load(open(V / "expected_results" / f"{system}.json"))
    fn = {"HIP111805": "HIP111805.in3", "ZetaAqr": "ZetaAqr.in2"}[system]
    parsed = parse_triple_file(str(V / "benchmark_systems" / system / fn))
    obs, rv = parsed["observations"], parsed["rv_observations"]
    if parsed["metadata"].get("format") == "in2":
        obs = _swap_legacy(obs)
    pub = HIP_PUB if system == "HIP111805" else ZETA
    return exp, obs, rv, {k: v for k, v in pub.items() if exp["fitted_flags"].get(k)}


def chi2(r):
    return float(np.sum(np.asarray(r, float) ** 2))


def legal(start):
    s = dict(start)
    for k in ("Omega", "omega", "Omega2", "omega2"):
        if k in s:
            s[k] = s[k] % 360.0
    for k in ("i", "i2"):
        if k in s:
            s[k] = min(max(abs(s[k]), 0.5), 179.5)
    for k in ("e", "e2"):
        if k in s:
            s[k] = min(abs(s[k]), 0.95)
    for k in ("P", "a", "K1", "K2", "P2", "a2", "K3", "K4"):
        if k in s:
            s[k] = abs(s[k])
    return s


def jac_check(exp, obs, rv, pub):
    el = _build_orbit3_elements(exp["initial_parameters"], exp["fitted_flags"])
    mask = np.asarray(el.fitted_mask(), bool)
    names = [n for n, m in zip(ORDER, mask) if m]
    p0 = orbit3_to_array(el)[mask]
    J = orbit3_jacobian(p0, el, obs, rv)
    out = {}
    for j, n in enumerate(names):
        h = 1e-3 * pub[n][1] if n in pub else 1e-6 * max(abs(p0[j]), 1.0)
        pp, pm = p0.copy(), p0.copy()
        pp[j] += h
        pm[j] -= h
        cd = (orbit3_fit_flag_residuals(pp, el, obs, rv) - orbit3_fit_flag_residuals(pm, el, obs, rv)) / (2 * h)
        out[n] = float(np.linalg.norm(J[:, j] - cd) / max(np.linalg.norm(cd), 1e-300))
    return out


def fit(start, exp, obs, rv, how):
    el = _build_orbit3_elements(start, exp["fitted_flags"])
    mask = np.asarray(el.fitted_mask(), bool)
    names = [n for n, m in zip(ORDER, mask) if m]
    p0 = orbit3_to_array(el)[mask]
    c0 = chi2(orbit3_fit_flag_residuals(p0, el, obs, rv))
    try:
        if how == "lm_orbit3":
            fr = fit_orbit3(el, obs, rv, method="lm", jac="orbit3")
            vals = _flatten_orbit3_elements(fr.elements)
            c1, msg, nfev = chi2(fr.residuals), str(fr.optimizer_result.message), int(fr.optimizer_result.nfev)
        else:
            if how == "lm_numeric":
                r = least_squares(orbit3_fit_flag_residuals, p0, args=(el, obs, rv), method="lm",
                                  jac="2-point", x_scale="jac")
            else:  # trf_numeric
                lo, hi = LO[mask], HI[mask]
                r = least_squares(orbit3_fit_flag_residuals, np.clip(p0, lo, hi), args=(el, obs, rv),
                                  method="trf", jac="2-point", x_scale="jac", bounds=(lo, hi))
            vals = dict(zip(names, r.x.tolist()))
            c1, msg, nfev = chi2(r.fun), str(r.message), int(r.nfev)
    except Exception as exc:
        return {"chi2_start": c0, "error": repr(exc)}
    return {"chi2_start": c0, "chi2_final": c1, "message": msg, "nfev": nfev,
            "values": {n: float(vals[n]) for n in names}}


def run_system(system):
    exp, obs, rv, pub = setup(system)
    res = {"jacobian_rel_error": jac_check(exp, obs, rv, pub), "runs": []}
    start0 = dict(exp["initial_parameters"])
    for how in ("lm_orbit3", "lm_numeric", "trf_numeric"):
        res["runs"].append({"start": "published", "trial": -1, "how": how, **fit(start0, exp, obs, rv, how)})
    rng = np.random.default_rng(2017)
    for k in (3.0, 10.0):
        for t in range(10):
            sgn = rng.choice([-1.0, 1.0], size=len(pub))
            st = dict(start0)
            for (n, (v, s)), g in zip(pub.items(), sgn):
                st[n] = v + g * k * s
            st = legal(st)
            for how in ("lm_numeric", "trf_numeric"):
                res["runs"].append({"start": f"{k:g}sigma", "trial": t, "how": how, **fit(st, exp, obs, rv, how)})
    ok = [r for r in res["runs"] if "chi2_final" in r]
    best = min(ok, key=lambda r: r["chi2_final"])
    res["chi2_min"] = best["chi2_final"]
    res["best"] = {"how": best["how"], "start": best["start"], "trial": best["trial"],
                   "d_pub": {n: (best["values"][n] - pub[n][0]) / pub[n][1] for n in pub if n in best["values"]}}
    return res


def main():
    allres = {}
    for system in ("HIP111805", "ZetaAqr"):
        r = allres[system] = run_system(system)
        print(f"\n===== {system}: lowest chi2 found {r['chi2_min']:.4f}")
        print("Jacobian column error |J_orbit3 - J_central| / |J_central|:")
        print("   " + "  ".join(f"{k}:{v:.2g}" for k, v in r["jacobian_rel_error"].items()))
        for how in ("lm_orbit3", "lm_numeric", "trf_numeric"):
            rr = [x for x in r["runs"] if x["how"] == how]
            pubrun = [x for x in rr if x["start"] == "published"][0]
            fin = [x["chi2_final"] for x in rr if "chi2_final" in x and x["start"] != "published"]
            nerr = sum("error" in x for x in rr)
            hit = sum(c - r["chi2_min"] < 0.1 for c in fin)
            line = f"{how:12s} from published: {pubrun.get('chi2_start', float('nan')):.2f} -> {pubrun.get('chi2_final', float('nan')):.4f}"
            if fin:
                line += f" | displaced starts: {hit}/{len(fin)} reach min (+0.1), worst {max(fin):.1f}"
            if nerr:
                line += f" | {nerr} errors"
            print(line)
        d = r["best"]["d_pub"]
        print(f"best solution ({r['best']['how']}, {r['best']['start']}): (fit - published)/sigma_pub:")
        print("   " + "  ".join(f"{k}:{v:+.2f}" for k, v in d.items()))
    json.dump(allres, open(V / "results" / "orbit3_convergence_check.json", "w"), indent=1)
    print("\nwrote validation/results/orbit3_convergence_check.json")


if __name__ == "__main__":
    main()
