"""
HIP 111805 refit in the configuration of the published solution.

Tokovinin & Latham (2017, ApJ 838, 54, Sect. 6): "In the last iteration, we fixed
the outer period and used only the speckle data for A,B in fitting the remaining
19 free parameters." The benchmark instead fits all 20 elements to every
uncommented measure in HIP111805.in3 (incl. micrometer data of A,B since 1900).

Here: P2 fixed at 30.127; A,B positions restricted to epochs >= 1978.6 (the
"C Speckle data" block of the .in3 file; it also contains one Hipparcos, one AO
and three interferometric measures, which cannot be separated after parsing).
Fits from the published start with (a) the current default (LM + ORBIT3 Jacobian)
and (b) method="trf" (SciPy numerical Jacobian, bounded), and with (b) from 10
starts displaced by 3 sigma_pub. Needs scipy (project venv).
Writes validation/results/hip111805_paper_config.json
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "validation"))
sys.path.insert(0, str(ROOT / "src"))

from benchmark_runner import _build_orbit3_elements, _flatten_orbit3_elements  # noqa: E402
from hip111805_recovery import PUB  # noqa: E402
from pyseshat.triple.orbit3_fit import fit_orbit3  # noqa: E402
from pyseshat.triple.triple_parser import parse_triple_file  # noqa: E402
from pyseshat.triple.triple_residuals import (  # noqa: E402
    _orbit3_rv_residual_blocks, _orbit3_visual_residual_blocks)

V = ROOT / "validation"
SPECKLE_EPOCH = 1978.6


def blocks(obs, rv, el):
    X1, Y1, X2, Y2 = _orbit3_visual_residual_blocks(obs, el)
    R1, R2, R3 = _orbit3_rv_residual_blocks(rv, el)
    return {"inner X": X1, "inner Y": Y1, "outer X": X2, "outer Y": Y2, "RV Va": R1, "RV Vb": R2, "RV Vc": R3}


def run(start, flags, obs, rv, method):
    el = _build_orbit3_elements(start, flags)
    c0 = float(sum(np.sum(np.asarray(b) ** 2) for b in blocks(obs, rv, el).values()))
    fr = fit_orbit3(el, obs, rv, method=method, jac="orbit3" if method == "lm" else "2-point")
    fit = _flatten_orbit3_elements(fr.elements)
    bl = blocks(obs, rv, fr.elements)
    names = [k for k, v in flags.items() if v]
    out = {"method": method, "chi2_start": c0, "chi2_final": float(np.sum(np.asarray(fr.residuals) ** 2)),
           "message": str(fr.optimizer_result.message), "nfev": int(fr.optimizer_result.nfev),
           "values": fit, "sigma": {k: float(v) for k, v in fr.uncertainties.items()},
           "chi2_per_N": {k: float(np.mean(np.asarray(b) ** 2)) for k, b in bl.items() if len(b)},
           "N": {k: len(b) for k, b in bl.items() if len(b)}}
    out["d_comb"] = {k: (fit[k] - PUB[k][0]) / np.hypot(PUB[k][1], out["sigma"].get(k, 0.0)) for k in names}
    return out


def main():
    exp = json.load(open(V / "expected_results" / "HIP111805.json"))
    parsed = parse_triple_file(str(V / "benchmark_systems" / "HIP111805" / "HIP111805.in3"))
    obs = [o for o in parsed["observations"] if not (o.orbit_id == "I2" and o.epoch < SPECKLE_EPOCH)]
    rv = parsed["rv_observations"]
    flags = dict(exp["fitted_flags"], P2=False)
    pub = dict(exp["initial_parameters"])
    runs = [run(pub, flags, obs, rv, "lm"), run(pub, flags, obs, rv, "trf")]
    rng = np.random.default_rng(838)
    names = [k for k, v in flags.items() if v]
    for t in range(10):
        st = dict(pub)
        for k in names:
            st[k] = PUB[k][0] + rng.choice([-1.0, 1.0]) * 3.0 * PUB[k][1]
        for k in ("Omega", "omega", "Omega2", "omega2"):
            st[k] %= 360.0
        st["e"], st["e2"] = abs(st["e"]), abs(st["e2"])
        try:
            r = run(st, flags, obs, rv, "trf")
            r["start"] = f"3sigma #{t}"
        except Exception as exc:
            r = {"start": f"3sigma #{t}", "error": repr(exc)}
        runs.append(r)
    runs[0]["start"] = runs[1]["start"] = "published"
    json.dump({"config": "P2 fixed, A,B epochs >= 1978.6", "runs": runs},
              open(V / "results" / "hip111805_paper_config.json", "w"), indent=1)
    n_outer = sum(o.orbit_id == "I2" for o in obs)
    print(f"A,B positions used: {n_outer} (of 100); 19 free elements, P2 fixed at {pub['P2']}")
    ok = [r for r in runs if "chi2_final" in r]
    cmin = min(r["chi2_final"] for r in ok)
    for r in runs:
        if "error" in r:
            print(f"{r['start']:12s} trf  ERROR {r['error']}")
            continue
        print(f"{r['start']:12s} {r['method']:4s} chi2 {r['chi2_start']:9.2f} -> {r['chi2_final']:9.4f}  nfev {r['nfev']:4d}  {r['message'][:38]}")
    print(f"\nlowest chi2: {cmin:.4f};  displaced trf starts reaching it (+0.1): "
          f"{sum(r['chi2_final'] - cmin < 0.1 for r in ok[2:])}/{len(ok) - 2}")
    best = runs[1]
    print("\ntrf from published start, chi2/N per data set:")
    print("   " + "  ".join(f"{k} {v:.2f} (N={best['N'][k]})" for k, v in best["chi2_per_N"].items()))
    print("(fit - published) / combined sigma:")
    d = best["d_comb"]
    print("   " + "  ".join(f"{k}:{v:+.2f}" for k, v in d.items()))
    print(f"within 1 sigma: {sum(abs(v) <= 1 for v in d.values())}/{len(d)},  max |d| = {max(abs(v) for v in d.values()):.2f}")
    print("sigma_PySeshat / sigma_published:")
    print("   " + "  ".join(f"{k}:{best['sigma'][k] / PUB[k][1]:.2f}" for k in d))


if __name__ == "__main__":
    main()
