"""
HIP 111805 recovery test for the ORBIT3 path.

The benchmark fit starts at the published solution (Tokovinin & Latham 2017,
ApJ 838, 54, Table 2, = the HIP111805.in3 header) and does not move
(initial chi2 = final chi2 = 502.1038), so it says nothing about the optimizer.
Here every one of the 20 fitted elements is displaced by k * sigma_pub with a
random sign (k = 3 and 10; 10 seeded trials each) and refitted with the default
ORBIT3 LM path. As a diagnostic of whether the published point is itself a
chi2 minimum, the published start is also refitted with method="trf"
(SciPy numerical Jacobian).

Run in the project venv (needs scipy):  python scripts\\hip111805_recovery.py
Writes validation/results/hip111805_recovery.csv and .json.
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "validation"))
sys.path.insert(0, str(ROOT / "src"))

from benchmark_runner import _build_orbit3_elements, _flatten_orbit3_elements  # noqa: E402
from pyseshat.triple.orbit3_fit import fit_orbit3  # noqa: E402
from pyseshat.triple.triple_parser import parse_triple_file  # noqa: E402
from pyseshat.triple.triple_residuals import orbit3_residuals  # noqa: E402

V = ROOT / "validation"
# Tokovinin & Latham (2017) Table 2, HIP 111805: inner = BAG 15 Ba,Bb (no suffix),
# outer = HDO 295 B,A (suffix 2; K3, K4 = outer K1, K2). Value, 1 sigma.
PUB = {
    "P": (1.50119, 0.0004), "T0": (1986.093, 0.093), "e": (0.022, 0.011), "a": (0.0385, 0.0010),
    "Omega": (334.5, 1.0), "omega": (232.9, 22.3), "i": (85.8, 1.6), "K1": (13.13, 0.25),
    "K2": (19.21, 3.1), "f": (-0.33, 0.015),
    "P2": (30.127, 0.031), "T02": (2010.179, 0.073), "e2": (0.324, 0.004), "a2": (0.3361, 0.0015),
    "Omega2": (154.25, 0.09), "omega2": (84.92, 0.18), "i2": (88.28, 0.10), "K3": (6.057, 0.14),
    "K4": (8.6, 0.23), "V0": (-22.575, 0.08),
}
LEVELS = (3.0, 10.0)
NTRIAL = 10


def load():
    exp = json.load(open(V / "expected_results" / "HIP111805.json"))
    parsed = parse_triple_file(str(V / "benchmark_systems" / "HIP111805" / "HIP111805.in3"))
    return exp, parsed["observations"], parsed["rv_observations"]


def chi2(obs, rv, el):
    return float(np.sum(orbit3_residuals(obs, rv, el) ** 2))


def run(start, flags, obs, rv, method="lm"):
    el = _build_orbit3_elements(start, flags)
    c0 = chi2(obs, rv, el)
    try:
        fr = fit_orbit3(el, obs, rv, method=method, jac="orbit3" if method == "lm" else "2-point")
    except Exception as exc:  # record failures instead of stopping
        return {"chi2_start": c0, "error": repr(exc)}
    fit = _flatten_orbit3_elements(fr.elements)
    r = np.asarray(fr.residuals, float)
    out = {"chi2_start": c0, "chi2_final": float(np.sum(r ** 2)),
           "success": bool(fr.optimizer_result.success), "message": str(fr.optimizer_result.message),
           "nfev": int(getattr(fr.optimizer_result, "nfev", -1)), "fitted": fit,
           "sigma": {k: float(v) for k, v in fr.uncertainties.items()}}
    out["d_pub"] = {k: (fit[k] - PUB[k][0]) / PUB[k][1] for k in PUB}
    return out


def main():
    exp, obs, rv = load()
    flags = exp["fitted_flags"]
    pub = {k: v[0] for k, v in PUB.items()}
    assert all(abs(pub[k] - exp["initial_parameters"][k]) < 1e-9 for k in pub), "Table 2 != .in3 header"
    runs = [("published", 0.0, -1, "lm", run(pub, flags, obs, rv, "lm")),
            ("published", 0.0, -1, "trf", run(pub, flags, obs, rv, "trf"))]
    rng = np.random.default_rng(111805)
    for k in LEVELS:
        for t in range(NTRIAL):
            sgn = rng.choice([-1.0, 1.0], size=len(PUB))
            start = {p: PUB[p][0] + s * k * PUB[p][1] for p, s in zip(PUB, sgn)}
            start["e"] = abs(start["e"])          # keep e >= 0
            start["e2"] = abs(start["e2"])
            runs.append((f"{k:g}sigma", k, t, "lm", run(start, flags, obs, rv, "lm")))
    cmin = min(r["chi2_final"] for *_, r in runs if "chi2_final" in r)
    with open(V / "results" / "hip111805_recovery.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["start", "trial", "method", "chi2_start", "chi2_final", "success", "nfev",
                    "max_abs_d_pub", "worst_param", "message"] + [f"d_{p}" for p in PUB])
        for lab, k, t, m, r in runs:
            if "error" in r:
                w.writerow([lab, t, m, r["chi2_start"], "", False, "", "", "", r["error"]])
                continue
            d = r["d_pub"]
            worst = max(d, key=lambda q: abs(d[q]))
            w.writerow([lab, t, m, r["chi2_start"], r["chi2_final"], r["success"], r["nfev"],
                        abs(d[worst]), worst, r["message"]] + [d[p] for p in PUB])
    json.dump({"published": PUB, "runs": [dict(start=l, level=k, trial=t, method=m, **r)
                                          for l, k, t, m, r in runs], "chi2_min": cmin},
              open(V / "results" / "hip111805_recovery.json", "w"), indent=1)
    print(f"lowest chi2 found: {cmin:.4f}   (published start: 502.1038)")
    for lab, k, t, m, r in runs:
        if "error" in r:
            print(f"{lab:10s} {t:3d} {m:4s} ERROR {r['error']}")
            continue
        d = r["d_pub"]
        worst = max(d, key=lambda q: abs(d[q]))
        print(f"{lab:10s} {t:3d} {m:4s} chi2 {r['chi2_start']:12.2f} -> {r['chi2_final']:9.4f}  "
              f"nfev {r['nfev']:4d}  max|fit-pub|/sig_pub {abs(d[worst]):6.2f} ({worst})  {r['message'][:40]}")


if __name__ == "__main__":
    main()
