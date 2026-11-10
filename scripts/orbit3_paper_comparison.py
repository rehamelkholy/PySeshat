"""
Compare PySeshat ORBIT3 residual statistics with those reported in the papers.

For HIP 111805 (Tokovinin & Latham 2017, ApJ 838, 54, Sect. 6) and zeta Aqr
(Tokovinin 2016, ApJ 831, 151, Sect. 3) prints, per data set, N, chi2/N and the
weighted rms in ORBIT3's definition (orbit3.pro `fitorb`: sqrt(sum(res^2/sig^2)/sum(1/sig^2))),
at (a) the published elements and (b) the converged PySeshat solution
(method="trf", numerical Jacobian, from the published start; taken from
validation/results/orbit3_convergence_check.json). numpy only.
Writes validation/results/orbit3_paper_comparison.json
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
from triple_dataset_chi2 import _elements, _swap_legacy  # noqa: E402
from pyseshat.triple.triple_parser import parse_triple_file  # noqa: E402
from pyseshat.triple.triple_residuals import (  # noqa: E402
    _orbit3_rv_residual_blocks, _orbit3_visual_residual_blocks)

V = ROOT / "validation"
FILES = {"HIP111805": "HIP111805.in3", "ZetaAqr": "ZetaAqr.in2"}


def stats(r, sig, scale):
    r, sig = np.asarray(r, float), np.asarray(sig, float)
    return {"N": int(r.size), "chi2": float(np.sum(r ** 2)), "chi2_per_N": float(np.mean(r ** 2)),
            "wrms": float(np.sqrt(np.sum(r ** 2) / np.sum(sig ** -2.0)) * scale)}


def evaluate(system, el_values, speckle_epoch=None):
    parsed = parse_triple_file(str(V / "benchmark_systems" / system / FILES[system]))
    obs, rv = parsed["observations"], parsed["rv_observations"]
    if parsed["metadata"].get("format") == "in2":
        obs = _swap_legacy(obs)
    exp = json.load(open(V / "expected_results" / f"{system}.json"))
    el = _elements(el_values, exp["fitted_flags"])
    X1, Y1, X2, Y2 = _orbit3_visual_residual_blocks(obs, el)
    R = _orbit3_rv_residual_blocks(rv, el)
    s1 = np.array([o.sigma for o in obs if o.orbit_id == "I1"])
    o2 = [o for o in obs if o.orbit_id == "I2"]
    s2 = np.array([o.sigma for o in o2])
    assert len(s1) == len(X1) and len(s2) == len(X2), "block/observation mismatch"
    out = {"inner X": stats(X1, s1, 1e3), "inner Y": stats(Y1, s1, 1e3),
           "outer X": stats(X2, s2, 1e3), "outer Y": stats(Y2, s2, 1e3)}
    if speckle_epoch is not None:
        m = np.array([o.epoch >= speckle_epoch for o in o2])
        out["outer X (speckle era)"] = stats(np.asarray(X2)[m], s2[m], 1e3)
        out["outer Y (speckle era)"] = stats(np.asarray(Y2)[m], s2[m], 1e3)
    for tag, blk in zip(("Va", "Vb", "Vc"), R):
        if len(blk):
            s = np.array([o.sigma for o in rv if o.orbit_id == tag])
            assert len(s) == len(blk)
            out[f"RV {tag}"] = stats(blk, s, 1.0)
    out["total chi2"] = float(sum(v["chi2"] for k, v in out.items() if "speckle" not in k))
    return out


def main():
    conv = json.load(open(V / "results" / "orbit3_convergence_check.json"))
    res = {}
    for system in FILES:
        exp = json.load(open(V / "expected_results" / f"{system}.json"))
        pub = dict(exp["initial_parameters"])
        run = [r for r in conv[system]["runs"] if r["start"] == "published" and r["how"] == "trf_numeric"][0]
        fit = dict(pub, **run["values"])
        se = 1978.6 if system == "HIP111805" else None   # first speckle measure of A,B in the .in3 file
        res[system] = {"published": evaluate(system, pub, se), "pyseshat_converged": evaluate(system, fit, se),
                       "converged_elements": fit}
    json.dump(res, open(V / "results" / "orbit3_paper_comparison.json", "w"), indent=1)
    for system, r in res.items():
        print(f"\n== {system}   total chi2: published {r['published']['total chi2']:.2f}, "
              f"converged {r['pyseshat_converged']['total chi2']:.2f}")
        print(f"{'data set':24s} {'N':>4s} | {'chi2/N':>7s} {'wrms':>7s} (published) | {'chi2/N':>7s} {'wrms':>7s} (converged)")
        for k, a in r["published"].items():
            if k == "total chi2":
                continue
            b = r["pyseshat_converged"][k]
            u = "km/s" if k.startswith("RV") else "mas"
            print(f"{k:24s} {a['N']:4d} | {a['chi2_per_N']:7.2f} {a['wrms']:7.2f} | {b['chi2_per_N']:7.2f} {b['wrms']:7.2f}  {u}")


if __name__ == "__main__":
    main()
