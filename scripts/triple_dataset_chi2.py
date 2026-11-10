"""
Per-dataset chi2 of the ORBIT3 triple fits (zeta Aqr, HIP 111805).

Evaluates the ORBIT3 residual blocks (inner astrometry, outer astrometry,
RV Va, Vb, Vc) at the fitted elements stored in validation/results/<sys>.json,
with the quoted errors (ORBIT3 fits do not rescale). nu_k = N_k - J N_k / N,
as in pyseshat.inference.dataset_weighting. Needs numpy only (no scipy).
Writes validation/results/triple_dataset_chi2.json.
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pyseshat.orbit.orbital_elements import OrbitalElements  # noqa: E402
from pyseshat.triple.triple_elements import TripleOrbitalElements  # noqa: E402
from pyseshat.triple.triple_observation import TripleObservation  # noqa: E402
from pyseshat.triple.triple_parser import parse_triple_file  # noqa: E402
from pyseshat.triple.triple_residuals import (  # noqa: E402
    _orbit3_rv_residual_blocks, _orbit3_visual_residual_blocks)

V = ROOT / "validation"
SYSTEMS = {"ZetaAqr": "ZetaAqr.in2", "HIP111805": "HIP111805.in3"}
BLOCKS = ("inner astrometry", "outer astrometry", "RV Va", "RV Vb", "RV Vc")


def _swap_legacy(obs):
    # .in2 files tag the outer pair I1 and the inner pair I2 (reverse of ORBIT3);
    # same conversion as validation/benchmark_runner._convert_legacy_orbit3_observations
    sw = {"I1": "I2", "I2": "I1"}
    return [TripleObservation(epoch=o.epoch, theta=o.theta, rho=o.rho, sigma=o.sigma,
                              orbit_id=sw[o.orbit_id], velocity=o.velocity) for o in obs]


def _elements(p, flags):
    inner = OrbitalElements(P=p["P"], T0=p["T0"], e=p["e"], a=p["a"], Omega=p["Omega"],
                            omega=p["omega"], i=p["i"])
    outer = OrbitalElements(P=p["P2"], T0=p["T02"], e=p["e2"], a=p["a2"], Omega=p["Omega2"],
                            omega=p["omega2"], i=p["i2"])
    el = TripleOrbitalElements(outer=outer, inner=inner, K1=p.get("K1", 0.0), K2=p.get("K2", 0.0),
                               f=p.get("f", 0.0), K3=p.get("K3", 0.0), K4=p.get("K4", 0.0),
                               V0=p.get("V0", 0.0))
    for k, v in flags.items():
        el.fit_flags[k] = bool(v)
    return el


def evaluate(system, paper_config=False):
    """paper_config (HIP 111805 only): P2 fixed, A,B positions >= 1978.6, elements from
    validation/results/hip111805_paper_config.json (trf fit from the published start)."""
    parsed = parse_triple_file(str(V / "benchmark_systems" / system / SYSTEMS[system]))
    obs, rv = parsed["observations"], parsed["rv_observations"]
    if parsed["metadata"].get("format") == "in2":
        obs = _swap_legacy(obs)
    res = json.load(open(V / "results" / f"{system}.json"))
    exp = json.load(open(V / "expected_results" / f"{system}.json"))
    flags = dict(exp["fitted_flags"])
    values = res["fitted_elements"]
    if paper_config:
        obs = [o for o in obs if not (o.orbit_id == "I2" and o.epoch < 1978.6)]
        flags["P2"] = False
        pc = json.load(open(V / "results" / "hip111805_paper_config.json"))
        run = [r for r in pc["runs"] if r.get("start") == "published" and r.get("method") == "trf"][0]
        values = run["values"]
        res = {"statistics": {"n_params": sum(bool(v) for v in flags.values()),
                              "chi_square": run["chi2_final"]}}
    el = _elements(values, flags)
    X1, Y1, X2, Y2 = _orbit3_visual_residual_blocks(obs, el)
    R1, R2, R3 = _orbit3_rv_residual_blocks(rv, el)
    blocks = [np.concatenate([X1, Y1]), np.concatenate([X2, Y2]), R1, R2, R3]
    N = sum(len(b) for b in blocks)
    J = res["statistics"]["n_params"]
    chi2 = sum(float(np.sum(np.asarray(b) ** 2)) for b in blocks)
    out = {"chi2_total": chi2, "chi2_total_json": res["statistics"]["chi_square"], "N": N, "J": J,
           "datasets": {}}
    for name, b in zip(BLOCKS, blocks):
        n = len(b)
        if n == 0:
            continue
        nu = n - J * n / N
        c = float(np.sum(np.asarray(b) ** 2))
        out["datasets"][name] = {"n": n, "chi2": c, "nu": nu, "chi2_per_dof": c / nu}
    return out


if __name__ == "__main__":
    allres = {s: evaluate(s) for s in SYSTEMS}
    if (V / "results" / "hip111805_paper_config.json").exists():
        allres["HIP111805_paper"] = evaluate("HIP111805", paper_config=True)
    (V / "results" / "triple_dataset_chi2.json").write_text(json.dumps(allres, indent=2))
    for s, r in allres.items():
        print(f"{s}: chi2 {r['chi2_total']:.4f} (results json {r['chi2_total_json']:.4f}), N={r['N']}, J={r['J']}")
        for k, d in r["datasets"].items():
            print(f"   {k:17s} n={d['n']:4d} chi2/nu={d['chi2_per_dof']:.2f}")
