"""
Sensitivity of combined visual+RV benchmark fits to dataset-level weighting.

Fits each combined benchmark with ``weighting="fixed"`` (per-datum sigmas
only) and ``weighting="dataset"`` (visual and RV errors rescaled so each
dataset has reduced chi2 = 1) and writes a JSON/CSV comparison.
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np

from pyseshat.inference.optimizer import fit_visual_orbit
from pyseshat.io.orbit_parser import parse_orbit_file

ROOT = Path(__file__).resolve().parents[1]
SYSTEMS = {
    "FIN379": "validation/benchmark_systems/FIN379/FIN379.inp",
    "GL765": "validation/benchmark_systems/GL765/GL765.inp",
    "Capella": "validation/benchmark_systems/Capella_orbit/Capella_orbit.inp",
}


def run(path, weighting):
    parsed = parse_orbit_file(str(path))
    vd = parsed["visual_data"]
    th = np.deg2rad(vd[:, 1])
    rho = vd[:, 2]
    s = vd[:, 3]
    X, Y = rho * np.cos(th), rho * np.sin(th)
    # same convention as validation/benchmark_runner.py
    return fit_visual_orbit(
        parsed["elements"], vd[:, 0], X, Y, s, s,
        rv_data=parsed["rv_data"], weighting=weighting,
    )


def main(out=ROOT / "validation/results/weighting_sensitivity"):
    rows, full = [], {}
    for name, rel in SYSTEMS.items():
        path = ROOT / rel
        fits = {w: run(path, w) for w in ("fixed", "dataset")}
        a, b = fits["fixed"], fits["dataset"]
        names = a.elements.fitted_parameter_names()
        pa, pb = a.elements.fitted_parameters(), b.elements.fitted_parameters()
        ua = [a.uncertainties[n] for n in names]
        ub = [b.uncertainties[n] for n in names]
        full[name] = {
            "diagnostics_fixed": a.metadata["dataset_diagnostics"],
            "diagnostics_dataset": b.metadata["dataset_diagnostics"],
            "error_scales": b.metadata["error_scales"],
        }
        for n, x, y, sa, sb in zip(names, pa, pb, ua, ub):
            rows.append({
                "system": name, "parameter": n,
                "fixed": x, "sigma_fixed": sa,
                "dataset": y, "sigma_dataset": sb,
                "sigma_ratio": sb / sa if sa else float("nan"),
                "shift_in_sigma_fixed": (y - x) / sa if sa else float("nan"),
            })
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(full, open(f"{out}.json", "w"), indent=2)
    with open(f"{out}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    return rows, full


if __name__ == "__main__":
    rows, full = main()
    for k, v in full.items():
        print(k, "scales", v["error_scales"])
        for w in ("fixed", "dataset"):
            d = v["diagnostics_" + w]
            print("  ", w, {t: round(e["chi2_per_dof"], 2) for t, e in d.items()})
    for r in rows:
        print(f'{r["system"]:8s}{r["parameter"]:8s}{r["fixed"]:12.5g}'
              f'{r["dataset"]:12.5g} sig_ratio={r["sigma_ratio"]:.2f} '
              f'shift={r["shift_in_sigma_fixed"]:+.2f}')
