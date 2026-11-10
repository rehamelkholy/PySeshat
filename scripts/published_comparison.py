"""
PySeshat vs published orbital solutions, with differences in units of the
combined 1-sigma uncertainty: (x_PySeshat - x_pub) / sqrt(s_PySeshat^2 + s_pub^2).

FIN 379: Tokovinin (2016, AJ 152, 10), Tables 3 and 4.
GJ 765.2: Balega et al. (2007, A&A 464, 635), Table 3, refitted from the data
published in the same paper (validation/benchmark_systems/GL765_Balega2007).
"""
import csv
import sys
from pathlib import Path

import numpy as np

from pyseshat.inference.optimizer import fit_visual_orbit
from pyseshat.io.orbit_parser import parse_orbit_file

ROOT = Path(__file__).resolve().parents[1]

PUBLISHED = {
    "FIN379": {
        "file": "validation/benchmark_systems/FIN379/FIN379.inp",
        "ref": "Tokovinin 2016 (AJ 152, 10)",
        "values": {"P": (6.68917, 0.0051), "T0": (2008.846, 0.007),
                   "e": (0.4999, 0.0015), "a": (0.1002, 0.0006),
                   "Omega": (183.77, 0.49), "omega": (191.20, 0.59),
                   "i": (42.00, 0.72), "K1": (7.440, 0.041),
                   "K2": (8.050, 0.041), "V0": (-0.490, 0.040)},
    },
    "GL765_Balega2007": {
        "file": "validation/benchmark_systems/GL765_Balega2007/GL765_Balega2007.inp",
        "ref": "Balega et al. 2007 (A&A 464, 635)",
        "values": {"P": (11.919, 0.003), "T0": (1993.150, 0.004),
                   "e": (0.240, 0.003), "a": (0.189, 0.002),
                   "Omega": (293.0, 0.6), "omega": (250.0, 0.1),
                   "i": (80.2, 0.2), "K1": (7.40, 0.09),
                   "K2": (8.05, 0.08), "V0": (-3.73, 0.05)},
    },
}


def fit(path, weighting="auto"):
    p = parse_orbit_file(str(path))
    vd = p["visual_data"]
    th = np.deg2rad(vd[:, 1])
    X, Y, s = vd[:, 2] * np.cos(th), vd[:, 2] * np.sin(th), vd[:, 3]
    return fit_visual_orbit(p["elements"], vd[:, 0], X, Y, s, s,
                            rv_data=p["rv_data"], weighting=weighting)


def main(out=ROOT / "validation/results/published_comparison.csv"):
    rows = []
    for name, spec in PUBLISHED.items():
        f = fit(ROOT / spec["file"])
        for n, v in zip(f.elements.fitted_parameter_names(),
                        f.elements.fitted_parameters()):
            s = f.uncertainties[n]
            pv, ps = spec["values"][n]
            rows.append({
                "system": name, "reference": spec["ref"], "parameter": n,
                "pyseshat": v, "sigma_pyseshat": s,
                "published": pv, "sigma_published": ps,
                "diff": v - pv,
                "diff_in_combined_sigma": (v - pv) / np.hypot(s, ps),
            })
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    return rows


if __name__ == "__main__":
    for r in main():
        print(f'{r["system"]:18s}{r["parameter"]:7s}{r["pyseshat"]:11.5g}'
              f'±{r["sigma_pyseshat"]:.2g}  {r["published"]:10.5g}'
              f'±{r["sigma_published"]:.2g}  {r["diff_in_combined_sigma"]:+.2f}')
