"""
Real-data alpha Cen AB benchmark: PySeshat vs Akeson et al. (2021, AJ 162, 14).

Data (CDS J/AJ/162/14): table6 = 106 relative positions of B w.r.t. A
(1940-2020), table7 = 17567 HARPS velocities (5184 A, 12383 B).
Cartesian errors are propagated from the quoted (sigma_theta, sigma_rho).
Published elements: Akeson+2021 Table 8 (reference epoch 2019.5).
Velocities are averaged per star and night (weighted mean). A constant A-B
offset was tested and is consistent with zero in this table, so none is applied.
Omega/omega are compared on the same branch of the (Omega+180, omega+180)
ambiguity as the fit.
"""
import json
import sys
from pathlib import Path

import numpy as np

from pyseshat.inference.optimizer import fit_visual_orbit
from pyseshat.orbit.orbital_elements import OrbitalElements

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "validation/benchmark_systems/AlphaCenAB_Akeson2021"

if not (DATA / "table6.dat").exists():
    raise SystemExit("Missing AlphaCenAB_Akeson2021/table6.dat: the VizieR tables are not redistributed; see "
                     "validation/benchmark_systems/AlphaCenAB_Akeson2021/README.md for the download instructions.")

# Akeson+2021 Table 8: value, 1-sigma
PUBLISHED = {
    "P": (79.762, 0.019), "T0": (1955.564, 0.015), "e": (0.51947, 0.00015),
    "a": (17.4930, 0.0096), "Omega": (205.073, 0.025),
    "omega": (231.519, 0.027), "i": (79.2430, 0.0089), "V0": (-22.3796, 0.0020),
}


def load():
    t6 = np.loadtxt(DATA / "table6.dat", usecols=(0, 1, 2, 3, 4))
    ep, pa, epa, sep, esep = t6.T
    th = np.deg2rad(pa)
    eth = np.deg2rad(epa)
    X, Y = sep * np.cos(th), sep * np.sin(th)
    sx = np.hypot(np.cos(th) * esep, sep * np.sin(th) * eth)
    sy = np.hypot(np.sin(th) * esep, sep * np.cos(th) * eth)
    rows = []
    for line in open(DATA / "table7.dat"):
        if line.strip():
            rows.append((float(line[0:11]), line[13], float(line[15:25]),
                         float(line[29:36])))
    mjd = np.array([r[0] for r in rows])
    comp = np.array([0.0 if r[1] == "A" else 1.0 for r in rows])
    rv = np.array([r[2] for r in rows])
    er = np.array([r[3] for r in rows])
    # nightly weighted means per star (several exposures per night)
    night = np.floor(mjd + 0.5).astype(int)
    keys = sorted(set(zip(night, comp)))
    tb, rb, eb, cb = [], [], [], []
    for k in keys:
        m = (night == k[0]) & (comp == k[1])
        w = 1.0 / er[m] ** 2
        tb.append(np.average(mjd[m], weights=w))
        rb.append(np.average(rv[m], weights=w))
        eb.append(1.0 / np.sqrt(w.sum()))
        cb.append(k[1])
    mjd, rv, er, comp = map(np.array, (tb, rb, eb, cb))
    year = 2000.0 + (mjd + 2400000.5 - 2451545.0) / 365.25
    rv_data = np.column_stack([year, rv, er, comp])
    return ep, X, Y, sx, sy, rv_data


def start_elements():
    el = OrbitalElements(P=79.762, T0=1955.564, e=0.51947, a=17.4930,
                         Omega=205.073, omega=231.519, i=79.2430,
                         K1=4.58, K2=5.43, V0=-22.3796)
    return el


def main(out=ROOT / "validation/results/alpha_cen_akeson2021.csv"):
    import csv
    ep, X, Y, sx, sy, rv = load()
    rows, info = [], {}
    for w in ("fixed", "dataset"):
        f = fit_visual_orbit(start_elements(), ep, X, Y, sx, sy, rv_data=rv,
                             weighting=w)
        info[w] = {"diag": f.metadata["dataset_diagnostics"],
                   "scales": f.metadata["error_scales"]}
        flipped = abs(((f.elements.Omega - PUBLISHED["Omega"][0] + 180) % 360) - 180) > 90
        for n, v in zip(f.elements.fitted_parameter_names(),
                        f.elements.fitted_parameters()):
            if n not in PUBLISHED:
                continue
            s_ = f.uncertainties[n]
            pv, ps = PUBLISHED[n]
            if flipped and n in ("Omega", "omega"):
                pv = (pv + 180.0) % 360.0
            rows.append({"weighting": w, "parameter": n, "pyseshat": v,
                         "sigma_pyseshat": s_, "published": pv,
                         "sigma_published": ps,
                         "diff_in_combined_sigma": (v - pv) / np.hypot(s_, ps),
                         "node_branch_flipped": flipped})
        info[w]["K1"] = float(f.elements.K1)
        info[w]["K2"] = float(f.elements.K2)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    return rows, info


if __name__ == "__main__":
    rows, info = main()
    for w in info:
        d = {k: round(v["chi2_per_dof"], 2) for k, v in info[w]["diag"].items()}
        print(w, d, "scales", {k: round(float(v), 2) for k, v in info[w]["scales"].items()},
              "K1,K2 =", round(info[w]["K1"], 3), round(info[w]["K2"], 3))
    for r in rows:
        print(f'{r["weighting"]:8s}{r["parameter"]:7s}{r["pyseshat"]:12.5f}±{r["sigma_pyseshat"]:.5f}'
              f'  pub {r["published"]:12.5f}±{r["sigma_published"]:.5f}'
              f'  {r["diff_in_combined_sigma"]:+.2f}')
