"""
Real-data Capella benchmark: PySeshat vs Torres, Claret & Young (2009, ApJ 700, 1349).

Data (CDS J/ApJ/700/1349):
  table1 = 162 CfA RVs (A and B), table4 = polar astrometry (Hummel+1994 = ref 44),
  table5 = Cartesian astrometry (Koechlin 51/52, Baldwin 53, Kraus 54),
  table6 = speckle/imaging.
Torres+2009 solved a *global* 24-parameter model (nine historical RV sets with
zero-point offsets, Hipparcos abscissae, a scale factor f_rho for Merrill and
Kulagin).  PySeshat fits only the orbital elements, so the comparison is made on
subsets with no extra nuisance parameters:
  subset "A": CfA RVs + Hummel (Mark III) + Baldwin + Kraus
  subset "B": A + Koechlin + speckle/imaging (both theta and rho present)
Torres's orbit is circular (e = 0), so e = 0 and omega = 0 are held fixed and T0
is the epoch of the zero-omega node passage; T0 is compared modulo P and modulo
P/2 (omega -> omega+180 shifts T0 by P/2 for a circular orbit).
Omega is compared on the same branch of the (Omega+180, omega+180) ambiguity.
Times use the HJD column (the JYr column is too coarse for P = 104 d).
"""
import csv
from pathlib import Path

import numpy as np

from pyseshat.inference.optimizer import fit_visual_orbit
from pyseshat.orbit.orbital_elements import OrbitalElements

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "validation/benchmark_systems/Capella_Torres2009"
PYR = 365.25

if not (DATA / "table1.dat").exists():
    raise SystemExit("Missing Capella_Torres2009/table1.dat: the VizieR tables are not redistributed; see "
                     "validation/benchmark_systems/Capella_Torres2009/README.md for the download instructions.")
HJD0 = 2451545.0


def yr(hjd):
    return 2000.0 + (np.asarray(hjd, float) - HJD0) / PYR


# Torres+2009 Table 9 (value, 1 sigma); a in arcsec, P in years, T in year
PUBLISHED = {
    "P": (104.02173 / PYR, 0.00022 / PYR),
    "T0": (float(yr(2400000 + 47528.514)), 0.016 / PYR),
    "a": (0.056408, 0.000025),
    "i": (137.212, 0.051),
    "Omega": (40.421, 0.064),
    "K1": (26.005, 0.036),
    "K2": (26.260, 0.087),
    "V0": (29.653, 0.035),
}


def _f(s):
    s = s.strip()
    return float(s) if s else np.nan


def read_polar(fn):
    out = []
    for ln in open(DATA / fn):
        if not ln.strip():
            continue
        out.append((float(ln[0:12]), _f(ln[22:28]), _f(ln[29:34]), _f(ln[35:40]),
                    _f(ln[41:45]), int(ln[66:68])))
    return out


def read_cart(fn):
    out = []
    for ln in open(DATA / fn):
        if not ln.strip():
            continue
        out.append((float(ln[0:12]), _f(ln[22:28]), _f(ln[29:35]), _f(ln[36:40]),
                    _f(ln[41:45]), int(ln[65:67])))
    return out


def astrometry(refs_polar, refs_cart):
    """Return t, X(north), Y(east), sx, sy in arcsec."""
    t, X, Y, sx, sy = [], [], [], [], []
    for hjd, th, rho, eth, erho, ref in read_polar("table4.dat") + read_polar("table6.dat"):
        if ref not in refs_polar or np.isnan(th) or np.isnan(rho) or np.isnan(eth) or np.isnan(erho):
            continue
        a = np.deg2rad(th)
        r = rho / 1000.0
        er = erho / 1000.0
        ea = np.deg2rad(eth)
        t.append(yr(hjd)); X.append(r * np.cos(a)); Y.append(r * np.sin(a))
        sx.append(np.hypot(np.cos(a) * er, r * np.sin(a) * ea))
        sy.append(np.hypot(np.sin(a) * er, r * np.cos(a) * ea))
    for hjd, dx, dy, edx, edy, ref in read_cart("table5.dat"):
        if ref not in refs_cart:
            continue
        # DelX = east (rho sin theta), DelY = north (rho cos theta); checked
        # against Hummel positions at equal phase.
        t.append(yr(hjd)); X.append(dy / 1000.0); Y.append(dx / 1000.0)
        sx.append(edy / 1000.0); sy.append(edx / 1000.0)
    o = np.argsort(t)
    return tuple(np.array(v)[o] for v in (t, X, Y, sx, sy))


def rv_cfa():
    rows = []
    for ln in open(DATA / "table1.dat"):
        if ln.strip():
            rows.append((float(ln[0:12]), float(ln[22:28]), float(ln[29:35]),
                         float(ln[36:40]), float(ln[41:45])))
    h, a, b, ea, eb = map(np.array, zip(*rows))
    t = yr(h)
    return np.vstack([np.column_stack([t, a, ea, np.zeros_like(t)]),
                      np.column_stack([t, b, eb, np.ones_like(t)])])


def start():
    el = OrbitalElements(P=PUBLISHED["P"][0], T0=PUBLISHED["T0"][0], e=0.0,
                         a=PUBLISHED["a"][0], Omega=PUBLISHED["Omega"][0],
                         omega=0.0, i=PUBLISHED["i"][0], K1=25.9, K2=26.3, V0=29.6)
    for k in ("e", "omega"):
        el.fit_flags[k] = False
    return el


SUBSETS = {
    "A": dict(refs_polar={44}, refs_cart={53, 54}),
    "B": dict(refs_polar={44, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74,
                          75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87},
              refs_cart={51, 52, 53, 54}),
}


def main(out=ROOT / "validation/results/capella_torres2009.csv"):
    rv = rv_cfa()
    rows, info = [], {}
    for sub, kw in SUBSETS.items():
        t, X, Y, sx, sy = astrometry(**kw)
        for w in ("fixed", "dataset"):
            f = fit_visual_orbit(start(), t, X, Y, sx, sy, rv_data=rv, weighting=w)
            key = f"{sub}/{w}"
            info[key] = {"n_astrom": len(t),
                         "diag": f.metadata["dataset_diagnostics"],
                         "scales": f.metadata["error_scales"]}
            el = f.elements
            flipped = abs(((el.Omega - PUBLISHED["Omega"][0] + 180) % 360) - 180) > 90
            P = el.P
            dT = (el.T0 - PUBLISHED["T0"][0] + P / 4) % (P / 2) - P / 4  # mod P/2
            vals = {n: v for n, v in zip(el.fitted_parameter_names(), el.fitted_parameters())}
            for n, (pv, ps) in PUBLISHED.items():
                v = vals[n]
                s_ = f.uncertainties[n]
                if n == "T0":
                    d = dT
                    v = PUBLISHED["T0"][0] + dT
                else:
                    if flipped and n == "Omega":
                        pv = (pv + 180.0) % 360.0
                    d = v - pv
                rows.append({"subset": sub, "weighting": w, "parameter": n,
                             "pyseshat": v, "sigma_pyseshat": s_, "published": pv,
                             "sigma_published": ps,
                             "diff_in_combined_sigma": d / np.hypot(s_, ps),
                             "node_branch_flipped": flipped})
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0]))
        wr.writeheader(); wr.writerows(rows)
    return rows, info


if __name__ == "__main__":
    rows, info = main()
    for k, v in info.items():
        d = {a: round(b["chi2_per_dof"], 2) for a, b in v["diag"].items()}
        print(k, "N_astrom", v["n_astrom"], d,
              {a: round(float(b), 2) for a, b in v["scales"].items()})
    for r in rows:
        print(f'{r["subset"]}/{r["weighting"]:8s}{r["parameter"]:6s}{r["pyseshat"]:12.5f}'
              f'±{r["sigma_pyseshat"]:.5f}  pub {r["published"]:12.5f}±{r["sigma_published"]:.5f}'
              f'  {r["diff_in_combined_sigma"]:+.2f}  flip={r["node_branch_flipped"]}')
