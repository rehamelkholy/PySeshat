"""
Orbit-coverage metrics for the benchmark systems.

For each system and each orbit (outer / inner) and data type (visual, RV):
  N          number of measurements
  baseline/P time baseline in units of the orbital period
  gap        largest empty interval in orbital phase (0-1, circular; 1 - gap
             is the phase interval spanned by the data)
  covered    fraction of 20 equal phase bins containing >= 1 measurement
Phases use the literature period only (the metrics are independent of T0 for
`gap`; `covered` uses T0 from the starting elements).  RV for the triples:
Va, Vb = Aa, Ab (inner pair, P1); Vc = B (outer, P2); in .in3 files I1 = inner
positions, I2 = outer; the legacy .in2 convention is reversed (see
compare_orbitt2_orbit3.py).

Output: validation/results/orbit_coverage.csv
"""
import csv
from pathlib import Path

import numpy as np

from pyseshat.io.orbit_parser import parse_orbit_file
from pyseshat.triple.triple_parser import parse_triple_file

ROOT = Path(__file__).resolve().parents[1]
S = ROOT / "validation/benchmark_systems"


def metrics(t, P, T0=0.0, nb=20):
    t = np.asarray(t, float)
    if len(t) == 0:
        return None
    ph = np.sort(((t - T0) / P) % 1.0)
    gaps = np.diff(np.concatenate([ph, [ph[0] + 1.0]]))
    cov = len(set(np.floor(ph * nb).astype(int))) / nb
    return dict(N=len(t), baseline_over_P=float((t.max() - t.min()) / P),
                largest_gap=float(gaps.max()), covered=float(cov))


def rows():
    out = []

    def add(system, orbit, kind, t, P, T0=0.0):
        m = metrics(t, P, T0)
        if m:
            out.append(dict(system=system, orbit=orbit, data=kind, P_yr=P, **m))

    for name, f in (("FIN379", "FIN379/FIN379.inp"),
                    ("GL765", "GL765/GL765.inp")):
        p = parse_orbit_file(str(S / f))
        el = p["elements"]
        add(name, "binary", "visual", p["visual_data"][:, 0], el.P, el.T0)
        add(name, "binary", "RV", p["rv_data"][:, 0], el.P, el.T0)

    # triples: (file, P_inner, P_outer, visual tag -> orbit)
    for name, f, Pin, Pout, tags in (
        ("ZetaAqr", "ZetaAqr/ZetaAqr.in2", 25.95, 540.0, {"I1": "outer", "I2": "inner"}),
        ("HIP111805", "HIP111805/HIP111805.in3", 1.5012, 30.127, {"I1": "inner", "I2": "outer"}),
    ):
        p = parse_triple_file(str(S / f))
        P = {"inner": Pin, "outer": Pout}
        for tag, orb in tags.items():
            add(name, orb, "visual", [o.epoch for o in p["observations"] if o.orbit_id == tag], P[orb])
        if name == "HIP111805":   # configuration of Tokovinin & Latham (2017): A,B speckle era only
            add(name, "outer", "visual >=1978.6",
                [o.epoch for o in p["observations"] if o.orbit_id == "I2" and o.epoch >= 1978.6], P["outer"])
        for tag, orb in (("Va", "inner"), ("Vb", "inner"), ("Vc", "outer")):
            add(name, orb, "RV " + tag, [o.epoch for o in p["rv_rows" if False else "rv_observations"]
                                         if o.orbit_id == tag], P[orb])

    import capella_real_data as cap
    import alpha_cen_real_data as acen
    el = cap.start()
    t, *_ = cap.astrometry(**cap.SUBSETS["B"])
    add("Capella", "binary", "visual", t, el.P, el.T0)
    rv = cap.rv_cfa()
    add("Capella", "binary", "RV", rv[rv[:, 3] == 0][:, 0], el.P, el.T0)
    el = acen.start_elements()
    ep, X, Y, sx, sy, rv = acen.load()
    add("AlphaCenAB", "binary", "visual", ep, el.P, el.T0)
    add("AlphaCenAB", "binary", "RV", rv[rv[:, 3] == 0][:, 0], el.P, el.T0)
    return out


if __name__ == "__main__":
    r = rows()
    path = ROOT / "validation/results/orbit_coverage.csv"
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(r[0]))
        w.writeheader(); w.writerows(r)
    for x in r:
        print(f'{x["system"]:11s}{x["orbit"]:7s}{x["data"]:10s} P={x["P_yr"]:8.3f} N={x["N"]:6d} '
              f'base/P={x["baseline_over_P"]:8.2f} gap={x["largest_gap"]:.3f} cov={x["covered"]:.2f}')
