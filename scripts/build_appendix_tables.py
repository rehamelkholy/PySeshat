"""
Build the appendix PySeshat-vs-published comparison table.

Inputs (all produced by the scripts in this folder, none edited by hand):
  validation/results/published_comparison.csv   FIN379, GL765 (Balega refit)
  validation/results/alpha_cen_akeson2021.csv   alpha Cen AB
  validation/results/capella_torres2009.csv     Capella (subset B)
  validation/results/ZetaAqr.json               zeta Aqr (ORBIT3 path)
Published zeta Aqr values and 1-sigma: Tokovinin (2016, ApJ 831, 151) Table 5.
Outputs: validation/results/appendix_published_comparison.csv and .tex
"""
import csv
import json
from pathlib import Path

import numpy as np

R = Path(__file__).resolve().parents[1] / "validation/results"

# Tokovinin 2016 (ApJ 831, 151) Table 5.  Index 1 = outer A,B (PySeshat suffix "2");
# index 2 = inner Aa,Ab (PySeshat no suffix).  a(inner) is fixed in the published fit;
# F = 3.50 +- 0.09 corresponds to PySeshat f = 1/F.
ZETA = {
    "P2": (540.0, 15.0), "T02": (1981.50, 0.58), "e2": (0.419, 0.011),
    "a2": (3.496, 0.046), "Omega2": (131.3, 0.8), "omega2": (269.3, 1.7),
    "i2": (142.0, 0.4),
    "P": (25.95, 0.048), "T0": (2006.52, 0.13), "e": (0.872, 0.006),
    "Omega": (293.7, 74.0), "omega": (100.9, 73.0), "i": (11.8, 6.7),
    "f": (1 / 3.50, 0.09 / 3.50**2),
}
# Tokovinin & Latham (2017, ApJ 838, 54) Table 2, HIP 111805 (inner Ba,Bb = no suffix;
# outer B,A = suffix 2; K3, K4 = K1, K2 of the outer orbit). Compared in the configuration
# of their last iteration: P2 fixed, A,B positions >= 1978.6 (scripts/hip111805_paper_config.py).
HIP = {
    "P": (1.50119, 0.0004), "T0": (1986.093, 0.093), "e": (0.022, 0.011), "a": (0.0385, 0.0010),
    "Omega": (334.5, 1.0), "omega": (232.9, 22.3), "i": (85.8, 1.6), "K1": (13.13, 0.25),
    "K2": (19.21, 3.1), "f": (-0.33, 0.015),
    "T02": (2010.179, 0.073), "e2": (0.324, 0.004), "a2": (0.3361, 0.0015),
    "Omega2": (154.25, 0.09), "omega2": (84.92, 0.18), "i2": (88.28, 0.10), "K3": (6.057, 0.14),
    "K4": (8.6, 0.23), "V0": (-22.575, 0.08),
}
HIP_LABEL = {"P": "P_in", "T0": "T0_in", "e": "e_in", "a": "a_in", "Omega": "Omega_in",
             "omega": "omega_in", "i": "i_in", "K1": "K1_in", "K2": "K2_in", "f": "f",
             "T02": "T0_out", "e2": "e_out", "a2": "a_out", "Omega2": "Omega_out",
             "omega2": "omega_out", "i2": "i_out", "K3": "K1_out", "K4": "K2_out", "V0": "V0"}
LABEL = {"P2": "P_out", "T02": "T0_out", "e2": "e_out", "a2": "a_out",
         "Omega2": "Omega_out", "omega2": "omega_out", "i2": "i_out",
         "P": "P_in", "T0": "T0_in", "e": "e_in", "Omega": "Omega_in",
         "omega": "omega_in", "i": "i_in", "f": "f=1/F"}


def rows():
    out = []
    for r in csv.DictReader(open(R / "published_comparison.csv")):
        out.append((r["system"], r["reference"], r["parameter"],
                    float(r["pyseshat"]), float(r["sigma_pyseshat"]),
                    float(r["published"]), float(r["sigma_published"])))
    for r in csv.DictReader(open(R / "alpha_cen_akeson2021.csv")):
        if r["weighting"] == "dataset":
            out.append(("AlphaCenAB", "Akeson et al. 2021 (AJ 162, 14)", r["parameter"],
                        float(r["pyseshat"]), float(r["sigma_pyseshat"]),
                        float(r["published"]), float(r["sigma_published"])))
    for r in csv.DictReader(open(R / "capella_torres2009.csv")):
        if r["subset"] == "B" and r["weighting"] == "dataset":
            out.append(("Capella", "Torres et al. 2009 (ApJ 700, 1349)", r["parameter"],
                        float(r["pyseshat"]), float(r["sigma_pyseshat"]),
                        float(r["published"]), float(r["sigma_published"])))
    z = json.load(open(R / "ZetaAqr.json"))
    for k, (pv, ps) in ZETA.items():
        s = z["parameter_uncertainties"].get(k)
        if s is None:
            continue
        out.append(("ZetaAqr", "Tokovinin 2016 (ApJ 831, 151)", LABEL[k],
                    z["fitted_elements"][k], s, pv, ps))
    pc = json.load(open(R / "hip111805_paper_config.json"))
    run = [r for r in pc["runs"] if r.get("start") == "published" and r.get("method") == "trf"][0]
    for k, (pv, ps) in HIP.items():
        out.append(("HIP111805", "Tokovinin & Latham 2017 (ApJ 838, 54)", HIP_LABEL[k],
                    run["values"][k], run["sigma"][k], pv, ps))
    return out


def main():
    data = rows()
    with open(R / "appendix_published_comparison.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["system", "reference", "parameter", "pyseshat", "sigma_pyseshat",
                    "published", "sigma_published", "diff_in_combined_sigma",
                    "sigma_ratio", "within_1sigma"])
        summary = {}
        for s, ref, p, v, sv, pv, ps in data:
            d = (v - pv) / np.hypot(sv, ps)
            w.writerow([s, ref, p, v, sv, pv, ps, d, sv / ps, abs(d) <= 1.0])
            summary.setdefault(s, []).append(abs(d))
    for s, v in summary.items():
        print(f"{s:12s} n={len(v):2d} max|d|={max(v):.2f} within1sigma={sum(x <= 1 for x in v)}/{len(v)}")
    return data


if __name__ == "__main__":
    main()


# ----------------------------------------------------------------------------
# LaTeX output (appendix tables of the manuscript)
# ----------------------------------------------------------------------------
import math

SYS_TITLE = {
    "FIN379": r"FIN379 \citep{Tokovinin2016a}",
    "GL765_Balega2007": r"GL765, refit of the data of \citet{Balega2007}",
    "AlphaCenAB": r"$\alpha$ Cen AB \citep{Akeson2021}",
    "Capella": r"Capella, subset B \citep{Torres2009}",
    "ZetaAqr": r"$\zeta$ Aqr \citep{Tokovinin2016b}",
    "HIP111805": r"HIP\,111805 \citep{Tokovinin2017}, $P_{\rm out}$ fixed, A,B positions since 1978.6",
}
PAR_TEX = {
    "P": "$P$", "T0": "$T_0$", "e": "$e$", "a": "$a$", "Omega": r"$\Omega$",
    "omega": r"$\omega$", "i": "$i$", "K1": "$K_1$", "K2": "$K_2$", "V0": "$V_0$",
    "f": "$f$", "f=1/F": "$f$",
}


def par_tex(name):
    if name in PAR_TEX:
        return PAR_TEX[name]
    base, _, sub = name.partition("_")
    if len(base) > 1 and base[-1].isdigit():
        return "$%s_{%s,\\rm %s}$" % (base[:-1], base[-1], sub)
    b = PAR_TEX.get(base, base).strip("$")
    return "$%s_{\\rm %s}$" % (b, sub)


def num(v, s):
    d = 0 if s <= 0 else max(0, 1 - math.floor(math.log10(s)))
    return f"${v:.{d}f} \\pm {s:.{d}f}$"


def latex_published(red=False):
    data = rows()
    cols = r">{\color{red}}l>{\color{red}}c>{\color{red}}c>{\color{red}}r>{\color{red}}r" if red else "lccrr"
    lines = [r"\begin{longtable}{%s}" % cols,
             (r"\caption{\color{red}" if red else r"\caption{") + r"\label{tab:app-pub}Orbital elements fitted with PySeshat compared with the published solutions. "
             r"Uncertainties are $1\sigma$. $\Delta/\sigma$ is the difference divided by the quadrature sum of the two "
             r"uncertainties and $\sigma_{\rm PS}/\sigma_{\rm pub}$ the ratio of uncertainties; entries with $|\Delta/\sigma|>1$ are in bold. "
             r"Periods and epochs are in years (Capella: $P$ in days), $a$ in arcsec, angles in degrees, velocities in km\,s$^{-1}$. "
             r"For the triples, the suffixes `in' and `out' denote the inner and outer orbits; $f$ is the ORBIT3 wobble factor. "
             r"$\Omega$ and $\omega$ of $\alpha$ Cen AB are compared on the same branch of the $(\Omega+180^\circ,\omega+180^\circ)$ ambiguity.}\\",
             r"\hline\hline", r"Parameter & PySeshat & Published & $\Delta/\sigma$ & $\sigma_{\rm PS}/\sigma_{\rm pub}$\\", r"\hline",
             r"\endfirsthead", r"\hline", r"Parameter & PySeshat & Published & $\Delta/\sigma$ & $\sigma_{\rm PS}/\sigma_{\rm pub}$\\", r"\hline",
             r"\endhead", r"\hline", r"\endfoot"]
    cur = None
    for s, ref, p, v, sv, pv, ps in data:
        if s != cur:
            lines.append(r"\multicolumn{5}{%s}{\textit{%s}}\\" % (r">{\color{red}}l" if red else "l", SYS_TITLE[s]))
            cur = s
        if s == "Capella" and p == "P":
            v, sv, pv, ps = (x * 365.25 for x in (v, sv, pv, ps))
        d = (v - pv) / math.hypot(sv, ps)
        ds = f"{d:+.2f}"
        if abs(d) > 1:
            ds = r"\textbf{%s}" % ds
        lines.append(f"{par_tex(p)} & {num(v, sv)} & {num(pv, ps)} & {ds} & {sv / ps:.2f}\\\\")
    # no page break after a system title row, and none within the last three rows (no orphan row)
    for k, ln in enumerate(lines):
        if ln.startswith(r"\multicolumn{5}"):
            lines[k] = ln[:-2] + r"\\*"
    last = len(lines) - 1
    for k in (last - 2, last - 1):
        lines[k] = lines[k][:-2] + r"\\*"
    lines += [r"\end{longtable}"]
    return "\n".join(lines)


DIAG_LOG = {}


def _diag_rows():
    """(system, dataset, N, chi2nu_fixed, scale, chi2nu_after, maxshift, ratio_lo, ratio_hi).
    Reuses validation/results/weighting_diagnostics.json when present (the binary fits do not
    depend on the ORBIT3 code); set PYSESHAT_REFIT=1 to redo the alpha Cen / Capella refits."""
    import os
    cache = R / "weighting_diagnostics.json"
    if cache.exists() and os.environ.get("PYSESHAT_REFIT") != "1":
        c = json.load(open(cache))
        DIAG_LOG.update(c.get("refits", {}))
        return [tuple(r) for r in c["binary_rows"]]
    return _diag_rows_refit()


def _diag_rows_refit():
    out = []
    ws = json.load(open(R / "weighting_sensitivity.json"))
    cs = list(csv.DictReader(open(R / "weighting_sensitivity.csv")))
    for sysname, label in (("FIN379", "FIN379"), ("GL765", "GL765")):
        d = ws[sysname]
        rr = [r for r in cs if r["system"] == sysname]
        sh = max(abs(float(r["shift_in_sigma_fixed"])) for r in rr)
        lo = min(float(r["sigma_ratio"]) for r in rr)
        hi = max(float(r["sigma_ratio"]) for r in rr)
        for k, nm in (("visual", "astrometry"), ("rv", "RV")):
            a = d["diagnostics_fixed"][k]
            b = d["diagnostics_dataset"][k]
            out.append((label, nm, a["n"], a["chi2_per_dof"], d["error_scales"][k], b["chi2_per_dof"], sh, lo, hi))
    # real-data Capella (subset B) and alpha Cen: refit here
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import alpha_cen_real_data as ac
    import capella_real_data as cp
    from pyseshat.inference.optimizer import fit_visual_orbit
    jobs = []
    ep, X, Y, sx, sy, rv = ac.load()
    jobs.append(("$\\alpha$ Cen AB", lambda w: fit_visual_orbit(ac.start_elements(), ep, X, Y, sx, sy, rv_data=rv, weighting=w)))
    t, X2, Y2, sx2, sy2 = cp.astrometry(**cp.SUBSETS["B"])
    rv2 = cp.rv_cfa()
    jobs.append(("Capella", lambda w: fit_visual_orbit(cp.start(), t, X2, Y2, sx2, sy2, rv_data=rv2, weighting=w)))
    for label, fit in jobs:
        a, b = fit("fixed"), fit("dataset")
        names = a.elements.fitted_parameter_names()
        va, vb = np.array(a.elements.fitted_parameters()), np.array(b.elements.fitted_parameters())
        ua = np.array([a.uncertainties[n] for n in names])
        ub = np.array([b.uncertainties[n] for n in names])
        # ignore the angle branch ambiguity: compare modulo 180 for Omega/omega
        dv = vb - va
        for j, n in enumerate(names):
            if n in ("Omega", "omega"):
                dv[j] = (dv[j] + 90) % 180 - 90
            if n == "T0" and label == "Capella":
                dv[j] = (dv[j] + a.elements.P / 4) % (a.elements.P / 2) - a.elements.P / 4
        ratio = ub / ua
        sh = float(np.max(np.abs(dv) / ua))
        DIAG_LOG[label] = {"scales": {k: float(v) for k, v in b.metadata["error_scales"].items()},
                           "diag_fixed": a.metadata["dataset_diagnostics"],
                           "diag_dataset": b.metadata["dataset_diagnostics"],
                           "shift_in_sigma_fixed": dict(zip(names, (dv / ua).tolist())),
                           "sigma_ratio": dict(zip(names, ratio.tolist()))}
        for k, nm in (("visual", "astrometry"), ("rv", "RV")):
            da, db = a.metadata["dataset_diagnostics"][k], b.metadata["dataset_diagnostics"][k]
            out.append((label, nm, da["n"], da["chi2_per_dof"], b.metadata["error_scales"][k], db["chi2_per_dof"],
                        sh, float(ratio.min()), float(ratio.max())))
    return out


COV_KEY = {  # (coverage system, orbit, data) for each weighting row (system label, dataset)
    ("FIN379", "astrometry"): ("FIN379", "binary", "visual"), ("FIN379", "RV"): ("FIN379", "binary", "RV"),
    ("GL765", "astrometry"): ("GL765", "binary", "visual"), ("GL765", "RV"): ("GL765", "binary", "RV"),
    ("$\\alpha$ Cen AB", "astrometry"): ("AlphaCenAB", "binary", "visual"),
    ("$\\alpha$ Cen AB", "RV"): ("AlphaCenAB", "binary", "RV"),
    ("Capella", "astrometry"): ("Capella", "binary", "visual"), ("Capella", "RV"): ("Capella", "binary", "RV"),
}


def _coverage():
    return {(r["system"], r["orbit"], r["data"]): r for r in csv.DictReader(open(R / "orbit_coverage.csv"))}


def latex_main(red=False):
    """Main-text Table 2 (tab:orb-benchmark): coverage, weighting and agreement for the six orbit systems."""
    cov = _coverage()
    summ = summary_cells()
    tri = json.load(open(R / "triple_dataset_chi2.json"))   # from scripts/triple_dataset_chi2.py
    diag = _diag_rows()
    json.dump({"binary_rows": diag, "refits": DIAG_LOG}, open(R / "weighting_diagnostics.json", "w"), indent=2)
    tri = dict(tri, HIP111805=tri["HIP111805_paper"])
    L = [r"\begin{table*}"] + ([r"\color{red}"] if red else []) + [
         r"\caption{\label{tab:orb-benchmark}Orbit-fitting benchmarks: data coverage, dataset weighting and agreement with the published solutions. "
         r"Epochs is the number of measurement epochs (for $\alpha$ Cen AB the RVs are nightly bins), $N$ the number of residuals, "
         r"$\Delta t/P$ the time baseline in orbital periods and Gap the largest empty interval of orbital phase. "
         r"$\chi^2_\nu$ is the reduced $\chi^2$ of each dataset with the quoted errors and $s_k$ the converged error scale factor of the "
         r"dataset-level weighting (Sect.~\ref{subsec:orb-fit}; errors are only inflated, so $s_k\ge1$). "
         r"The last two columns give the number of fitted parameters that agree with the published solution within $1\sigma$ "
         r"and the largest $|\Delta|/\sigma$ (Table~\ref{tab:app-pub}).}",
         r"\centering\small", r"\begin{tabular}{llrrrrrrcc}", r"\hline\hline",
         r"System & Data & Epochs & $N$ & $\Delta t/P$ & Gap & $\chi^2_\nu$ & $s_k$ & $\le1\sigma$ & $\max|\Delta|/\sigma$\\", r"\hline"]
    key = {"FIN379": "FIN379", "GL765": "GL765_Balega2007", "$\\alpha$ Cen AB": "AlphaCenAB", "Capella": "Capella"}
    tex = {"FIN379": "FIN379", "GL765": r"GL765\tablefootmark{a}", "$\\alpha$ Cen AB": r"$\alpha$ Cen AB", "Capella": "Capella"}
    prev = None
    for sysn, ds, n, c0, sk, c1, sh, lo, hi in diag:
        c = cov[COV_KEY[(sysn, ds)]]
        first = sysn != prev
        w = ""
        if first:
            k, t, m = summ[key[sysn]]
            w = f"{k}/{t} & {m:.2f}"
        else:
            w = " & "
        L.append(f"{tex[sysn] if first else ''} & {ds} & {c['N']} & {n} & {float(c['baseline_over_P']):.2f} & "
                 f"{float(c['largest_gap']):.2f} & {c0:.2f} & {sk:.2f} & {w}\\\\")
        prev = sysn
    tri_rows = {  # coverage key -> triple_dataset_chi2 block
        ("ZetaAqr", "outer", "visual"): "outer astrometry", ("ZetaAqr", "inner", "visual"): "inner astrometry",
        ("HIP111805", "inner", "visual"): "inner astrometry", ("HIP111805", "outer", "visual >=1978.6"): "outer astrometry",
        ("HIP111805", "inner", "RV Va"): "RV Va", ("HIP111805", "inner", "RV Vb"): "RV Vb",
        ("HIP111805", "outer", "RV Vc"): "RV Vc"}
    names = {"ZetaAqr": r"$\zeta$ Aqr\tablefootmark{b}", "HIP111805": r"HIP\,111805\tablefootmark{b,c}"}
    prev = None
    for ck, blk in tri_rows.items():
        sysn = ck[0]
        c, d = cov[ck], tri[sysn]["datasets"][blk]
        first = sysn != prev
        lab = f"{ck[1]} {ck[2].replace(' >=1978.6', '').replace('visual', 'astrometry')}".replace("RV Va", "RV Aa").replace("RV Vb", "RV Ab").replace("RV Vc", "RV B")
        if first:
            k, t, m = summ[sysn]
            w = f"{k}/{t} & {m:.2f}"
        else:
            w = " & "
        L.append(f"{names[sysn] if first else ''} & {lab} & {c['N']} & {d['n']} & {float(c['baseline_over_P']):.2f} & "
                 f"{float(c['largest_gap']):.2f} & {d['chi2_per_dof']:.2f} & -- & {w}\\\\")
        prev = sysn
    L += [r"\hline", r"\end{tabular}",
          r"\tablefoot{\tablefoottext{a}{Coverage and weighting refer to the data of the ORBIT example file; the comparison refers to the refit of the data of \citet{Balega2007}.} "
          r"\tablefoottext{b}{ORBIT3 fits use the quoted errors without rescaling.} "
          r"\tablefoottext{c}{Configuration of the published solution \citep{Tokovinin2017}: outer period fixed and only the A,B positions since 1978.6 (speckle era), 19 free elements.}}",
          r"\end{table*}"]
    return "\n".join(L)


def summary_cells():
    """Per-system 'within 1 sigma' count and max |Delta|/sigma for the main-text Table 2."""
    from collections import defaultdict
    acc = defaultdict(list)
    for r in csv.DictReader(open(R / "appendix_published_comparison.csv")):
        acc[r["system"]].append(abs(float(r["diff_in_combined_sigma"])))
    return {k: (sum(x <= 1 for x in v), len(v), max(v)) for k, v in acc.items()}


def write_tex(path=R / "appendix_tables.tex", main_path=R / "table2_orb_benchmark.tex", red=True):
    hdr = "% Generated by scripts/build_appendix_tables.py -- do not edit by hand\n"
    Path(path).write_text(hdr + latex_published(red) + "\n")
    Path(main_path).write_text(hdr + latex_main(red) + "\n")
    for k, (n, t, m) in summary_cells().items():
        print(f"{k:18s} {n}/{t} within 1 sigma, max |d| = {m:.2f}")
    for lab, d in DIAG_LOG.items():
        sh = d["shift_in_sigma_fixed"]
        km = max(sh, key=lambda q: abs(sh[q]))
        print(f"{lab}: scales {d['scales']}, chi2nu fixed "
              f"{ {k: round(v['chi2_per_dof'], 2) for k, v in d['diag_fixed'].items()} }, max shift {sh[km]:+.2f} ({km})")
    return path, main_path


if __name__ == "__main__":
    print("wrote", write_tex())
