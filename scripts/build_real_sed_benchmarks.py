"""
Build the real-data SED benchmark files for Capella and alpha Cen AB
(validation/expected_results/Capella_photometry.json and
AlphaCenAB_photometry.json), linking the SED fit to the orbit fit through
the distance.

Distance: the orbital parallax from PySeshat's own fit of the real orbit
data (dataset weighting; Capella: subset B of scripts/capella_real_data.py,
Torres et al. 2009 data; alpha Cen: scripts/alpha_cen_real_data.py,
Akeson et al. 2021 data),

    pi_orb = a'' / a[au],   a[au] sin i = (K1 + K2) P sqrt(1 - e^2) / (2 pi),

with its uncertainty propagated through the fit covariance matrix.

Photometry (catalogue files in validation/benchmark_systems/SED_catalogs/,
VizieR downloads, not redistributed; see the README there):
  * Mermilliod (1991) UBV means, VizieR II/168/ubvmeans: Capella (combined
    light) and alpha Cen A and B (separately).
  * Hauck & Mermilliod (1998) uvby, VizieR II/215: Capella only; recorded
    under "excluded_measurements" (see the note there).
  * Capella component flux ratios l_cool/l_hot in B and V: Torres et al.
    (2009, ApJ, 700, 1349), Table 13.
Uncertainties: II/168 standard deviations where given (Capella). For
alpha Cen, II/168 gives none (two observations, one publication); the
typical single-value uncertainties of Maiz Apellaniz (2006, AJ, 131, 1184,
Tables 9 and 10), 0.020 mag in B-V and 0.028 mag in U-B, are used, with
an assumed 0.02 mag in V.

Writes the two expected_results files and
validation/results/real_sed_orbital_parallax.json. Needs the VizieR orbit
tables of both benchmarks and the catalogue files.
"""

import csv
import io
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

CATALOGS = ROOT / "validation" / "benchmark_systems" / "SED_catalogs"
EXPECTED = ROOT / "validation" / "expected_results"
RESULTS = ROOT / "validation" / "results"

AU_KM = 1.495978707e8
DAY_S = 86400.0
YEAR_D = 365.25
LN10_04 = 2.5 / math.log(10.0)  # 1.0857: d(mag) per unit relative flux

ALPHA_CEN_V_SIGMA = 0.02          # assumed
MAIZ2006_SIGMA = {"B-V": 0.020, "U-B": 0.028}   # Maiz Apellaniz 2006, Tables 9-10


# ----------------------------------------------------------------------
# Orbital parallax
# ----------------------------------------------------------------------

def orbital_parallax_mas(p):
    """p: dict with P (yr), a (arcsec), e, i (deg), K1, K2 (km/s)."""
    a_au_sini = ((p["K1"] + p["K2"]) * p["P"] * YEAR_D * DAY_S
                 * math.sqrt(1.0 - p["e"] ** 2) / (2.0 * math.pi) / AU_KM)
    return 1000.0 * p["a"] * math.sin(math.radians(p["i"])) / a_au_sini


def parallax_with_error(fit):
    el = fit.elements
    names = list(el.fitted_parameter_names())
    values = dict(zip(names, el.fitted_parameters()))
    base = {k: float(getattr(el, k)) for k in ("P", "a", "e", "i", "K1", "K2")}
    pi0 = orbital_parallax_mas(base)
    cov = np.asarray(fit.covariance, dtype=float)
    grad = np.zeros(len(names))
    for j, n in enumerate(names):
        if n not in base:
            continue
        h = 1e-6 * max(abs(values[n]), 1e-3)
        up = dict(base, **{n: base[n] + h})
        dn = dict(base, **{n: base[n] - h})
        grad[j] = (orbital_parallax_mas(up) - orbital_parallax_mas(dn)) / (2 * h)
    sigma = float(math.sqrt(grad @ cov @ grad))
    return pi0, sigma, base


def capella_orbit():
    import capella_real_data as cr
    t, X, Y, sx, sy = cr.astrometry(**cr.SUBSETS["B"])
    from pyseshat.inference.optimizer import fit_visual_orbit
    return fit_visual_orbit(cr.start(), t, X, Y, sx, sy, rv_data=cr.rv_cfa(),
                            weighting="dataset")


def alpha_cen_orbit():
    import alpha_cen_real_data as ac
    ep, X, Y, sx, sy, rv = ac.load()
    from pyseshat.inference.optimizer import fit_visual_orbit
    return fit_visual_orbit(ac.start_elements(), ep, X, Y, sx, sy, rv_data=rv,
                            weighting="dataset")


# ----------------------------------------------------------------------
# Catalogue files (VizieR tab-separated output)
# ----------------------------------------------------------------------

def read_vizier_tsv(path):
    lines = [ln for ln in Path(path).read_text(encoding="utf-8").splitlines()
             if ln.strip() and not ln.startswith("#")]
    header, rows = lines[0].split("\t"), []
    for ln in lines[3:]:          # skip units and dashes lines
        rows.append({k.strip(): v.strip() for k, v in zip(header, ln.split("\t"))})
    return rows


def ubv_row(rows, simbad_name):
    match = [r for r in rows if r["SimbadName"].strip() == simbad_name]
    if len(match) != 1:
        raise SystemExit(f"{simbad_name}: expected one II/168 row, found {len(match)}")
    return match[0]


def _num(s):
    return float(s) if s.strip() else None


def combine(mags):
    return -2.5 * math.log10(sum(10 ** (-0.4 * m) for m in mags))


# ----------------------------------------------------------------------
# Builders
# ----------------------------------------------------------------------

def capella_photometry():
    r = ubv_row(read_vizier_tsv(CATALOGS / "HD34029_II168_ubvmeans.tsv"), "HD 034029")
    V, BV, UB = _num(r["Vmag"]), _num(r["B-V"]), _num(r["U-B"])
    eV, eBV, eUB = _num(r["e_Vmag"]), _num(r["e_B-V"]), _num(r["e_U-B"])
    meas = [
        {"passband": "Johnson U", "magnitude": round(V + BV + UB, 4),
         "uncertainty": round(math.sqrt(eV ** 2 + eBV ** 2 + eUB ** 2), 4)},
        {"passband": "Johnson B", "magnitude": round(V + BV, 4),
         "uncertainty": round(math.hypot(eV, eBV), 4)},
        {"passband": "Johnson V", "magnitude": V, "uncertainty": eV},
    ]
    # Torres et al. (2009) Table 13: l_cool/l_hot (= l_A/l_B). Primary = A
    # (cooler, more massive), so m_secondary - m_primary = 2.5 log(l_A/l_B).
    ratios = {"Johnson B": (0.715, 0.058), "Johnson V": (0.888, 0.011)}
    dms = [{"passband": b, "value": round(2.5 * math.log10(l), 4),
            "uncertainty": round(LN10_04 * s / l, 4)} for b, (l, s) in ratios.items()]

    s = read_vizier_tsv(CATALOGS / "HD34029_II215.tsv")
    if len(s) != 1:
        raise SystemExit("HD34029_II215.tsv: expected one row")
    s = s[0]
    y, by, m1, c1 = (_num(s[k]) for k in ("Vmag", "b-y", "m1", "c1"))
    b = y + by
    v = b + m1 + by
    u = v + c1 + (v - b)
    excluded = [{"passband": f"Stromgren {k}", "magnitude": round(val, 4),
                 "note": ("Hauck & Mermilliod (1998), VizieR II/215, one source, no errors; "
                          "y taken as the catalogue V. Excluded: at the published parameters "
                          "all four bands sit about 0.06 mag fainter than the II/168 UBV "
                          "photometry (catalogue V 0.100 vs 0.073), which UBV alone does not "
                          "show.")}
                for k, val in (("u", u), ("v", v), ("b", b), ("y", y))]
    return meas, dms, excluded


def alpha_cen_photometry():
    rows = read_vizier_tsv(CATALOGS / "HD128620_II168_ubvmeans.tsv")
    comp = {}
    for label, name in (("A", "HD 128620"), ("B", "HD 128621")):
        r = ubv_row(rows, name)
        V, BV, UB = _num(r["Vmag"]), _num(r["B-V"]), _num(r["U-B"])
        comp[label] = {"Johnson V": V, "Johnson B": V + BV, "Johnson U": V + BV + UB}
    sig = {"Johnson V": ALPHA_CEN_V_SIGMA,
           "Johnson B": math.hypot(ALPHA_CEN_V_SIGMA, MAIZ2006_SIGMA["B-V"]),
           "Johnson U": math.sqrt(ALPHA_CEN_V_SIGMA ** 2 + MAIZ2006_SIGMA["B-V"] ** 2
                                  + MAIZ2006_SIGMA["U-B"] ** 2)}
    meas, dms = [], []
    for band in ("Johnson U", "Johnson B", "Johnson V"):
        mA, mB = comp["A"][band], comp["B"][band]
        fA, fB = 10 ** (-0.4 * mA), 10 ** (-0.4 * mB)
        s_comb = math.hypot(fA * sig[band], fB * sig[band]) / (fA + fB)
        meas.append({"passband": band, "magnitude": round(combine([mA, mB]), 4),
                     "uncertainty": round(s_comb, 4)})
        dms.append({"passband": band, "value": round(mB - mA, 4),
                    "uncertainty": round(math.sqrt(2.0) * sig[band], 4)})
    return meas, dms, comp


def main():
    out = {}
    for key, fitter in (("Capella", capella_orbit), ("AlphaCenAB", alpha_cen_orbit)):
        fit = fitter()
        pi, spi, el = parallax_with_error(fit)
        out[key] = {"orbital_parallax_mas": pi, "sigma_mas": spi,
                    "distance_pc": 1000.0 / pi, "sigma_pc": 1000.0 * spi / pi ** 2,
                    "elements": el, "weighting": "dataset"}
        print(f"{key}: pi_orb = {pi:.3f} +- {spi:.3f} mas, d = {1000/pi:.4f} pc")

    published = {"Capella": "Torres et al. 2009 Table 9: 76.67 +- 0.17 mas (Torres et al. 2015, "
                            "with Weber & Strassmeier 2011 RVs: 75.994 +- 0.089 mas)",
                 "AlphaCenAB": "Akeson et al. 2021 Table 8: 750.81 +- 0.38 mas"}
    for k in out:
        out[k]["published"] = published[k]
    RESULTS.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(RESULTS / "real_sed_orbital_parallax.json", "w"), indent=1)

    # ---------------- Capella ----------------
    meas, dms, excluded = capella_photometry()
    cap = json.load(open(EXPECTED / "Capella_photometry.json"))
    cap.update({
        "reference": ("Torres, Claret, Pavlovski & Dotter 2015, ApJ 807, 26, Table 3 "
                      "(Teff, R, log g, [Fe/H]); photometry: Mermilliod 1991 (VizieR II/168); "
                      "flux ratios: Torres, Claret & Young 2009, ApJ 700, 1349, Table 13"),
        "data_provenance": (
            "REAL measurements (0.2.0): combined-light Johnson UBV from VizieR II/168/ubvmeans "
            "(V = 0.073 +- 0.014, B-V = 0.797 +- 0.012, U-B = 0.441 +- 0.010; catalogue standard "
            "deviations of 24/24/20 observations), and B and V component magnitude differences "
            "from the flux ratios of Torres et al. (2009) Table 13. Distance = 1/pi_orb from "
            "PySeshat's own orbit fit of the Torres et al. (2009) data (subset B, dataset "
            "weighting; scripts/build_real_sed_benchmarks.py, "
            "validation/results/real_sed_orbital_parallax.json). Replaces the synthetic "
            "photometry used before 0.2.0."),
        "distance_pc": {"value": round(out["Capella"]["distance_pc"], 5),
                        "sigma": round(out["Capella"]["sigma_pc"], 5),
                        "source": "orbital parallax of PySeshat's real-data orbit fit"},
        "metallicity": {"value": 0.0, "source": (
            "nearest CK04 grid point (ckp00) to [Fe/H] = -0.04 +- 0.06 (Torres et al. 2015)")},
        "primary_logg": {"value": 2.691, "source": "Torres et al. 2015 Table 3"},
        "secondary_logg": {"value": 2.941, "source": "Torres et al. 2015 Table 3"},
        "atmosphere_files": [f"ckp00/ckp00_{t}.fits" for t in range(4000, 7001, 250)],
        "measurements": meas,
        "magnitude_differences": dms,
        "excluded_measurements": excluded,
        "initial_parameters": {"primary_temperature": 5200.0, "secondary_temperature": 5500.0,
                               "primary_radius": 11.98, "secondary_radius": 8.83, "ebv": 0.0},
        "free_parameters": ["primary_temperature", "secondary_temperature",
                            "primary_radius", "secondary_radius"],
        "literature_component_parameters": {"primary": {"teff_K": 4970.0, "radius_Rsun": 11.98},
                                            "secondary": {"teff_K": 5730.0, "radius_Rsun": 8.83}},
        "ebv_note": "E(B-V) held at 0 (13 pc).",
    })
    json.dump(cap, open(EXPECTED / "Capella_photometry.json", "w"), indent=2)

    # ---------------- alpha Cen ----------------
    meas, dms, comp = alpha_cen_photometry()
    ac = json.load(open(EXPECTED / "AlphaCenAB_photometry.json"))
    ac.update({
        "reference": ("Heiter et al. 2015, A&A 582, A49 (Teff); Kervella et al. 2017, A&A 597, "
                      "A137 (R, log g); photometry: Mermilliod 1991 (VizieR II/168)"),
        "data_provenance": (
            "REAL measurements (0.2.0): Johnson UBV of alpha Cen A (V 0.00, B-V 0.69, U-B 0.23) "
            "and B (V 1.35, B-V 0.90, U-B 0.63) from VizieR II/168/ubvmeans (two observations, "
            "one publication, no errors), combined into total-light magnitudes and B - A "
            "magnitude differences. Uncertainties: 0.020 (B-V) and 0.028 (U-B) mag, the typical "
            "single-value uncertainties of Maiz Apellaniz (2006) Tables 9-10, and an assumed "
            "0.02 mag in V; correlations between the combined magnitudes and the differences "
            "are ignored. Distance = 1/pi_orb from PySeshat's own orbit fit of the Akeson et "
            "al. (2021) data (dataset weighting; scripts/build_real_sed_benchmarks.py). "
            "Replaces the synthetic photometry used before 0.2.0."),
        "component_magnitudes": comp,
        "distance_pc": {"value": round(out["AlphaCenAB"]["distance_pc"], 6),
                        "sigma": round(out["AlphaCenAB"]["sigma_pc"], 6),
                        "source": "orbital parallax of PySeshat's real-data orbit fit"},
        "metallicity": {"value": 0.2, "source": (
            "nearest CK04 grid point (ckp02) to [Fe/H] ~ +0.23 (Morel 2018, A&A 615, A172, "
            "Sect. 5.1: 'a robust value, [Fe/H] ~ +0.23 dex')")},
        "atmosphere_files": [f"ckp02/ckp02_{t}.fits" for t in range(4250, 6751, 250)],
        "measurements": meas,
        "magnitude_differences": dms,
        "initial_parameters": {"primary_temperature": 5500.0, "secondary_temperature": 5500.0,
                               "primary_radius": 1.2234, "secondary_radius": 0.8632, "ebv": 0.0},
        "free_parameters": ["primary_temperature", "secondary_temperature",
                            "primary_radius", "secondary_radius"],
        "ebv_note": "E(B-V) held at 0 (1.3 pc).",
    })
    json.dump(ac, open(EXPECTED / "AlphaCenAB_photometry.json", "w"), indent=2)
    print("wrote", EXPECTED / "Capella_photometry.json", "and", EXPECTED / "AlphaCenAB_photometry.json")


if __name__ == "__main__":
    main()
