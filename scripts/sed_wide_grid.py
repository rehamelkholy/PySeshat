"""
SED benchmarks on one common, wide atmosphere grid (paper Sect. 3.4).

The benchmark files validation/expected_results/HD*.json restricted each system's
CK04 grid to 3-4 nodes bracketing its published temperatures (e.g. HD130669:
5000-5500 K for published 5290/5230 K), which bounds the temperature agreement by
construction, and started every fit at the published temperatures. Here each of the
four Abushattal et al. (2025) systems is refitted with
  * the same grid for all systems: CK04 solar metallicity, 3500-10000 K (27 models);
  * five starting points: the published temperatures and four neutral pairs
    (4500/4500, 6000/6000, 7500/5500, 5000/7000 K);
everything else as in validation/benchmark_runner.run_sed_benchmark (radius prior
10 per cent around the published radius, 0.30 mag errors, E(B-V) free unless --ebv0).
With --ebv0, E(B-V) is held at 0 (all four systems lie within 75 pc). With --dm, the
Johnson-V component magnitude difference implied by Abushattal et al. (2025) Table 4 is added
as a constraint (sigma 0.05 mag), as in the published method.
Needs scipy. Writes validation/results/sed_wide_grid[_ebv0][_dm].json.
"""
import copy
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT / "src", ROOT / "validation"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from benchmark_runner import run_sed_benchmark  # noqa: E402

SYSTEMS = ("HD214222", "HD191854", "HD130669", "HD184467")
WIDE = [f"ckp00/ckp00_{t}.fits" for t in range(3500, 10001, 250)]
STARTS = {"published": None, "4500/4500": (4500.0, 4500.0), "6000/6000": (6000.0, 6000.0),
          "7500/5500": (7500.0, 5500.0), "5000/7000": (5000.0, 7000.0)}
V = ROOT / "validation"
# Component magnitude difference m_B - m_A in Johnson V from Abushattal et al. (2025, AJ 170, 268)
# Table 4 (component rows A and B of their adopted solution); the paper states the 0.09 mag
# value as the input for HD 130669. Assumed uncertainty 0.05 mag (typical of speckle Delta m).
DM_V = {"HD130669": 9.22 - 9.13, "HD184467": 7.53 - 7.20, "HD191854": 9.31 - 7.61,
        "HD214222": 8.95 - 8.80}
DM_SIGMA = 0.05


def main(fix_ebv=False, use_dm=False):
    """fix_ebv=True: E(B-V) held at 0 (all four systems are within 75 pc).
    use_dm=True: add the Johnson-V component magnitude difference DM_V (sigma DM_SIGMA)."""
    missing = [f for f in WIDE if not (V / "atmosphere_models" / f).exists()]
    if missing:
        raise SystemExit(f"missing grid files: {missing}")
    out = {"grid": "CK04 ckp00 3500-10000 K (27 models)", "ebv_fixed_zero": fix_ebv,
           "delta_m_V": {k: round(v, 3) for k, v in DM_V.items()} if use_dm else None,
           "delta_m_sigma": DM_SIGMA if use_dm else None, "systems": {}}
    for s in SYSTEMS:
        exp = json.load(open(V / "expected_results" / f"{s}.json"))
        runner = json.load(open(V / "results" / f"{s}.json"))  # benchmark_runner.py result
        lit = exp["literature_component_parameters"]
        res = {"published_teff": [lit["primary"]["teff_K"], lit["secondary"]["teff_K"]],
               "benchmark_grid_files": exp["atmosphere_files"],
               "benchmark_runner_result": {"fitted": runner["fitted_elements"],
                                           "reduced_chi2": runner["statistics"]["reduced_chi_square"]},
               "runs": {}}
        for name, st in STARTS.items():
            e = copy.deepcopy(exp)
            e["atmosphere_files"] = WIDE
            # Set both switches explicitly: since 0.2.0 the expected_results files themselves hold
            # E(B-V) = 0 and carry the Delta m constraint, so the "without" variants must remove them.
            if use_dm:
                e["magnitude_differences"] = [{"passband": "Johnson V", "value": round(DM_V[s], 3),
                                               "uncertainty": DM_SIGMA}]
            else:
                e["magnitude_differences"] = []
            if fix_ebv:
                e["free_parameters"] = [k for k in e["free_parameters"] if k != "ebv"]
                e["initial_parameters"]["ebv"] = 0.0
            elif "ebv" not in e["free_parameters"]:
                e["free_parameters"] = list(e["free_parameters"]) + ["ebv"]
            if st is not None:
                e["initial_parameters"]["primary_temperature"], e["initial_parameters"]["secondary_temperature"] = st
            r = run_sed_benchmark(s, e)
            f = r["fitted_elements"]
            res["runs"][name] = {"chi2": r["statistics"]["chi_square"],
                                 "reduced_chi2": r["statistics"]["reduced_chi_square"],
                                 "fitted": f, "success": r["optimizer_success"],
                                 "rel_err_pct": {k: 100 * v["relative_error"] for k, v in r["parameter_errors"].items()}}
        best = min(res["runs"], key=lambda k: res["runs"][k]["chi2"])
        cmin = res["runs"][best]["chi2"]
        near = [k for k, v in res["runs"].items() if v["chi2"] - cmin < 1.0]
        res["best_start"], res["chi2_min"], res["starts_within_dchi2_1"] = best, cmin, near
        res["teff_range_within_dchi2_1"] = {
            c: [min(res["runs"][k]["fitted"][f"{c}_temperature"] for k in near),
                max(res["runs"][k]["fitted"][f"{c}_temperature"] for k in near)] for c in ("primary", "secondary")}
        out["systems"][s] = res
        print(f"\n== {s}: published Teff {res['published_teff'][0]:.0f} / {res['published_teff'][1]:.0f} K; "
              f"benchmark runner -> "
              f"{runner['fitted_elements']['primary_temperature']:.0f} / {runner['fitted_elements']['secondary_temperature']:.0f} K")
        for k, v in res["runs"].items():
            f = v["fitted"]
            print(f"   start {k:10s} chi2 {v['chi2']:7.3f} (red {v['reduced_chi2']:.3f})  T1 {f['primary_temperature']:7.0f}  "
                  f"T2 {f['secondary_temperature']:7.0f}  R1 {f['primary_radius']:.3f}  R2 {f['secondary_radius']:.3f}  "
                  f"E(B-V) {f['ebv']:.3f}  max|Teff err| {max(v['rel_err_pct']['primary_temperature'], v['rel_err_pct']['secondary_temperature'], key=abs):+.1f}%")
        print(f"   starts within dchi2<1 of the minimum: {near}; Teff ranges {res['teff_range_within_dchi2_1']}")
    name = "sed_wide_grid" + ("_ebv0" if fix_ebv else "") + ("_dm" if use_dm else "") + ".json"
    (V / "results" / name).write_text(json.dumps(out, indent=1))
    print(f"\nwrote validation/results/{name}")


if __name__ == "__main__":
    main(fix_ebv="--ebv0" in sys.argv, use_dm="--dm" in sys.argv)
