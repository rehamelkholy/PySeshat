"""Option-C fit (wide grid, E(B-V)=0, Delta m_V, 10% radius prior) repeated with the distance
scaled so that PySeshat's photometry at the published parameters matches Abushattal et al.
(2025) Table 4 on average (mean over the ten fitted bands, entire-system row).
Run after scripts/published_params_photometry.py. Needs scipy.
Writes validation/results/refit_scaled_distance.json."""
import copy, json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path[:0] = [str(ROOT/"src"), str(ROOT/"validation")]
from benchmark_runner import run_sed_benchmark
V = ROOT/"validation"; pub = json.load(open(V/"results"/"published_params_photometry.json"))
WIDE = [f"ckp00/ckp00_{t}.fits" for t in range(3500, 10001, 250)]
DM_V = {"HD130669": 0.09, "HD184467": 0.33, "HD191854": 1.70, "HD214222": 0.15}
out = {}
for s in ("HD214222", "HD191854", "HD130669", "HD184467"):
    exp = json.load(open(V/"expected_results"/f"{s}.json"))
    off = pub[s]["rows"]["E"]["mean_all_bands"]
    for label, d in (("adopted", exp["distance_pc"]["value"]), ("scaled", exp["distance_pc"]["value"]*10**(-off/5))):
        e = copy.deepcopy(exp); e["atmosphere_files"] = WIDE; e["distance_pc"]["value"] = d
        e["free_parameters"] = [k for k in e["free_parameters"] if k != "ebv"]; e["initial_parameters"]["ebv"] = 0.0
        e["magnitude_differences"] = [{"passband": "Johnson V", "value": DM_V[s], "uncertainty": 0.05}]
        r = run_sed_benchmark(s, e); f = r["fitted_elements"]; pe = r["parameter_errors"]
        out.setdefault(s, {})[label] = {"distance_pc": round(d, 2), "teff": [round(f["primary_temperature"]), round(f["secondary_temperature"])],
            "teff_err_pct": [round(100*pe["primary_temperature"]["relative_error"], 2), round(100*pe["secondary_temperature"]["relative_error"], 2)],
            "radius": [round(f["primary_radius"], 3), round(f["secondary_radius"], 3)],
            "radius_err_pct": [round(100*pe["primary_radius"]["relative_error"], 2), round(100*pe["secondary_radius"]["relative_error"], 2)],
            "reduced_chi2": round(r["statistics"]["reduced_chi_square"], 4)}
        o = out[s][label]
        print(f"{s} {label:8s} d={o['distance_pc']:6.2f} pc  Teff {o['teff']}  err% {o['teff_err_pct']}  R err% {o['radius_err_pct']}  chi2nu {o['reduced_chi2']}")
json.dump(out, open(V/"results"/"refit_scaled_distance.json", "w"), indent=1)
