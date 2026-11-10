"""
Ad-hoc comparison script (not a permanent part of the repo) to test
whether re-fitting AlphaCenAB_photometry and HD184467 at a
closer-to-literature CK04 metallicity slice (now that the full grid
is downloaded) changes their reported fit quality.

Usage: MPLBACKEND=Agg PYTHONPATH=src python3 scripts/ck04_metallicity_rerun_check.py
"""

import copy
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "validation"))

from benchmark_runner import run_sed_benchmark, _VALIDATION_DIR  # noqa: E402


def load_expected(name):
    with open(_VALIDATION_DIR / "expected_results" / f"{name}.json") as f:
        return json.load(f)


def summarize(label, result):
    stats = result["statistics"]
    print(f"\n--- {label} ---")
    print(f"  reduced_chi_square = {stats['reduced_chi_square']:.4f}  (chi2={stats['chi_square']:.3f}, n_data={stats['n_data']}, n_params={stats['n_params']})")
    for name, err in result["parameter_errors"].items():
        print(f"  {name}: fitted={err['fitted']:.2f}  literature={err['expected']:.2f}  rel_err={err['relative_error']*100:.2f}%")
    print(f"  optimizer_success={result['optimizer_success']}  msg={result['optimizer_message']!r}")


# ==========================================================
# AlphaCenAB_photometry: current 0.0 (ckp00) vs +0.2 (ckp02),
# literature ~+0.23 (Morel 2018, A&A 615, A172)
# ==========================================================

expected = load_expected("AlphaCenAB_photometry")

print("=" * 70)
print("AlphaCenAB_photometry -- literature [Fe/H] ~= +0.23 (Morel 2018)")
print("=" * 70)

baseline = run_sed_benchmark("AlphaCenAB_photometry", copy.deepcopy(expected))
summarize("baseline: metallicity=0.0 (ckp00, current)", baseline)

variant = copy.deepcopy(expected)
variant["metallicity"]["value"] = 0.2
variant["atmosphere_files"] = [
    p.replace("ckp00/ckp00_", "ckp02/ckp02_")
    for p in variant["atmosphere_files"]
]
result_p02 = run_sed_benchmark("AlphaCenAB_photometry", variant)
summarize("variant: metallicity=+0.2 (ckp02, nearest to literature)", result_p02)


# ==========================================================
# HD184467: current 0.0 (ckp00) vs -0.5 (ckm05),
# literature -0.25 (exact tie in dex-distance between 0.0 and -0.5)
# ==========================================================

expected2 = load_expected("HD184467")

print()
print("=" * 70)
print("HD184467 -- literature [Fe/H] = -0.25 (exact tie between 0.0 and -0.5)")
print("=" * 70)

baseline2 = run_sed_benchmark("HD184467", copy.deepcopy(expected2))
summarize("baseline: metallicity=0.0 (ckp00, current)", baseline2)

variant2 = copy.deepcopy(expected2)
variant2["metallicity"]["value"] = -0.5
variant2["atmosphere_files"] = [
    p.replace("ckp00/ckp00_", "ckm05/ckm05_")
    for p in variant2["atmosphere_files"]
]
result_m05 = run_sed_benchmark("HD184467", variant2)
summarize("variant: metallicity=-0.5 (ckm05, equidistant alternative)", result_m05)
