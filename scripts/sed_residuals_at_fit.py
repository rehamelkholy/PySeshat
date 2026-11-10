"""Band residuals (PySeshat - Abushattal et al. 2025 Table 4, entire system) at the option-C best fits
(validation/results/sed_wide_grid_ebv0_dm.json, start "published"); Stromgren u is fitted since 0.2.0.
Needs scipy. Writes validation/results/sed_residuals_at_fit.json."""
import json, sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]; sys.path[:0]=[str(ROOT/"src"),str(ROOT/"validation")]
import benchmark_runner as br
from pyseshat.photometry.model import synthetic_binary_magnitudes
V=ROOT/"validation"
B=["Johnson U","Johnson B","Johnson V","Cousins R","Stromgren u","Stromgren v","Stromgren b","Stromgren y","Tycho BT","Tycho VT"]
grid=br.load_ck04_grid([V/"atmosphere_models"/f"ckp00/ckp00_{t}.fits" for t in range(3500,10001,250)])
pbs={b:br.load_passband(filename=V/"photometry"/"passbands"/br._SED_PASSBAND_FILES[b],name=b) for b in B}
zps={b:br._SED_ZERO_POINTS[b] for b in B}
W=json.load(open(V/"results"/"sed_wide_grid_ebv0_dm.json"))["systems"]
pub=json.load(open(V/"results"/"published_params_photometry.json"))
out={}
for s in ("HD214222","HD191854","HD130669","HD184467"):
    e=json.load(open(V/"expected_results"/f"{s}.json")); f=W[s]["runs"]["published"]["fitted"]
    m=synthetic_binary_magnitudes(grid,pbs,zps,f["primary_temperature"],f["secondary_temperature"],e["primary_logg"]["value"],e["secondary_logg"]["value"],0.0,f["primary_radius"],f["secondary_radius"],e["distance_pc"]["value"])
    obs={x["passband"]:x["magnitude"] for x in e["measurements"]}
    r={b:m[b]-obs[b] for b in B}
    fit=[r[b] for b in B]
    raw=pub[s]["rows"]["E"]["minus_table4"]; rawo=[abs(raw[b]) for b in B]
    out[s]={"residuals_mag":{b:round(float(r[b]),3) for b in B},"rms_fitted_bands":round(float(np.sqrt(np.mean(np.square(fit)))),3),"max_abs_fitted_bands":round(float(max(map(abs,fit))),3)}
    print(f"{s}: at fit: rms(10 bands) {np.sqrt(np.mean(np.square(fit))):.3f}, max|r| {max(map(abs,fit)):.3f} ({max(B,key=lambda b:abs(r[b]))}), u {r['Stromgren u']:+.3f};  at published: |off| {min(rawo):.2f}-{max(rawo):.2f}")
json.dump(out,open(V/"results"/"sed_residuals_at_fit.json","w"),indent=1)
