"""PySeshat synthetic magnitudes at the published Abushattal et al. (2025) parameters
(Table 5 Teff, R; distances and log g as in validation/expected_results; E(B-V)=0)
compared band by band with their Table 4 synthetic magnitudes (rows E, A, B), and their
Table 5 M_V compared with Table 4 V at the adopted distance. Zero points: benchmark_runner._SED_ZERO_POINTS
(standard systems since 0.2.0). Needs scipy.
Writes validation/results/published_params_photometry.json."""
import json, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]; sys.path[:0] = [str(ROOT/"src"), str(ROOT/"validation")]
from benchmark_runner import _SED_ZERO_POINTS, _SED_PASSBAND_FILES
from pyseshat.photometry.model import synthetic_binary_magnitudes
from benchmark_runner import load_ck04_grid
V = ROOT/"validation"
BANDS = ["Johnson U","Johnson B","Johnson V","Cousins R","Stromgren u","Stromgren v","Stromgren b","Stromgren y","Tycho BT","Tycho VT"]
# Abushattal et al. 2025, AJ 170, 268, Table 4: U B V R u v b y BT VT
T4 = {
 "HD130669": {"E":[9.78,9.28,8.42,7.95,10.94,9.76,8.88,8.38,9.52,8.51],
              "A":[10.46,9.98,9.13,8.67,11.62,10.45,9.58,9.09,10.21,9.22],
              "B":[10.62,10.09,9.22,8.74,11.78,10.56,9.68,9.17,10.33,9.31]},
 "HD184467": {"E":[7.95,7.46,6.60,6.13,9.11,7.93,7.05,6.56,7.69,6.69],
              "A":[8.49,8.04,7.20,6.75,9.64,8.50,7.64,7.16,8.27,7.29],
              "B":[8.97,8.41,7.53,7.04,10.14,8.92,7.99,7.48,7.74,7.16]},
 "HD191854": {"E":[8.26,8.07,7.40,7.04,9.40,8.43,7.77,7.37,8.24,7.47],
              "A":[8.40,8.25,7.61,7.26,9.55,8.59,7.97,7.58,8.41,7.68],
              "B":[10.50,10.12,9.31,8.88,11.65,10.55,9.75,9.28,10.33,9.40]},
 "HD214222": {"E":[8.85,8.73,8.19,7.79,9.99,9.06,8.46,8.09,8.87,8.19],
              "A":[9.51,9.40,8.80,8.47,10.66,9.73,9.14,8.77,9.55,8.87],
              "B":[9.70,9.57,8.95,8.61,10.85,9.91,9.30,8.92,9.72,9.02]}}
# Table 5 M_V of A and B
MV5 = {"HD130669":(5.4,5.55),"HD184467":(5.70,6.01),"HD191854":(3.73,5.40),"HD214222":(4.32,4.46)}
import benchmark_runner as br
lp = br.load_passband
grid = load_ck04_grid([V/"atmosphere_models"/f"ckp00/ckp00_{t}.fits" for t in range(3500,10001,250)])
pbs = {b: lp(filename=V/"photometry"/"passbands"/_SED_PASSBAND_FILES[b], name=b) for b in BANDS}
zps = {b: _SED_ZERO_POINTS[b] for b in BANDS}
out = {}
for s in T4:
    e = json.load(open(V/"expected_results"/f"{s}.json"))
    L = e["literature_component_parameters"]; d = e["distance_pc"]["value"]
    T1,T2 = L["primary"]["teff_K"], L["secondary"]["teff_K"]; R1,R2 = L["primary"]["radius_Rsun"], L["secondary"]["radius_Rsun"]
    g1,g2 = e["primary_logg"]["value"], e["secondary_logg"]["value"]
    tiny = 1e-6
    def mags(r1, r2):
        m = synthetic_binary_magnitudes(grid, pbs, zps, T1, T2, g1, g2, 0.0, r1, r2, d, ebv=0.0)
        return np.array([m[b] for b in BANDS])
    model = {"E": mags(R1,R2), "A": mags(R1,tiny), "B": mags(tiny,R2)}
    # tiny-radius companion adds 2.5log10(1+1e-12) -> negligible
    res = {"distance_pc": d, "teff": [T1,T2], "radius": [R1,R2], "rows": {}}
    print(f"\n== {s}  d={d} pc  Teff {T1}/{T2} K  R {R1}/{R2} Rsun   (PySeshat - Table4, mag)")
    print("      " + " ".join(f"{b.split()[1]:>6s}" for b in BANDS) + "   mean(10 bands)  rms-about-mean")
    for row in "EAB":
        diff = model[row] - np.array(T4[s][row]); m = diff
        res["rows"][row] = {"pyseshat": dict(zip(BANDS, np.round(model[row],3).tolist())),
                            "minus_table4": dict(zip(BANDS, np.round(diff,3).tolist())),
                            "mean_all_bands": round(float(m.mean()),3), "rms_about_mean": round(float(m.std()),3)}
        print(f"  {row}   " + " ".join(f"{x:+6.2f}" for x in diff) + f"   {m.mean():+.3f}        {m.std():.3f}")
    dm = 5*np.log10(d) - 5
    mvA, mvB = MV5[s]
    res["table5_MV_plus_DM_minus_table4_V"] = [round(mvA+dm-T4[s]["A"][2],3), round(mvB+dm-T4[s]["B"][2],3)]
    res["distance_implied_by_table4_V_and_table5_MV_pc"] = [round(10**((T4[s]["A"][2]-mvA+5)/5),1), round(10**((T4[s]["B"][2]-mvB+5)/5),1)]
    print(f"  Table5 M_V + DM({d} pc) - Table4 V: A {res['table5_MV_plus_DM_minus_table4_V'][0]:+.2f}, B {res['table5_MV_plus_DM_minus_table4_V'][1]:+.2f};"
          f"  distance implied by Table4 V and Table5 M_V: {res['distance_implied_by_table4_V_and_table5_MV_pc']} pc")
    pyMV = model["A"][2]-dm, model["B"][2]-dm
    res["pyseshat_MV"] = [round(float(x),3) for x in pyMV]
    print(f"  PySeshat M_V at published Teff,R: A {pyMV[0]:.2f}, B {pyMV[1]:.2f}  (Table 5: {mvA}, {mvB})")
    out[s] = res
json.dump(out, open(V/"results"/"published_params_photometry.json", "w"), indent=1)
