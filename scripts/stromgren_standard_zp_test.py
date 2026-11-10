"""Comparison made before the 0.2.0 zero-point fix (kept for the record): option-C SED
fits with the Stromgren zero points moved from 'Vega = 0' (SVO) to the standard uvby system, i.e. Vega u,v,b,y = 1.444, 0.195, 0.034, 0.030
(Bessell 2011, PASP 123, 1442: y = 0.03, b-y = 0.004, m1 = 0.157, c1 = 1.088 from Hauck &
Mermilliod 1998), with and without Stromgren u included. Needs scipy.
Writes validation/results/stromgren_standard_zp_test.json.
No longer runnable as is: it needs the pre-0.2.0 passband files (replaced
by the Maiz Apellaniz 2006 photon-counting curves) and the old "Johnson R"
band name. Running it exits with a message; the recorded results stay in
validation/results/stromgren_standard_zp_test.json."""
import sys


def _historical():
    import copy, json, sys
    from pathlib import Path
    ROOT = Path(__file__).resolve().parents[1]; sys.path[:0] = [str(ROOT/"src"), str(ROOT/"validation")]
    import benchmark_runner as br
    V = ROOT/"validation"
    VEGA_STD = {"Stromgren u": 1.444, "Stromgren v": 0.195, "Stromgren b": 0.034, "Stromgren y": 0.030}
    ZP0 = dict(br._SVO_VEGA_ZERO_POINTS)  # Vega = 0 in every band (pre-0.2.0 convention)
    WIDE = [f"ckp00/ckp00_{t}.fits" for t in range(3500, 10001, 250)]
    DM_V = {"HD130669": 0.09, "HD184467": 0.33, "HD191854": 1.70, "HD214222": 0.15}
    out = {"vega_standard_mags": VEGA_STD, "systems": {}}
    for s in ("HD214222", "HD191854", "HD130669", "HD184467"):
        exp = json.load(open(V/"expected_results"/f"{s}.json"))
        lit = exp["literature_component_parameters"]
        for label, std, use_u in (("svo_vega0", False, False), ("standard", True, False), ("standard_with_u", True, True)):
            br._SED_ZERO_POINTS.clear(); br._SED_ZERO_POINTS.update(ZP0)
            if std:
                for b, m in VEGA_STD.items():
                    br._SED_ZERO_POINTS[b] = ZP0[b] * 10 ** (0.4 * m)  # m = -2.5 log(F/ZP0) + m_Vega
            e = copy.deepcopy(exp); e["atmosphere_files"] = WIDE
            e["free_parameters"] = [k for k in e["free_parameters"] if k != "ebv"]; e["initial_parameters"]["ebv"] = 0.0
            e["magnitude_differences"] = [{"passband": "Johnson V", "value": DM_V[s], "uncertainty": 0.05}]
            if not use_u:
                e["measurements"] = [x for x in e["measurements"] if x["passband"] != "Stromgren u"]
            r = br.run_sed_benchmark(s, e); f = r["fitted_elements"]; pe = r["parameter_errors"]
            o = {"teff": [round(f["primary_temperature"]), round(f["secondary_temperature"])],
                 "teff_err_pct": [round(100*pe["primary_temperature"]["relative_error"], 2), round(100*pe["secondary_temperature"]["relative_error"], 2)],
                 "split_K": round(f["primary_temperature"] - f["secondary_temperature"]),
                 "published_split_K": lit["primary"]["teff_K"] - lit["secondary"]["teff_K"],
                 "reduced_chi2": round(r["statistics"]["reduced_chi_square"], 4)}
            out["systems"].setdefault(s, {})[label] = o
            print(f"{s} {label:16s} Teff {o['teff']} err% {o['teff_err_pct']} split {o['split_K']} (pub {o['published_split_K']}) chi2nu {o['reduced_chi2']}")
    br._SED_ZERO_POINTS.clear(); br._SED_ZERO_POINTS.update(ZP0)
    json.dump(out, open(V/"results"/"stromgren_standard_zp_test.json", "w"), indent=1)


if __name__ == "__main__":
    sys.exit("scripts/stromgren_standard_zp_test.py is a pre-0.2.0 record and needs the old "
             "passband files; see its docstring. Recorded results: "
             "validation/results/stromgren_standard_zp_test.json")
