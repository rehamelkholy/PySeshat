from pyseshat.workflows.visual_fit import run_visual_fit

result = run_visual_fit(
    "MinimalOrbitSolver/FIN379.inp"
)

print(result["statistics"])

print(result["fitted_elements"])