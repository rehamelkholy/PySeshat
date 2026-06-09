import numpy as np

from pyseshat.io.orbit_parser import (
    parse_orbit_file
)

from pyseshat.analysis.dynamics import (
    separation_vs_time,
    separation_table
)

from pyseshat.visualization.dynamics_plot import (
    plot_separation_vs_time
)

parsed = parse_orbit_file(
    "validation/benchmark_systems/FIN379/FIN379.inp"
)

elements = parsed["elements"]

epochs = np.linspace(
    2000,
    2025,
    500
)

data = separation_vs_time(
    elements,
    epochs
)

print()

print(data.keys())

print()

print(
    separation_table(
        elements,
        epochs
    ).head()
)

fig, ax = plot_separation_vs_time(

    data["epoch"],

    data["rho"],

    show=True
)

print()

print(
    "Plot generated."
)

print()

print(
    "Min separation:",
    np.min(data["rho"])
)

print(
    "Max separation:",
    np.max(data["rho"])
)