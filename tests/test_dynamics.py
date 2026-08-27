"""
Regression test for analysis.dynamics/visualization.dynamics_plot,
rewritten 2026-08-27 from a print-only demo script (no
``def test_...`` function, no assertions -- the same defect fixed
for several other files in docs/project_decisions.md's "Whole-
codebase audit" section) into a real assertion-bearing test, per
docs/development_plan_phase2.md Phase 4's charge to strengthen
plot-correctness coverage.
"""

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
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


def test_separation_vs_time_matches_orbit_model_and_plots_correctly():

    parsed = parse_orbit_file(
        "validation/benchmark_systems/FIN379/FIN379.inp"
    )

    elements = parsed["elements"]

    epochs = np.linspace(2000, 2025, 500)

    data = separation_vs_time(elements, epochs)

    assert set(data.keys()) >= {"epoch", "rho"}

    np.testing.assert_allclose(data["epoch"], epochs)

    # Separation is a physical distance -- always non-negative,
    # and (for FIN379's e=0.4988, a=0.1003") never larger than
    # a(1+e).

    assert np.all(data["rho"] >= 0.0)

    assert np.max(data["rho"]) <= elements.a * (1.0 + elements.e) * 1.001

    table = separation_table(elements, epochs)

    np.testing.assert_allclose(
        table["Separation"].to_numpy(),
        data["rho"],
    )

    fig, ax = plot_separation_vs_time(
        data["epoch"],
        data["rho"],
        show=False,
    )

    line = ax.get_lines()[0]

    np.testing.assert_allclose(line.get_xdata(), data["epoch"])
    np.testing.assert_allclose(line.get_ydata(), data["rho"])

    assert ax.get_xlabel() == "Epoch"
    assert ax.get_ylabel() == "Separation (arcsec)"

    plt.close(fig)
