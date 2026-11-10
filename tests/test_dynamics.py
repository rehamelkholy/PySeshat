"""
Regression test for analysis.dynamics/visualization.dynamics_plot,
with assertions on the computed values and plotted data.
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


# ==========================================================
# show_grid / output_format / save_path
#
# Tests for the show_grid/output_format/save_path parameters --
# same coverage pattern as
# tests/test_orbit_plot.py.
# ==========================================================

def test_plot_separation_vs_time_show_grid_toggle():

    epochs = np.linspace(2000, 2025, 10)
    rho = np.abs(np.sin(epochs))

    fig_default, ax_default = plot_separation_vs_time(
        epochs, rho, show=False
    )
    assert (
        ax_default.xaxis.get_gridlines()[0].get_visible()
        and ax_default.yaxis.get_gridlines()[0].get_visible()
    )
    plt.close(fig_default)

    fig_off, ax_off = plot_separation_vs_time(
        epochs, rho, show=False, show_grid=False
    )
    assert not (
        ax_off.xaxis.get_gridlines()[0].get_visible()
        or ax_off.yaxis.get_gridlines()[0].get_visible()
    )
    plt.close(fig_off)


def test_plot_separation_vs_time_save_path_writes_requested_format(
    tmp_path,
):

    epochs = np.linspace(2000, 2025, 10)
    rho = np.abs(np.sin(epochs))

    out = tmp_path / "separation.pdf"

    fig, ax = plot_separation_vs_time(
        epochs, rho,
        show=False,
        output_format="pdf",
        save_path=out,
    )

    assert out.exists()
    assert out.stat().st_size > 0
    assert out.read_bytes()[:4] == b"%PDF"

    plt.close(fig)
