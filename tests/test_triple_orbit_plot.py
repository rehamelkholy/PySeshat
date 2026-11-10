import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from pyseshat.visualization.triple_orbit_plot import plot_triple_orbit


# ==========================================================
# plot_triple_orbit
# ==========================================================
#
# Same correctness standard as tests/test_orbit_plot.py: assert on
# the actual plotted arrays (ax.collections[].get_offsets(),
# ax.get_lines()[].get_xdata()/get_ydata()), not just "it ran
# without raising" -- this catches a swapped-panel or
# swapped-axis regression a smoke test would miss.


def _orbit3_model():

    return {
        "POS1": {
            "X": np.linspace(-0.1, 0.1, 50),
            "Y": np.linspace(-0.1, 0.1, 50) ** 2,
        },
        "POS2": {
            "X": np.linspace(-3.0, 3.0, 50),
            "Y": np.linspace(-3.0, 3.0, 50) ** 2,
        },
    }


def test_plot_triple_orbit_returns_two_panels():

    inner_X = np.array([0.02, -0.03, 0.01])
    inner_Y = np.array([0.05, -0.02, -0.04])

    outer_X = np.array([1.0, -2.0, 0.5])
    outer_Y = np.array([1.5, 0.5, -1.0])

    fig, (ax_inner, ax_outer) = plot_triple_orbit(
        inner_X,
        inner_Y,
        outer_X,
        outer_Y,
        _orbit3_model(),
        show=False,
    )

    assert fig is not None
    assert ax_inner is not None
    assert ax_outer is not None
    assert ax_inner is not ax_outer

    assert ax_inner.get_xlabel() == "East (arcsec)"
    assert ax_inner.get_ylabel() == "North (arcsec)"
    assert ax_outer.get_xlabel() == "East (arcsec)"
    assert ax_outer.get_ylabel() == "North (arcsec)"

    plt.close(fig)


def test_plot_triple_orbit_draws_correct_data_in_correct_panel():

    # Distinct, easy-to-mis-transpose-or-swap values for each of
    # the two panels, so a swapped-panel or swapped-axis regression
    # would be caught.

    inner_X = np.array([0.02, -0.03, 0.01])
    inner_Y = np.array([0.05, -0.02, -0.04])

    outer_X = np.array([1.0, -2.0, 0.5])
    outer_Y = np.array([1.5, 0.5, -1.0])

    model = _orbit3_model()

    fig, (ax_inner, ax_outer) = plot_triple_orbit(
        inner_X,
        inner_Y,
        outer_X,
        outer_Y,
        model,
        show=False,
    )

    # Inner panel: observation scatter is the first PathCollection
    # (see _plot_triple_panel's "Observations" block, added before
    # the model line and the system-center marker).

    inner_scatter = ax_inner.collections[0]
    inner_offsets = np.asarray(inner_scatter.get_offsets())

    np.testing.assert_allclose(inner_offsets[:, 0], inner_Y)
    np.testing.assert_allclose(inner_offsets[:, 1], inner_X)

    inner_line = ax_inner.get_lines()[0]
    np.testing.assert_allclose(
        inner_line.get_xdata(), model["POS1"]["Y"]
    )
    np.testing.assert_allclose(
        inner_line.get_ydata(), model["POS1"]["X"]
    )

    # Outer panel: same layout, must use POS2/outer data, not the
    # inner panel's.

    outer_scatter = ax_outer.collections[0]
    outer_offsets = np.asarray(outer_scatter.get_offsets())

    np.testing.assert_allclose(outer_offsets[:, 0], outer_Y)
    np.testing.assert_allclose(outer_offsets[:, 1], outer_X)

    outer_line = ax_outer.get_lines()[0]
    np.testing.assert_allclose(
        outer_line.get_xdata(), model["POS2"]["Y"]
    )
    np.testing.assert_allclose(
        outer_line.get_ydata(), model["POS2"]["X"]
    )

    plt.close(fig)


def test_plot_triple_orbit_is_north_up_east_left():

    # Same literature-convention regression test as
    # test_plot_visual_orbit_is_north_up_east_left: North stays
    # unmodified on the vertical axis, East is inverted on the
    # horizontal axis so East increases to the left.

    inner_X = np.array([0.0])
    inner_Y = np.array([0.0])

    outer_X = np.array([0.0])
    outer_Y = np.array([0.0])

    fig, (ax_inner, ax_outer) = plot_triple_orbit(
        inner_X,
        inner_Y,
        outer_X,
        outer_Y,
        _orbit3_model(),
        show=False,
    )

    assert ax_inner.xaxis_inverted()
    assert not ax_inner.yaxis_inverted()
    assert ax_outer.xaxis_inverted()
    assert not ax_outer.yaxis_inverted()

    plt.close(fig)


def test_plot_triple_orbit_highlight_mask_splits_observations():

    inner_X = np.array([0.02, -0.03, 0.01])
    inner_Y = np.array([0.05, -0.02, -0.04])

    outer_X = np.array([1.0, -2.0, 0.5])
    outer_Y = np.array([1.5, 0.5, -1.0])

    highlight_mask = np.array([False, True, False])

    fig, (ax_inner, ax_outer) = plot_triple_orbit(
        inner_X,
        inner_Y,
        outer_X,
        outer_Y,
        _orbit3_model(),
        inner_highlight_mask=highlight_mask,
        show=False,
    )

    # With a highlight mask given, the inner panel draws two
    # separate scatter series (normal + highlighted) before the
    # model line -- so ax.collections must contain both, and the
    # highlighted point's coordinates must appear in the second
    # series, not the first.

    normal_scatter = ax_inner.collections[0]
    highlight_scatter = ax_inner.collections[1]

    normal_offsets = np.asarray(normal_scatter.get_offsets())
    highlight_offsets = np.asarray(highlight_scatter.get_offsets())

    np.testing.assert_allclose(
        normal_offsets[:, 0], inner_Y[~highlight_mask]
    )
    np.testing.assert_allclose(
        normal_offsets[:, 1], inner_X[~highlight_mask]
    )

    np.testing.assert_allclose(
        highlight_offsets[:, 0], inner_Y[highlight_mask]
    )
    np.testing.assert_allclose(
        highlight_offsets[:, 1], inner_X[highlight_mask]
    )

    # Outer panel had no highlight mask -- must fall back to the
    # single-series path.

    assert len(ax_outer.collections) == 2  # observations + system-center marker

    plt.close(fig)


# ==========================================================
# show_grid / output_format / save_path
#
# Tests for the show_grid/output_format/save_path parameters --
# both panels must respond to
# show_grid identically, and save_path/output_format must
# actually write the requested file.
# ==========================================================

def test_plot_triple_orbit_show_grid_toggle_applies_to_both_panels():

    inner_X = np.array([0.02, -0.03, 0.01])
    inner_Y = np.array([0.05, -0.02, -0.04])
    outer_X = np.array([1.0, -2.0, 0.5])
    outer_Y = np.array([1.5, 0.5, -1.0])

    fig_default, (ax_inner_d, ax_outer_d) = plot_triple_orbit(
        inner_X, inner_Y, outer_X, outer_Y, _orbit3_model(), show=False
    )
    assert ax_inner_d.xaxis.get_gridlines()[0].get_visible()
    assert ax_outer_d.xaxis.get_gridlines()[0].get_visible()
    plt.close(fig_default)

    fig_off, (ax_inner_o, ax_outer_o) = plot_triple_orbit(
        inner_X, inner_Y, outer_X, outer_Y, _orbit3_model(),
        show=False, show_grid=False,
    )
    assert not ax_inner_o.xaxis.get_gridlines()[0].get_visible()
    assert not ax_outer_o.xaxis.get_gridlines()[0].get_visible()
    plt.close(fig_off)


def test_plot_triple_orbit_save_path_writes_requested_format(tmp_path):

    inner_X = np.array([0.02, -0.03, 0.01])
    inner_Y = np.array([0.05, -0.02, -0.04])
    outer_X = np.array([1.0, -2.0, 0.5])
    outer_Y = np.array([1.5, 0.5, -1.0])

    out = tmp_path / "triple.pdf"

    fig, axes = plot_triple_orbit(
        inner_X, inner_Y, outer_X, outer_Y, _orbit3_model(),
        show=False,
        output_format="pdf",
        save_path=out,
    )

    assert out.exists()
    assert out.stat().st_size > 0
    assert out.read_bytes()[:4] == b"%PDF"

    plt.close(fig)
