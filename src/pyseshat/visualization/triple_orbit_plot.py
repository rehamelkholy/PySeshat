import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt

# ==========================================================
# Hierarchical triple (ORBIT3) orbit plot
# ==========================================================
#
# See docs/guide/visualization.md and docs/guide/triple_systems.md
# ("No orbit plot yet for triples") for why a triple cannot simply
# reuse plot_visual_orbit() on elements.outer directly: the wide/
# outer-pair astrometric observations these systems are fit against
# already contain the unresolved inner subsystem's astrometric
# wobble (ORBIT3's f term: POS2 = r_outer + f*r_inner). Plotting a
# flat two-body outer-orbit curve against those observations would
# show a real, wobble-driven residual pattern a reader could easily
# mistake for a fit-quality problem, rather than the correctly-
# modeled physical effect it actually is.
#
# This module draws BOTH ORBIT3 visual observables side by side --
# POS1 (the resolved inner orbit) and POS2 (the wobble-corrected
# wide/outer orbit) -- so neither curve is plotted against
# observations it does not actually model. Same North-up/East-left
# convention and visual style as
# visualization.orbit_plot.plot_visual_orbit (settled 2026-08-27,
# see that module's docstring and docs/project_decisions.md): North
# on the vertical axis, East on the horizontal axis with the axis
# inverted (not negated) so East increases to the left.


def _plot_triple_panel(
    ax,
    X_obs,
    Y_obs,
    X_model,
    Y_model,
    subtitle,
    highlight_mask=None,
    highlight_label="Flagged observation",
    show_grid=True,
):
    """
    Draw one North-up/East-left panel (observations + model curve +
    system-center marker) for a single ORBIT3 visual observable
    (POS1 or POS2). Mirrors plot_visual_orbit's exact conventions
    (see that function's docstring) so this and the flat two-body
    plot are visually consistent within the same paper.

    Parameters
    ----------
    ax : matplotlib.axes.Axes

    X_obs, Y_obs : ndarray
        Observed Cartesian coordinates for this observable, in this
        codebase's internal ``X`` = North, ``Y`` = East convention
        (see plot_visual_orbit's docstring).

    X_model, Y_model : ndarray
        Model curve for this observable (POS1 or POS2), same
        convention.

    highlight_mask : ndarray of bool, optional
        If given, marks ``X_obs[highlight_mask]``/
        ``Y_obs[highlight_mask]`` with a distinct marker/color
        (e.g. a known residual-anomaly epoch) instead of the normal
        observation marker.

    show_grid : bool, default=True
        Whether to draw axes grid lines on this panel. Default
        preserves this function's original (always-on) behavior.
    """

    if highlight_mask is None:

        ax.scatter(
            Y_obs,
            X_obs,
            s=40,
            label="Observations",
        )

    else:

        highlight_mask = highlight_mask.astype(bool)
        normal_mask = ~highlight_mask

        ax.scatter(
            Y_obs[normal_mask],
            X_obs[normal_mask],
            s=40,
            label="Observations",
        )

        if highlight_mask.any():

            ax.scatter(
                Y_obs[highlight_mask],
                X_obs[highlight_mask],
                s=90,
                marker="D",
                color="red",
                zorder=5,
                label=highlight_label,
            )

    ax.plot(
        Y_model,
        X_model,
        linewidth=2,
        color="C1",
        label="Fitted Orbit",
    )

    ax.scatter(
        0,
        0,
        marker="*",
        s=200,
        color="black",
        label="System center",
    )

    ax.set_xlabel("East (arcsec)")
    ax.set_ylabel("North (arcsec)")

    # East increases to the left -- see plot_visual_orbit's
    # docstring for the full derivation.
    ax.invert_xaxis()

    ax.set_title(subtitle)

    ax.legend(fontsize=8)

    ax.set_aspect("equal")

    if show_grid:
        ax.grid(True)


def plot_triple_orbit(
    inner_X_obs,
    inner_Y_obs,
    outer_X_obs,
    outer_Y_obs,
    orbit3_model,
    title="Hierarchical Triple Orbit (ORBIT3)",
    inner_highlight_mask=None,
    outer_highlight_mask=None,
    highlight_label="Flagged observation",
    show=True,
    show_grid=True,
    output_format="png",
    save_path=None,
):
    """
    Plot a hierarchical triple's fitted ORBIT3 solution as two
    side-by-side North-up/East-left panels.

    Left panel: the resolved inner orbit, POS1 = r_inner (fit
    against I1 observations in this codebase's ORBIT3-native
    convention -- see scripts/compare_orbitt2_orbit3.py's module
    docstring for the I1/I2 tag convention and how it differs from
    the legacy ORBITT2 files).

    Right panel: the wobble-corrected wide/outer orbit,
    POS2 = r_outer + f * r_inner (fit against I2 observations) --
    NOT the raw outer two-body ellipse alone, which would not
    account for the astrometric wobble the wide-pair observations
    actually contain.

    Parameters
    ----------
    inner_X_obs, inner_Y_obs : ndarray
        Observed Cartesian coordinates (North, East) for the
        resolved-inner (I1/POS1) observation stream.

    outer_X_obs, outer_Y_obs : ndarray
        Observed Cartesian coordinates (North, East) for the
        wide-pair (I2/POS2) observation stream.

    orbit3_model : dict
        Output of
        ``pyseshat.triple.triple_orbit.compute_orbit3_visual()``,
        i.e. a dict with ``"POS1"`` and ``"POS2"`` sub-dicts each
        containing dense model ``"X"``/``"Y"`` arrays.

    inner_highlight_mask, outer_highlight_mask : ndarray of bool, optional
        Boolean masks (same length as the corresponding
        ``*_X_obs``/``*_Y_obs`` arrays) marking observations to draw
        with a distinct marker/color -- e.g. a known
        residual-anomaly epoch.

    show : bool
        If True, call ``plt.show()`` before returning.

    show_grid : bool, default=True
        Whether to draw axes grid lines on both panels. Default
        preserves this function's original (always-on) behavior.

    output_format : str, default="png"
        File format passed to ``Figure.savefig()`` (e.g. "png",
        "pdf", "svg"). Only takes effect when ``save_path`` is
        given.

    save_path : str or Path, optional
        If given, save the figure to this path (in ``output_format``)
        before returning. If None (default), matches this
        function's original behavior of not saving internally.

    Returns
    -------
    (Figure, (Axes, Axes))
        The inner-orbit axes first, then the outer/wobble-corrected
        axes.
    """

    fig, (ax_inner, ax_outer) = plt.subplots(
        1,
        2,
        figsize=(14, 7),
    )

    _plot_triple_panel(
        ax_inner,
        inner_X_obs,
        inner_Y_obs,
        orbit3_model["POS1"]["X"],
        orbit3_model["POS1"]["Y"],
        subtitle="Inner orbit (resolved, POS1)",
        highlight_mask=inner_highlight_mask,
        highlight_label=highlight_label,
        show_grid=show_grid,
    )

    _plot_triple_panel(
        ax_outer,
        outer_X_obs,
        outer_Y_obs,
        orbit3_model["POS2"]["X"],
        orbit3_model["POS2"]["Y"],
        subtitle="Outer orbit, wobble-corrected\n"
        "(POS2 = outer + f * inner)",
        highlight_mask=outer_highlight_mask,
        highlight_label=highlight_label,
        show_grid=show_grid,
    )

    fig.suptitle(title)

    plt.tight_layout()

    if save_path is not None:
        fig.savefig(
            save_path,
            format=output_format,
            dpi=150,
            bbox_inches="tight"
        )

    if show:
        plt.show()

    return fig, (ax_inner, ax_outer)
