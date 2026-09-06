"""
Al-Wardat SED-fit plot (essential figure 4, ``Paper_Roadmap.pdf``).

Why this module exists (2026-09-02)
------------------------------------------------------------
No plotting function for ``fit_photometry()``'s output existed
anywhere in ``pyseshat.visualization`` before this module -- confirmed
by grepping the whole ``visualization/`` package and
``validation/results/plots/`` for anything SED/photometry-plot related
(nothing beyond ``plot_binary_hrdiagram()``, a different figure) before
writing this. Tracked as future-work item 9 in
``docs/paper_reference_notes.md`` ("Build a SED-fit figure ... no
plotting function currently exists for ``fit_photometry()``'s output")
and as a genuine gap in ``docs/paper_reproduction_notes.md``'s essential-
figures table.

Visual convention, and why
------------------------------------------------------------
Checked the actual Al-Wardat-method literature rather than assuming a
convention: Al-Wardat (2002, 2012)'s own SED-fit figures plot flux
density versus wavelength ("Figure 1 shows the best fit between the
total synthetic SED and the observational one" -- ``wardat_2012.pdf``),
and Abushattal et al. 2025 (arXiv:2512.23652, the direct source of this
package's four benchmark systems) follows the same convention in its
own Figures 1-4: wavelength (Angstrom) on the x-axis, flux on the
y-axis, THREE curves per system (total/composite in one color,
component A and component B each in a different, fainter/dashed
color), with observed photometric points overplotted as discrete
markers at each passband's characteristic wavelength. This module
follows that convention directly: ``plot_sed_fit()`` draws the
composite model as a solid curve and each component as a fainter
dashed curve, with observed (and, distinctly, excluded) measurements
as error-barred points.

This is a "dumb" plotting function in the same sense as every other
function in this package (``plot_visual_orbit``, ``plot_residuals``,
``plot_binary_hrdiagram``, ...): it draws arrays it is given, and does
not itself interpolate atmospheres, apply extinction, or fit anything.
The model curves and observed-point wavelengths/fluxes are computed by
the caller -- see ``scripts/generate_benchmark_plots.py``'s
``generate_sed_fit_plots()`` for the canonical example, which uses
``photometry.model.synthetic_binary_sed()`` for the curves (a sibling
of the tested ``synthetic_binary_magnitudes()`` -- see that function's
own docstring for how the two are kept numerically self-consistent)
and each ``PhotometricPassband.pivot_wavelength`` for the point
placement.

Passband-label placement (2026-09-02, second pass)
------------------------------------------------------------
The first version of this function drew each ``observed_labels``/
``excluded_labels`` entry at a fixed offset (8 points straight above
its point, via ``Axes.annotate``). That collides for any two passbands
close enough in wavelength/flux to place their labels on top of each
other -- confirmed for real in the HD191854 figure, where "Stromgren
b"/"Johnson B" (~4400-4670 A) and "Johnson V"/"Tycho VT"/"Stromgren y"
(~5340-5500 A) overlapped into unreadable text, reported by the user
rather than caught in review.

Fixed with `adjustText <https://github.com/Phlya/adjustText>`_ (added
as a new pinned dependency, ``pyproject.toml``/``requirements.txt``),
an automatic label-collision-avoidance library rather than a manual
per-system offset scheme: every observed and excluded label is drawn
as a plain ``Axes.text`` at its point first, then ``adjust_text()`` is
called once (after every other plot element, including
``Axes.margins()``, is in place -- it needs final axis limits to judge
overlap correctly) to iteratively repel every label away from the
other labels *and* from the plotted points themselves, drawing a thin
leader line back to a label's original point whenever it had to move.
This generalizes to any number of systems/passbands without per-figure
tuning, which a hand-picked stagger scheme would not -- the three
near-twin-component systems (HD214222, HD130669, HD184467) are exactly
the ones with the most tightly clustered model curves and the most
labels competing for the same small region of the figure, so a
non-generalizing fix would have needed re-tuning per system.

``adjust_text()`` repels labels from each other and from the plotted
points, but has no notion of the model curves -- an observed point
sits ON or right next to the composite curve by construction, so its
label can still be repelled onto a visually busy stretch of that
curve. Every label is therefore also drawn with a 3.5-point white halo
(``matplotlib.patheffects.withStroke``, module-level ``_LABEL_HALO``)
at ``zorder=20`` (above every curve/point in this figure) so it stays
legible regardless of what's directly behind it, rather than trying to
make ``adjust_text`` curve-aware -- confirmed necessary, not
decorative: HD130669's "Stromgren y" sits over a fast-oscillating
stretch of the composite curve even after ``adjust_text`` moved it
clear of every other label, and a thinner 2.5-point halo (the first
value tried) still let that oscillation show through legibly enough to
matter; 3.5 points at zorder 20 does not.

See ``tests/test_sed_plot.py::test_plot_sed_fit_avoids_overlapping_labels``
for the regression test (asserts zero pairwise label bounding-box
overlap, not just "it ran"), and ``docs/project_decisions.md``'s dated
entry for the before/after visual comparison.
"""

from __future__ import annotations

import numpy as np

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.patheffects as patheffects

from adjustText import adjust_text


# A white halo behind every passband label -- adjust_text() (below)
# repels labels away from OTHER LABELS and from the plotted points,
# but has no notion of the model curves themselves, so a repelled
# label can still land on top of a busy part of the composite/
# component curves (observed points sit ON or near the model curve by
# construction, so their labels start out right next to it). A halo
# keeps a label legible in that case without needing curve-avoidance
# logic; see module docstring's "Passband-label placement" section.
_LABEL_HALO = [patheffects.withStroke(linewidth=3.5, foreground="white")]


def plot_sed_fit(
    wavelength,
    composite_flux,
    primary_flux,
    secondary_flux,
    observed_wavelength,
    observed_flux,
    observed_flux_uncertainty,
    observed_labels=None,
    excluded_wavelength=None,
    excluded_flux=None,
    excluded_flux_uncertainty=None,
    excluded_labels=None,
    title="Al-Wardat SED Fit",
    show=True,
    show_grid=True,
    output_format="png",
    save_path=None,
):
    """
    Plot a fitted Al-Wardat unresolved-binary SED against observed
    photometric points, following the visual convention of
    Al-Wardat (2002, 2012) and Abushattal et al. 2025 (see module
    docstring): flux density vs. wavelength, with the composite
    (two-component) model, each component's individual model, and
    the observed points all shown together.

    Parameters
    ----------
    wavelength : array_like
        Model wavelength grid, in Angstrom, common to
        ``composite_flux``/``primary_flux``/``secondary_flux``.

    composite_flux : array_like
        Model composite (unresolved, two-component, extincted)
        flux density at each ``wavelength``, in the same units as
        ``observed_flux`` (erg s^-1 cm^-2 Angstrom^-1, for the
        canonical caller -- this function itself is unit-agnostic
        as long as model and observed arrays share one convention).

    primary_flux, secondary_flux : array_like
        Each component's own individual model flux density (same
        grid, same units as ``composite_flux``), drawn as fainter
        dashed curves alongside the composite -- common practice
        in this literature to show how the two components combine,
        and explicitly requested for this figure.

    observed_wavelength, observed_flux, observed_flux_uncertainty : array_like
        Observed (fitted-against) photometric points, converted to
        flux density, at each passband's characteristic wavelength
        (e.g. ``PhotometricPassband.pivot_wavelength`` -- see module
        docstring). All three must be the same length.

    observed_labels : sequence of str, optional
        Per-point passband name, annotated next to each observed
        marker (e.g. "Johnson V"). If None, no per-point labels are
        drawn (the legend still identifies the series as a whole).
        Placement is collision-avoided automatically via ``adjustText``
        (see module docstring's "Passband-label placement" section) --
        labels are repelled apart from each other and from every
        plotted point, with a thin leader line drawn back to a label's
        point whenever it had to move to avoid another label.

    excluded_wavelength, excluded_flux, excluded_flux_uncertainty : array_like, optional
        Measurements that exist but were excluded from the fit
        (e.g. this codebase's Stromgren-u exclusion -- see
        ``docs/guide/alwardat_sed.md``'s "Stromgren u" section).
        Drawn as open, distinctly-colored markers so the reader can
        see they exist and were deliberately not fit, rather than
        being silently omitted from the figure the way they are
        already NOT silently dropped from the JSON record. All
        three must be given together, or all omitted.

    excluded_labels : sequence of str, optional
        Per-point passband name for the excluded points, drawn the
        same way as ``observed_labels`` -- and collision-avoided
        jointly with ``observed_labels`` in the same ``adjustText``
        pass, so an excluded label can't collide with an observed
        one either.

    title : str
        Plot title.

    show : bool, default=True
        If True, call ``plt.show()`` before returning (set False
        for non-interactive / batch generation, matching every
        other function in this package).

    show_grid : bool, default=True
        Whether to draw axes grid lines. Matches this package's
        existing default (grid on) for interactive/general use;
        the paper figure-generation pipeline
        (``scripts/generate_benchmark_plots.py``) overrides this to
        False, per this package's established PDF/no-grid
        convention for submitted figures.

    output_format : str, default="png"
        File format passed to ``Figure.savefig()`` (e.g. "png",
        "pdf", "svg"). Only takes effect when ``save_path`` is
        given.

    save_path : str or Path, optional
        If given, save the figure to this path (in
        ``output_format``) before returning.

    Returns
    -------
    (Figure, Axes)
        Matches ``plot_visual_orbit``/``plot_residuals``/
        ``plot_rv_orbit``'s return convention (the majority of this
        package's plotting functions), so this function unpacks the
        same way in ``scripts/generate_benchmark_plots.py``.
    """

    wavelength = np.asarray(wavelength, dtype=float)
    composite_flux = np.asarray(composite_flux, dtype=float)
    primary_flux = np.asarray(primary_flux, dtype=float)
    secondary_flux = np.asarray(secondary_flux, dtype=float)

    observed_wavelength = np.asarray(observed_wavelength, dtype=float)
    observed_flux = np.asarray(observed_flux, dtype=float)
    observed_flux_uncertainty = np.asarray(
        observed_flux_uncertainty, dtype=float
    )

    fig, ax = plt.subplots(figsize=(8.0, 5.5))

    # ------------------------------------------------------
    # Model curves: composite solid, components fainter/dashed
    # (Al-Wardat 2002/2012 and Abushattal et al. 2025's own
    # three-curve convention -- see module docstring).
    # ------------------------------------------------------

    ax.plot(
        wavelength,
        composite_flux,
        color="black",
        linewidth=1.8,
        linestyle="-",
        zorder=5,
        label="Composite model",
    )

    ax.plot(
        wavelength,
        primary_flux,
        color="tab:blue",
        linewidth=1.2,
        linestyle="--",
        alpha=0.75,
        zorder=4,
        label="Primary component",
    )

    ax.plot(
        wavelength,
        secondary_flux,
        color="tab:orange",
        linewidth=1.2,
        linestyle="--",
        alpha=0.75,
        zorder=4,
        label="Secondary component",
    )

    # ------------------------------------------------------
    # Observed (fitted) points. Label text objects are created but
    # NOT positioned yet -- collision-avoided placement happens in one
    # adjust_text() pass below, after every other plot element
    # (including axis margins) is final; see module docstring's
    # "Passband-label placement" section.
    # ------------------------------------------------------

    ax.errorbar(
        observed_wavelength,
        observed_flux,
        yerr=observed_flux_uncertainty,
        fmt="o",
        markersize=6,
        markerfacecolor="black",
        markeredgecolor="black",
        ecolor="black",
        capsize=3,
        linewidth=1.2,
        zorder=10,
        label="Observed (fitted)",
    )

    label_points_x = []
    label_points_y = []
    label_texts = []

    if observed_labels is not None:
        for x, y, label in zip(
            observed_wavelength, observed_flux, observed_labels
        ):
            text = ax.text(
                x, y, label, fontsize=7, ha="center", color="black",
                zorder=20,
            )
            text.set_path_effects(_LABEL_HALO)
            label_texts.append(text)
            label_points_x.append(x)
            label_points_y.append(y)

    # ------------------------------------------------------
    # Excluded points (disclosed, not silently dropped).
    # ------------------------------------------------------

    has_excluded = (
        excluded_wavelength is not None
        and excluded_flux is not None
        and excluded_flux_uncertainty is not None
        and len(excluded_wavelength) > 0
    )

    if has_excluded:

        excluded_wavelength = np.asarray(excluded_wavelength, dtype=float)
        excluded_flux = np.asarray(excluded_flux, dtype=float)
        excluded_flux_uncertainty = np.asarray(
            excluded_flux_uncertainty, dtype=float
        )

        ax.errorbar(
            excluded_wavelength,
            excluded_flux,
            yerr=excluded_flux_uncertainty,
            fmt="o",
            markersize=6,
            markerfacecolor="none",
            markeredgecolor="gray",
            ecolor="gray",
            capsize=3,
            linewidth=1.0,
            zorder=9,
            label="Excluded (not fit)",
        )

        if excluded_labels is not None:
            for x, y, label in zip(
                excluded_wavelength, excluded_flux, excluded_labels
            ):
                text = ax.text(
                    x, y, label, fontsize=7, ha="center", color="gray",
                    zorder=20,
                )
                text.set_path_effects(_LABEL_HALO)
                label_texts.append(text)
                label_points_x.append(x)
                label_points_y.append(y)

    ax.set_xlabel(r"Wavelength ($\mathrm{\AA}$)")

    ax.set_ylabel(
        r"$F_\lambda\ \left(\mathrm{erg\ s^{-1}\ cm^{-2}\ \AA^{-1}}\right)$"
    )

    ax.set_title(title)

    if show_grid:
        ax.grid(alpha=0.3)

    ax.legend(frameon=True, fontsize=8)

    ax.margins(x=0.03, y=0.10)

    # ------------------------------------------------------
    # Collision-avoided label placement -- run last, after every
    # other plot element (curves, points, margins) has fixed the
    # axes' final extent, since adjustText judges overlap against the
    # current rendered layout (see module docstring). Repels every
    # label away from every other label AND from every plotted point
    # (x=label_points_x, y=label_points_y), and draws a thin gray
    # leader line back to a label's original point whenever it had to
    # move.
    # ------------------------------------------------------

    if label_texts:
        adjust_text(
            label_texts,
            x=label_points_x,
            y=label_points_y,
            ax=ax,
            expand=(1.3, 1.6),
            force_text=(0.3, 0.5),
            force_static=(0.3, 0.5),
            arrowprops=dict(
                arrowstyle="-", color="gray", lw=0.6, alpha=0.7, shrinkB=2
            ),
        )

    if save_path is not None:
        fig.savefig(
            save_path,
            format=output_format,
            dpi=300,
            bbox_inches="tight",
        )

    if show:
        plt.show()

    return fig, ax
