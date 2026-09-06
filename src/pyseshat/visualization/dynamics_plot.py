import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt


def plot_separation_vs_time(
    epochs,
    rho,
    title="Separation vs Time",
    show=True,
    show_grid=True,
    output_format="png",
    save_path=None
):
    """
    Plot projected angular separation (rho, in arcsec) against
    epoch for a visual binary or a triple's outer orbit.

    Parameters
    ----------
    epochs : array_like
        Epochs (decimal years), e.g. from
        ``analysis.dynamics.separation_vs_time()``'s ``"epoch"``
        key.
    rho : array_like
        Projected separation at each epoch (arcsec), same length
        as ``epochs``, e.g. that function's ``"rho"`` key.
    title : str
        Plot title.
    show : bool
        If True, call ``plt.show()`` before returning (set False
        for non-interactive / test use).
    show_grid : bool, default=True
        Whether to draw axes grid lines. Default preserves this
        function's original (always-on) behavior.
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
    (Figure, Axes)
        The plotted line's x/y data are exactly ``epochs``/``rho``
        as passed in (no resampling or smoothing) -- see
        ``tests/test_dynamics.py`` for a regression test asserting
        this.
    """

    fig, ax = plt.subplots()

    ax.plot(
        epochs,
        rho
    )

    ax.set_xlabel(
        "Epoch"
    )

    ax.set_ylabel(
        "Separation (arcsec)"
    )

    ax.set_title(
        title
    )

    if show_grid:
        ax.grid(True)

    if save_path is not None:
        fig.savefig(
            save_path,
            format=output_format,
            dpi=150,
            bbox_inches="tight"
        )

    if show:
        plt.show()

    return fig, ax