from dataclasses import dataclass
from matplotlib.ticker import ScalarFormatter

import numpy as np
import corner

from pyseshat.inference.base_fit_result import (
    BaseFitResult
)


@dataclass
class MCMCResult(BaseFitResult):
    """
    Container for Bayesian orbit-fitting results.

    In addition to the best-fit solution inherited from
    BaseFitResult, this class stores the posterior samples
    and MCMC diagnostics.
    """

    # ------------------------------------------------------
    # Posterior samples
    # ------------------------------------------------------

    samples: np.ndarray

    log_probability: np.ndarray

    parameter_names: list[str]

    # ------------------------------------------------------
    # Sampling diagnostics
    # ------------------------------------------------------

    acceptance_fraction: float | None = None

    autocorrelation_time: np.ndarray | None = None

    n_walkers: int = 0

    n_steps: int = 0

    burn_in: int = 0

    thin: int = 1

    # ------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------

    @property
    def n_samples(self):
        """
        Number of posterior samples.
        """

        return self.samples.shape[0]

    @property
    def n_parameters(self):
        """
        Number of fitted parameters.
        """

        return self.samples.shape[1]
    
    @property
    def posterior_median(self):
        """
        Posterior median for every fitted parameter.
        """

        return np.median(
            self.samples,
            axis=0
        )


    @property
    def credible_interval_16(self):
        """
        Lower 68% credible limit.
        """

        return np.percentile(
            self.samples,
            16,
            axis=0
        )


    @property
    def credible_interval_84(self):
        """
        Upper 68% credible limit.
        """

        return np.percentile(
            self.samples,
            84,
            axis=0
        )


    @property
    def posterior_std(self):
        """
        Posterior standard deviation.
        """

        return np.std(
            self.samples,
            axis=0,
            ddof=1
        )


    @property
    def maximum_posterior_index(self):
        """
        Index of the highest-posterior sample.
        """

        return np.argmax(
            self.log_probability
        )


    @property
    def maximum_posterior_sample(self):
        """
        Maximum-posterior sample.
        """

        return self.samples[
            self.maximum_posterior_index
        ]
    
    @property
    def posterior_summary(self):
        """
        Posterior summary statistics.

        Returns
        -------
        list[dict]
            One dictionary per fitted parameter containing:

            - name
            - median
            - lower_68
            - upper_68
            - std
            - map
        """

        summary = []

        medians = self.posterior_median

        lower = self.credible_interval_16

        upper = self.credible_interval_84

        std = self.posterior_std

        map_sample = self.maximum_posterior_sample

        for i, name in enumerate(
            self.parameter_names
        ):

            summary.append({

                "name": name,

                "median": medians[i],

                "lower_68": lower[i],

                "upper_68": upper[i],

                "std": std[i],

                "map": map_sample[i]

            })

        return summary
    
    # ======================================================
    # Full text report
    # ======================================================

    def report(self):
        """
        Generate a formatted Bayesian fit report.

        Returns
        -------
        str
        """

        lines = []

        lines.append("")
        lines.append(
            "BAYESIAN FIT REPORT"
        )

        lines.append(
            "=" * 80
        )

        # --------------------------------------------------
        # Posterior summary
        # --------------------------------------------------

        lines.append("")
        lines.append(
            "Posterior Summary"
        )

        lines.append(
            "-" * 80
        )

        lines.append(

            f"{'Parameter':<12}"
            f"{'Median':>15}"
            f"{'-1σ':>15}"
            f"{'+1σ':>15}"

        )

        for item in self.posterior_summary:

            lower = (
                item["median"]
                -
                item["lower_68"]
            )

            upper = (
                item["upper_68"]
                -
                item["median"]
            )

            lines.append(

                f"{item['name']:<12}"

                f"{item['median']:15.8f}"

                f"{lower:15.8f}"

                f"{upper:15.8f}"

            )

        # --------------------------------------------------
        # Diagnostics
        # --------------------------------------------------

        lines.append("")
        lines.append(
            "Sampling Diagnostics"
        )

        lines.append(
            "-" * 80
        )

        lines.append(

            f"Acceptance Fraction : "
            f"{self.acceptance_fraction:.4f}"

        )

        if self.has_autocorrelation_time:

            lines.append(

                f"Minimum ESS         : "
                f"{self.minimum_effective_sample_size:.1f}"

            )

            lines.append(

                f"Converged          : "
                f"{self.converged}"

            )

        else:

            lines.append(
                "Autocorrelation    : Not available"
            )

            lines.append(
                "Minimum ESS        : Not available"
            )

            lines.append(
                "Converged          : False"
            )

        # --------------------------------------------------
        # MAP solution
        # --------------------------------------------------

        lines.append("")
        lines.append(
            "Maximum A Posteriori Solution"
        )

        lines.append(
            "-" * 80
        )

        for item in self.posterior_summary:

            lines.append(

                f"{item['name']:<12}"
                f"{item['map']:15.8f}"

            )

        return "\n".join(
            lines
        )
    
    # ======================================================
    # Corner plot
    # ======================================================
    def corner_plot(
        self,
        parameters=None,
        show_titles=True,
        truths=None,
        max_samples=10000,
        random_seed=42,
        show_grid=False,
        output_format="png",
        save_path=None,
        **kwargs
    ):
        """
        Generate a publication-quality corner plot of the
        posterior samples.

        Parameters
        ----------
        parameters : sequence of str, optional
            Subset of parameters to display.
            If None, all fitted parameters are shown.

        show_titles : bool, default=True
            Display posterior medians and 68% credible
            intervals above the diagonal panels.

        truths : sequence of float, optional
            Reference parameter values to overplot.
            If None, posterior medians are used.

        max_samples : int, default=10000
            Maximum number of posterior samples plotted.
            Larger sample sets are randomly subsampled.

        random_seed : int, default=42
            Random seed used for plotting subsamples.

        show_grid : bool, default=False
            Whether to draw axes grid lines on each panel.
            Unlike every other plotting function in this
            codebase, ``corner.corner()`` draws NO grid lines
            by default, and this function has never added any.
            To genuinely preserve this function's *original*
            behavior (the same "no existing call site breaks"
            principle applied to every other plotting function),
            this default is ``False`` here rather than ``True`` --
            setting it to ``True`` would silently change the
            appearance of every corner plot ever produced by this
            function. Pass ``show_grid=True`` explicitly to opt
            into grid lines.

        output_format : str, default="png"
            File format passed to ``Figure.savefig()`` (e.g.
            "png", "pdf", "svg"). Only takes effect when
            ``save_path`` is given.

        save_path : str or Path, optional
            If given, save the figure to this path (in
            ``output_format``) before returning. If None
            (default), matches this function's original behavior
            of not saving internally -- the caller retrieves the
            returned ``Figure`` and saves it externally.

        **kwargs
            Additional keyword arguments passed to
            corner.corner(). User-supplied values override
            the defaults defined here.

        Returns
        -------
        matplotlib.figure.Figure
        """

        # ======================================================
        # Select parameters
        # ======================================================

        all_names = list(
            self.parameter_names
        )

        if parameters is None:

            indices = list(
                range(
                    len(all_names)
                )
            )

            names = all_names

        else:

            names = list(
                parameters
            )

            unknown = [

                name

                for name in names

                if name not in all_names

            ]

            if unknown:

                raise ValueError(

                    "Unknown posterior parameter(s): "

                    + ", ".join(
                        unknown
                    )

                )

            indices = [

                all_names.index(
                    name
                )

                for name in names

            ]

        samples = np.asarray(

            self.samples[
                :,
                indices
            ],

            dtype=float

        )

        # ======================================================
        # Validate samples
        # ======================================================

        if samples.ndim != 2:

            raise ValueError(

                "Posterior samples must be a "
                "two-dimensional array."

            )

        if samples.shape[0] == 0:

            raise ValueError(

                "Posterior samples are empty."

            )

        finite = np.all(

            np.isfinite(
                samples
            ),

            axis=1

        )

        samples = samples[
            finite
        ]

        if samples.shape[0] == 0:

            raise ValueError(

                "Posterior samples contain no "
                "fully finite rows."

            )

        # ======================================================
        # Prepare reference values
        # ======================================================

        if truths is not None:

            truths = np.asarray(

                truths,

                dtype=float

            )

            if truths.ndim != 1:

                raise ValueError(

                    "truths must be a "
                    "one-dimensional sequence."

                )

            if len(truths) == len(
                all_names
            ):

                truths = truths[
                    indices
                ]

            elif len(truths) != len(
                names
            ):

                raise ValueError(

                    "truths must contain either one "
                    "value per fitted parameter or "
                    "one value per displayed parameter."

                )

        # ======================================================
        # Posterior summaries
        # ======================================================

        q16, q50, q84 = np.percentile(

            samples,

            [
                16.0,
                50.0,
                84.0
            ],

            axis=0

        )

        if truths is None:

            truths = q50

        # ======================================================
        # Subsample for plotting only
        # ======================================================

        plot_samples = samples

        if (

            max_samples is not None

            and

            max_samples > 0

            and

            len(samples) > max_samples

        ):

            rng = np.random.default_rng(

                random_seed

            )

            selected = rng.choice(

                len(samples),

                size=max_samples,

                replace=False

            )

            plot_samples = samples[
                selected
            ]

        # ======================================================
        # Publication-quality parameter labels
        # ======================================================

        label_map = {

            "P": r"$P$",

            "T0": r"$T_0$",

            "e": r"$e$",

            "a": r"$a$",

            "Omega": r"$\Omega$",

            "omega": r"$\omega$",

            "i": r"$i$",

            "K1": r"$K_1$",

            "K2": r"$K_2$",

            "V0": r"$V_0$",

            "gamma": r"$\gamma$",

            "f": r"$f$"

        }

        labels = [

            label_map.get(
                name,
                name
            )

            for name in names

        ]

        # ======================================================
        # Corner defaults
        # ======================================================

        defaults = {

            "labels": labels,

            "truths": truths,

            "show_titles": show_titles,

            "title_quantiles": [
                0.16,
                0.50,
                0.84
            ],

            "title_fmt": ".4f",

            "title_kwargs": {

                "fontsize": 9,

                "pad": 7

            },

            "label_kwargs": {

                "fontsize": 11

            },

            "bins": 35,

            "max_n_ticks": 3,

            "smooth": 1.0,

            "smooth1d": 1.0,

            "quantiles": None,

            "levels": (

                0.393,

                0.865

            ),

            "fill_contours": True,

            "plot_density": True,

            "plot_datapoints": True,

            "hist_kwargs": {

                "linewidth": 1.2

            }

        }

        defaults.update(
            kwargs
        )

        # ======================================================
        # Generate plot
        # ======================================================

        fig = corner.corner(

            plot_samples,

            **defaults

        )

        ndim = len(
            names
        )

        axes = np.asarray(

            fig.axes,

            dtype=object

        ).reshape(

            ndim,

            ndim

        )

        # ======================================================
        # Figure geometry
        # ======================================================

        panel_size = 1.65

        fig.set_size_inches(

            panel_size * ndim
            + 2.8,

            panel_size * ndim
            + 3.6,

            forward=True

        )

        # Reserve real space outside the corner matrix:
        #
        # top    -> diagonal posterior titles
        # bottom -> rotated ticks and parameter labels
        # left   -> rotated ticks and parameter labels

        fig.subplots_adjust(

            left=0.07,

            right=0.985,

            bottom=0.12,

            top=0.955,

            wspace=0.055,

            hspace=0.055

        )

        # ======================================================
        # Format all visible axes
        # ======================================================

        for row in range(
            ndim
        ):

            for column in range(
                ndim
            ):

                ax = axes[
                    row,
                    column
                ]

                if not ax.get_visible():

                    continue

                ax.tick_params(

                    axis="both",

                    which="major",

                    labelsize=7,

                    pad=2

                )

        # ======================================================
        # Format bottom-row x axes
        # ======================================================

        for column in range(
            ndim
        ):

            ax = axes[
                ndim - 1,
                column
            ]

            # Remove corner's original x-axis label.
            #
            # The replacement label is positioned explicitly
            # below the rotated tick labels.

            ax.set_xlabel("")

            for tick in ax.get_xticklabels():

                tick.set_rotation(
                    45
                )

                tick.set_horizontalalignment(
                    "right"
                )

                tick.set_verticalalignment(
                    "top"
                )

                tick.set_rotation_mode(
                    "anchor"
                )

        # ======================================================
        # Disable offset notation for T0
        # ======================================================

        if "T0" in names:

            t0_index = names.index(
                "T0"
            )

            # --------------------------------------------------
            # Visible bottom T0 x-axis
            # --------------------------------------------------

            bottom_ax = axes[
                ndim - 1,
                t0_index
            ]

            x_formatter = ScalarFormatter(

                useOffset=False

            )

            x_formatter.set_scientific(

                False

            )

            bottom_ax.xaxis.set_major_formatter(

                x_formatter

            )

            bottom_ax.xaxis.get_offset_text().set_visible(

                False

            )

            # --------------------------------------------------
            # Visible T0 y-axis
            # --------------------------------------------------

            if t0_index > 0:

                left_ax = axes[
                    t0_index,
                    0
                ]

                y_formatter = ScalarFormatter(

                    useOffset=False

                )

                y_formatter.set_scientific(

                    False

                )

                left_ax.yaxis.set_major_formatter(

                    y_formatter

                )

                left_ax.yaxis.get_offset_text().set_visible(

                    False

                )

            # --------------------------------------------------
            # T0 y-axes
            # --------------------------------------------------

            for column in range(
                t0_index
            ):

                ax = axes[
                    t0_index,
                    column
                ]

                formatter = ScalarFormatter(

                    useOffset=False

                )

                formatter.set_scientific(
                    False
                )

                ax.yaxis.set_major_formatter(

                    formatter

                )

                ax.yaxis.get_offset_text().set_visible(

                    False

                )

        # ======================================================
        # Place bottom-row parameter labels
        #
        # corner's own x-axis label was cleared above (see that
        # loop's comment) and is replaced by an explicitly
        # positioned ``ax.text`` call, because corner.corner()'s
        # own label placement collides with the rotated tick
        # labels this function adds.
        #
        # The vertical offset is MEASURED, not a fixed guess: a
        # hardcoded axes-fraction offset (this function's original
        # implementation used -0.9) is either too small for long
        # tick-label strings or -- as was the case here -- much
        # larger than necessary for short ones, leaving a large,
        # inconsistent gap between the tick numbers and their
        # parameter label. Instead, after every tick/formatter
        # change above (including the T0 offset-notation fix,
        # which can change that column's tick-label text/width) is
        # already applied, this draws the figure once so
        # Matplotlib has computed each rotated tick label's actual
        # on-screen extent, finds the lowest (most negative, in
        # axes-fraction y) point among them per column, and places
        # the parameter label a small, fixed gap below THAT --
        # giving a consistent, tight, non-overlapping gap
        # regardless of tick-label length, font, or figure size.
        # ======================================================

        fig.canvas.draw()

        renderer = fig.canvas.get_renderer()

        # Small fixed clearance, in axes-fraction units, between
        # the bottom-most tick-label pixel and the parameter label
        # above it -- NOT the whole gap (which is dominated by the
        # tick labels' own measured extent).
        label_clearance = 0.05

        for column in range(
            ndim
        ):

            ax = axes[
                ndim - 1,
                column
            ]

            inv = ax.transAxes.inverted()

            tick_bottoms_axes_frac = []

            for tick in ax.get_xticklabels():

                bbox = tick.get_window_extent(
                    renderer=renderer
                )

                # Bottom-left corner of the tick label's pixel
                # bbox, converted into this axes' fraction
                # coordinates (0 = axis bottom, negative = below
                # the axis).

                _, y0_axes = inv.transform(
                    (bbox.x0, bbox.y0)
                )

                tick_bottoms_axes_frac.append(
                    y0_axes
                )

            if tick_bottoms_axes_frac:

                lowest = min(
                    tick_bottoms_axes_frac
                )

            else:

                # No tick labels to measure (shouldn't happen for
                # a visible bottom-row panel, but fall back to a
                # small, safe default rather than raising).

                lowest = -0.3

            ax.text(

                0.5,

                lowest - label_clearance,

                labels[
                    column
                ],

                transform=ax.transAxes,

                horizontalalignment="center",

                verticalalignment="top",

                fontsize=11,

                clip_on=False

            )

        # ======================================================
        # Format first-column y axes
        # ======================================================

        for row in range(
            1,
            ndim
        ):

            ax = axes[
                row,
                0
            ]

            ax.set_ylabel(

                labels[
                    row
                ],

                fontsize=11,

                labelpad=24

            )

            for tick in ax.get_yticklabels():

                tick.set_rotation(
                    45
                )

                tick.set_horizontalalignment(
                    "right"
                )

                tick.set_verticalalignment(
                    "center"
                )

                tick.set_rotation_mode(
                    "anchor"
                )

        # ======================================================
        # Preserve and format diagonal titles
        # ======================================================

        if show_titles:

            for index in range(
                ndim
            ):

                title = axes[
                    index,
                    index
                ].title

                title.set_fontsize(
                    9
                )

                title.set_y(
                    1.10
                )

        # ======================================================
        # Grid lines (opt-in -- see show_grid's docstring above:
        # corner.corner() draws none by default, and this
        # function never added any, so this loop is purely
        # additive and gated behind the flag).
        # ======================================================

        if show_grid:

            for row in range(
                ndim
            ):

                for column in range(
                    row + 1
                ):

                    axes[
                        row,
                        column
                    ].grid(
                        True
                    )

        # ======================================================
        # Save to disk
        # ======================================================

        if save_path is not None:

            fig.savefig(
                save_path,
                format=output_format,
                dpi=150,
                bbox_inches="tight"
            )

        return fig

    @property
    def has_autocorrelation_time(self):
        """
        Whether an autocorrelation estimate is available.
        """

        return self.autocorrelation_time is not None


    @property
    def post_burn_in_steps(self):
        """
        Number of retained MCMC steps per walker
        before thinning.
        """

        return max(
            self.n_steps - self.burn_in,
            0
        )


    @property
    def retained_steps(self):
        """
        Number of retained steps per walker
        after thinning.
        """

        if self.post_burn_in_steps == 0:
            return 0

        return (
            self.post_burn_in_steps
            //
            self.thin
        )


    @property
    def effective_sample_size(self):
        """
        Effective sample size for each parameter.
        """

        if not self.has_autocorrelation_time:
            return None

        tau = np.asarray(
            self.autocorrelation_time,
            dtype=float
        )

        return (
            self.n_walkers
            *
            self.retained_steps
            /
            tau
        )


    @property
    def minimum_effective_sample_size(self):
        """
        Smallest effective sample size.
        """

        ess = self.effective_sample_size

        if ess is None:
            return None

        return float(
            np.min(ess)
        )


    @property
    def converged(self):
        """
        Simple convergence criterion.

        emcee recommends chains longer than
        approximately 50 autocorrelation times.
        """

        if not self.has_autocorrelation_time:
            return False

        tau_max = np.max(
            self.autocorrelation_time
        )

        return (
            self.post_burn_in_steps
            >=
            50.0 * tau_max
        )