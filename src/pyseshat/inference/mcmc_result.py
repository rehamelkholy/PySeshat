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

            ax.text(

                0.5,

                -0.9,

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