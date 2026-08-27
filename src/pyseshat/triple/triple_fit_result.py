from dataclasses import dataclass

from pyseshat.inference.base_fit_result import (
    BaseFitResult
)

from pyseshat.triple.triple_diagnostics import (
    triple_fit_summary
)

from pyseshat.triple.triple_statistics import (
    correlation_matrix
)

from pyseshat.triple.triple_identifiability import (
    covariance_condition_number,
    covariance_rank,
    is_degenerate
)

from pyseshat.triple.triple_pca import (
    covariance_pca,
    best_constrained_mode,
    worst_constrained_mode,
    variance_ratio
)


@dataclass
class TripleFitResult(BaseFitResult):

    # ======================================================
    # Summary statistics
    # ======================================================

    def summary(self):

        n_parameters = len(
            self.elements.fitted_parameter_names()
        )

        return triple_fit_summary(
            self.residuals,
            n_parameters
        )

    # ======================================================
    # Correlation matrix
    # ======================================================

    def correlations(self):

        return correlation_matrix(
            self.covariance
        )

    # ======================================================
    # Condition number
    # ======================================================

    def condition_number(self):

        return (
            covariance_condition_number(
                self.covariance
            )
        )

    # ======================================================
    # Effective rank
    # ======================================================

    def rank(self):

        return covariance_rank(
            self.covariance
        )

    # ======================================================
    # Degeneracy flag
    # ======================================================

    def is_degenerate(self):

        return is_degenerate(
            self.covariance
        )

    # ======================================================
    # Full text report
    # ======================================================

    def report(self):

        stats = self.summary()

        lines = []

        lines.append("")
        lines.append(
            "TRIPLE FIT REPORT"
        )

        lines.append(
            "=" * 80
        )

        lines.append("")
        lines.append(
            "Fitted Parameters"
        )

        lines.append(
            "-" * 80
        )

        for name in (
            self.elements.fitted_parameter_names()
        ):

            # --------------------------------------------------
            # Inner-orbit parameters
            #
            # ORBIT3 parameter names without a suffix refer
            # to the inner orbit.
            # --------------------------------------------------

            if name in (
                "P",
                "T0",
                "e",
                "a",
                "Omega",
                "omega",
                "i",
            ):

                value = getattr(
                    self.elements.inner,
                    name
                )

            # --------------------------------------------------
            # Inner spectroscopic / wobble parameters
            # --------------------------------------------------

            elif name in (
                "K1",
                "K2",
                "f",
            ):

                value = getattr(
                    self.elements,
                    name
                )

            # --------------------------------------------------
            # Outer-orbit parameters
            #
            # ORBIT3 represents these with the *2 suffix in
            # the parameter vector.
            # --------------------------------------------------

            elif name in (
                "P2",
                "T02",
                "e2",
                "a2",
                "Omega2",
                "omega2",
                "i2",
            ):

                outer_parameter_name = {
                    "P2": "P",
                    "T02": "T0",
                    "e2": "e",
                    "a2": "a",
                    "Omega2": "Omega",
                    "omega2": "omega",
                    "i2": "i",
                }[name]

                value = getattr(
                    self.elements.outer,
                    outer_parameter_name
                )

            # --------------------------------------------------
            # Outer spectroscopic parameters
            # --------------------------------------------------

            elif name in (
                "K3",
                "K4",
                "V0",
            ):

                value = getattr(
                    self.elements,
                    name
                )

            # --------------------------------------------------
            # Additional triple parameter retained by the
            # general TripleOrbitalElements representation.
            # --------------------------------------------------

            elif name == "B":

                value = self.elements.B

            else:

                raise ValueError(
                    "Unknown TripleFitResult parameter "
                    f"'{name}'."
                )

            sigma = (
                self.uncertainties.get(
                    name,
                    float("nan")
                )
            )

            lines.append(
                f"{name:<10}"
                f"{value:>15.8f}"
                f" +/- {sigma:.8f}"
            )

        lines.append("")
        lines.append(
            "Fit Statistics"
        )

        lines.append(
            "-" * 80
        )

        lines.append(
            f"Chi-square         : "
            f"{stats['chi_square']:.6f}"
        )

        lines.append(
            f"Reduced Chi-square : "
            f"{stats['reduced_chi_square']:.6f}"
        )

        lines.append(
            f"RMS Residual       : "
            f"{stats['rms']:.6f}"
        )

        lines.append("")
        lines.append(
            "Identifiability"
        )

        lines.append(
            "-" * 80
        )

        lines.append(
            f"Condition Number : "
            f"{self.condition_number():.6e}"
        )

        lines.append(
            f"Effective Rank   : "
            f"{self.rank()}"
        )

        lines.append(
            f"Degenerate       : "
            f"{self.is_degenerate()}"
        )

        lines.append("")

        lines.append(
            "Correlation Matrix"
        )

        lines.append(
            "-" * 80
        )

        corr = self.correlations()

        names = (
            self.elements.fitted_parameter_names()
        )

        header = (
            " " * 12
            +
            "".join(
                f"{n:>12}"
                for n in names
            )
        )

        lines.append(
            header
        )

        for name, row in zip(
            names,
            corr
        ):

            lines.append(
                f"{name:<12}"
                +
                "".join(
                    f"{v:12.4f}"
                    for v in row
                )
            )

        lines.append("")
        lines.append(
            "Optimizer"
        )

        lines.append(
            "-" * 80
        )

        lines.append(
            f"Success : "
            f"{self.optimizer_result.success}"
        )

        lines.append(
            self.optimizer_result.message
        )

        return "\n".join(
            lines
        )

    # ======================================================
    # PCA
    # ======================================================

    def pca(self):

        return covariance_pca(
            self.covariance
        )

    # ======================================================
    # Best constrained mode
    # ======================================================

    def best_mode(self):

        return best_constrained_mode(
            self.covariance
        )

    # ======================================================
    # Worst constrained mode
    # ======================================================

    def worst_mode(self):

        return worst_constrained_mode(
            self.covariance
        )

    # ======================================================
    # Variance ratio
    # ======================================================

    def variance_ratio(self):

        return variance_ratio(
            self.covariance
        )