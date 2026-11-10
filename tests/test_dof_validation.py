"""
Regression guard for degrees-of-freedom handling.

Background
----------
An earlier audit flagged that `analysis.statistics.combined_fit_summary` and
`inference.covariance.estimate_covariance`'s `scale_by_reduced_chi2`
branch both silently floored a degenerate/over-parameterized fit's
degrees of freedom via `dof = max(n - n_params, 1)`, returning a
well-formed-looking but meaningless reduced chi-square/covariance
scale instead of failing loudly -- inconsistent with the sibling
`analysis.statistics.reduced_chi_square`, which correctly raises.

Both were fixed to raise `ValueError` instead of clamping. This test
makes that fix permanent.
"""

import numpy as np

import pytest

from pyseshat.analysis.statistics import (
    combined_fit_summary,
)

from pyseshat.inference.covariance import (
    estimate_covariance,
)


# ==========================================================
# combined_fit_summary
# ==========================================================

def test_combined_fit_summary_visual_only_raises_on_nonpositive_dof():

    visual_residuals = np.array([1.0, -1.0, 0.5])

    with pytest.raises(ValueError):

        combined_fit_summary(
            visual_residuals=visual_residuals,
            n_params=len(visual_residuals),
        )


def test_combined_fit_summary_combined_raises_on_nonpositive_dof():

    visual_residuals = np.array([1.0, -1.0])

    rv_residuals = np.array([0.5, -0.5])

    rv_components = np.array([0, 1])

    with pytest.raises(ValueError):

        combined_fit_summary(
            visual_residuals=visual_residuals,
            rv_residuals=rv_residuals,
            rv_components=rv_components,
            n_params=(
                len(visual_residuals)
                + len(rv_residuals)
            ),
        )


def test_combined_fit_summary_still_succeeds_with_positive_dof():

    visual_residuals = np.array([1.0, -1.0, 0.5, -0.5, 0.25])

    summary = combined_fit_summary(
        visual_residuals=visual_residuals,
        n_params=2,
    )

    assert summary["reduced_chi2"] == pytest.approx(
        np.sum(visual_residuals**2) / 3
    )


# ==========================================================
# estimate_covariance(scale_by_reduced_chi2=True)
# ==========================================================

class _DummyOptimizerResult:

    def __init__(self, jac):
        self.jac = jac


def test_estimate_covariance_scaled_raises_on_nonpositive_dof():

    # 2 residuals, 2 fitted parameters -> dof = 0
    jac = np.array([
        [1.0, 0.0],
        [0.0, 1.0],
    ])

    residuals = np.array([0.1, -0.1])

    result = _DummyOptimizerResult(jac)

    with pytest.raises(ValueError):

        estimate_covariance(
            result,
            residuals,
            scale_by_reduced_chi2=True,
        )


def test_estimate_covariance_scaled_succeeds_with_positive_dof():

    jac = np.array([
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 1.0],
    ])

    residuals = np.array([0.1, -0.1, 0.05])

    result = _DummyOptimizerResult(jac)

    covariance = estimate_covariance(
        result,
        residuals,
        scale_by_reduced_chi2=True,
    )

    assert covariance.shape == (2, 2)

    assert np.all(np.isfinite(covariance))
