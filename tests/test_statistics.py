"""
Regression tests for `analysis.statistics` against combined
visual(+RV) benchmark fits.

This file used to be a print-only script (module-level code with no
`assert`s, 0 items collected by pytest) that ran a fit for each
benchmark system and printed every statistic for manual inspection.
Rewritten to assert the actual invariants those printouts were
implicitly checking by eye.
"""

from pathlib import Path

import numpy as np

import pytest

from pyseshat.io.orbit_parser import (
    parse_orbit_file,
)

from pyseshat.inference.optimizer import (
    fit_visual_orbit,
)

from pyseshat.inference.likelihood import (
    visual_xy_residuals,
    rv_residuals,
)

from pyseshat.analysis.statistics import (
    chi_square,
    reduced_chi_square,
    rms_residual,
    fit_summary,
    combined_fit_summary,
    parameter_comparison,
)


ROOT = Path("validation/benchmark_systems")

BENCHMARKS = [
    ROOT / "FIN379" / "FIN379.inp",
    ROOT / "GL765" / "GL765.inp",
    ROOT / "ZetaAqr" / "ZetaAqr.in2",
]


def _fit_benchmark(filepath):

    parsed = parse_orbit_file(str(filepath))

    elements = parsed["elements"]

    visual_data = parsed["visual_data"]

    rv_data = parsed["rv_data"]

    t = visual_data[:, 0]

    theta = np.deg2rad(visual_data[:, 1])

    rho = visual_data[:, 2]

    sigma = visual_data[:, 3]

    X = rho * np.cos(theta)

    Y = rho * np.sin(theta)

    fit_result = fit_visual_orbit(
        elements, t, X, Y, sigma, sigma, rv_data=rv_data
    )

    return elements, fit_result, t, X, Y, sigma, rv_data


@pytest.mark.parametrize(
    "filepath",
    BENCHMARKS,
    ids=[f.name for f in BENCHMARKS],
)
def test_basic_statistics_are_internally_consistent(filepath):

    _, fit_result, *_ = _fit_benchmark(filepath)

    fitted = fit_result.elements

    residuals = fit_result.residuals

    n_fit = len(fitted.fitted_parameters())

    chi2 = chi_square(residuals)

    assert chi2 == pytest.approx(np.sum(residuals**2))

    redchi2 = reduced_chi_square(residuals, n_fit)

    assert redchi2 == pytest.approx(
        chi2 / (len(residuals) - n_fit)
    )

    rms = rms_residual(residuals)

    assert rms == pytest.approx(
        np.sqrt(np.mean(residuals**2))
    )

    summary = fit_summary(residuals, fitted)

    assert summary["chi_square"] == pytest.approx(chi2)

    assert summary["reduced_chi_square"] == pytest.approx(redchi2)

    assert summary["rms"] == pytest.approx(rms)


@pytest.mark.parametrize(
    "filepath",
    BENCHMARKS,
    ids=[f.name for f in BENCHMARKS],
)
def test_combined_fit_summary_matches_component_residuals(filepath):

    elements, fit_result, t, X, Y, sigma, rv_data = _fit_benchmark(
        filepath
    )

    fitted = fit_result.elements

    n_params = len(fitted.fitted_parameters())

    visual_res = visual_xy_residuals(
        fitted.fitted_parameters(), fitted, t, X, Y, sigma, sigma
    )

    if len(rv_data) > 0:

        rv_res = rv_residuals(
            fitted.fitted_parameters(), fitted, rv_data
        )

        combined = combined_fit_summary(
            visual_residuals=visual_res,
            rv_residuals=rv_res,
            rv_components=rv_data[:, 3],
            n_params=n_params,
        )

        expected_chi2_total = (
            np.sum(visual_res**2) + np.sum(rv_res**2)
        )

        expected_n_total = len(visual_res) + len(rv_res)

        assert combined["chi2_rv1"] + combined[
            "chi2_rv2"
        ] == pytest.approx(np.sum(rv_res**2))

    else:

        combined = combined_fit_summary(
            visual_residuals=visual_res,
            rv_residuals=None,
            rv_components=None,
            n_params=n_params,
        )

        expected_chi2_total = np.sum(visual_res**2)

        expected_n_total = len(visual_res)

    assert combined["chi2_visual"] == pytest.approx(
        np.sum(visual_res**2)
    )

    assert combined["chi2_total"] == pytest.approx(
        expected_chi2_total
    )

    expected_dof = expected_n_total - n_params

    assert expected_dof > 0

    assert combined["reduced_chi2"] == pytest.approx(
        expected_chi2_total / expected_dof
    )

    assert combined["rms"] == pytest.approx(
        np.sqrt(expected_chi2_total / expected_n_total)
    )


@pytest.mark.parametrize(
    "filepath",
    BENCHMARKS,
    ids=[f.name for f in BENCHMARKS],
)
def test_parameter_comparison_covers_all_elements(filepath):

    elements, fit_result, *_ = _fit_benchmark(filepath)

    fitted = fit_result.elements

    comparison = parameter_comparison(elements, fitted)

    assert set(comparison.keys()) == set(
        elements.PARAMETER_ORDER
    )

    for name in elements.PARAMETER_ORDER:

        entry = comparison[name]

        assert entry["initial"] == pytest.approx(
            getattr(elements, name)
        )

        assert entry["fitted"] == pytest.approx(
            getattr(fitted, name)
        )
