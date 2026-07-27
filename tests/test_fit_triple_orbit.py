from pathlib import Path

import numpy as np

from pyseshat.triple.triple_fit import fit_triple_orbit
from pyseshat.triple.triple_fit_result import TripleFitResult
from pyseshat.triple.triple_parser import parse_triple_file
from pyseshat.triple.triple_residuals import triple_visual_residuals


def load_benchmark():

    filename = (
        Path(__file__).resolve().parents[1]
        / "validation"
        / "benchmark_systems"
        / "ZetaAqr"
        / "ZetaAqr.in2"
    )

    parsed = parse_triple_file(
        filename
    )

    return (
        parsed["elements"],
        parsed["observations"],
    )


def test_fit_triple_orbit():

    elements, observations = load_benchmark()

    initial = np.sum(
        triple_visual_residuals(
            observations,
            elements,
        ) ** 2
    )

    result = fit_triple_orbit(
        elements,
        observations,
    )

    assert isinstance(
        result,
        TripleFitResult,
    )

    final = np.sum(
        triple_visual_residuals(
            observations,
            result.elements,
        ) ** 2
    )

    assert final <= initial


def test_optimizer_success_flag():

    elements, observations = load_benchmark()

    result = fit_triple_orbit(
        elements,
        observations,
    )

    assert isinstance(
        result.optimizer_result.success,
        bool,
    )


def test_summary_runs():

    elements, observations = load_benchmark()

    result = fit_triple_orbit(
        elements,
        observations,
    )

    summary = result.summary()

    assert isinstance(
        summary,
        dict,
    )

    assert "chi_square" in summary


def test_report_runs():

    elements, observations = load_benchmark()

    result = fit_triple_orbit(
        elements,
        observations,
    )

    report = result.report()

    assert isinstance(
        report,
        str,
    )

    assert "TRIPLE FIT REPORT" in report