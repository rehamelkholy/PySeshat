from pathlib import Path

import numpy as np

from pyseshat.io.orbit_parser import (
    parse_orbit_file,
)


# ==========================================================
# Benchmark datasets
# ==========================================================

ROOT = Path("validation/benchmark_systems")


# ==========================================================
# Basic parser smoke tests
# ==========================================================

def test_parse_fin379_benchmark():

    filepath = (
        ROOT
        / "FIN379"
        / "FIN379.inp"
    )

    parsed = parse_orbit_file(
        str(filepath)
    )

    assert "elements" in parsed
    assert "visual_data" in parsed
    assert "rv_data" in parsed
    assert "metadata" in parsed

    assert parsed["visual_data"].ndim == 2
    assert parsed["visual_data"].shape[1] == 4

    assert parsed["rv_data"].ndim == 2
    assert parsed["rv_data"].shape[1] == 4

    assert np.all(
        np.isfinite(
            parsed["visual_data"]
        )
    )

    assert np.all(
        np.isfinite(
            parsed["rv_data"]
        )
    )


def test_parse_gl765_benchmark():

    filepath = (
        ROOT
        / "GL765"
        / "GL765.inp"
    )

    parsed = parse_orbit_file(
        str(filepath)
    )

    assert "elements" in parsed
    assert "visual_data" in parsed
    assert "rv_data" in parsed
    assert "metadata" in parsed

    visual_data = (
        parsed["visual_data"]
    )

    rv_data = (
        parsed["rv_data"]
    )

    assert visual_data.ndim == 2
    assert visual_data.shape[1] == 4

    assert rv_data.ndim == 2
    assert rv_data.shape[1] == 4

    assert len(
        visual_data
    ) > 0

    assert len(
        rv_data
    ) > 0

    assert np.all(
        np.isfinite(
            visual_data
        )
    )

    assert np.all(
        np.isfinite(
            rv_data
        )
    )


def test_parse_zeta_aqr_benchmark():

    filepath = (
        ROOT
        / "ZetaAqr"
        / "ZetaAqr.in2"
    )

    parsed = parse_orbit_file(
        str(filepath)
    )

    assert "elements" in parsed
    assert "visual_data" in parsed
    assert "rv_data" in parsed
    assert "metadata" in parsed

    visual_data = (
        parsed["visual_data"]
    )

    assert visual_data.ndim == 2
    assert visual_data.shape[1] == 4

    assert np.all(
        np.isfinite(
            visual_data
        )
    )


# ==========================================================
# RV component parsing
# ==========================================================

def test_rv_component_marker_can_occur_after_extra_columns(
    tmp_path
):

    filename = (
        tmp_path
        / "example.inp"
    )

    filename.write_text(
        """
Object: TEST

P 10.0
T 2000.0
e 0.2
a 1.0
W 30.0
w 60.0
i 70.0

46632.364 -6.37 0.44 176 12.32 3.18 Va
46640.364 -5.21 0.51 176 12.11 3.20 Vb
""".strip()
        + "\n"
    )

    parsed = parse_orbit_file(
        str(filename)
    )

    rv_data = (
        parsed["rv_data"]
    )

    assert rv_data.shape == (
        2,
        4,
    )

    # The parser converts JD-2400000 epochs into
    # decimal years.

    np.testing.assert_allclose(
        rv_data[:, 0],
        np.array([
            1986.5499356605067,
            1986.5718384668037,
        ]),
    )

    np.testing.assert_allclose(
        rv_data[:, 1],
        np.array([
            -6.37,
            -5.21,
        ]),
    )

    np.testing.assert_allclose(
        rv_data[:, 2],
        np.array([
            0.44,
            0.51,
        ]),
    )

    np.testing.assert_array_equal(
        rv_data[:, 3],
        np.array([
            0,
            1,
        ]),
    )


def test_rv_component_marker_in_standard_position(
    tmp_path
):

    filename = (
        tmp_path
        / "example.inp"
    )

    filename.write_text(
        """
Object: TEST

P 10.0
T 2000.0
e 0.2
a 1.0
W 30.0
w 60.0
i 70.0

2000.0 12.0 0.5 Va
2001.0 -8.0 0.7 Vb
""".strip()
        + "\n"
    )

    parsed = parse_orbit_file(
        str(filename)
    )

    rv_data = (
        parsed["rv_data"]
    )

    assert rv_data.shape == (
        2,
        4,
    )

    np.testing.assert_array_equal(
        rv_data[:, 3],
        np.array([
            0,
            1,
        ]),
    )


# ==========================================================
# Va+b handling
# ==========================================================

def test_va_plus_b_is_not_silently_assigned_to_a_component(
    tmp_path
):

    filename = (
        tmp_path
        / "example.inp"
    )

    filename.write_text(
        """
Object: TEST

P 10.0
T 2000.0
e 0.2
a 1.0
W 30.0
w 60.0
i 70.0

2000.0 12.0 0.5 Va
2001.0 -8.0 0.7 Va+b
2002.0 4.0 0.6 Vb
""".strip()
        + "\n"
    )

    parsed = parse_orbit_file(
        str(filename)
    )

    rv_data = (
        parsed["rv_data"]
    )

    # Only the unambiguous Va and Vb records should
    # currently be represented in the binary RV array.
    #
    # Va+b has deliberately not been assigned to either
    # component.

    assert rv_data.shape == (
        2,
        4,
    )

    np.testing.assert_array_equal(
        rv_data[:, 3],
        np.array([
            0,
            1,
        ]),
    )

    np.testing.assert_allclose(
        rv_data[:, 0],
        np.array([
            2000.0,
            2002.0,
        ]),
    )


# ==========================================================
# GL 765 regression
# ==========================================================

def test_gl765_recovers_unambiguous_rv_records():

    filepath = (
        ROOT
        / "GL765"
        / "GL765.inp"
    )

    parsed = parse_orbit_file(
        str(filepath)
    )

    rv_data = (
        parsed["rv_data"]
    )

    # GL 765 contains 108 RV records whose component
    # identifiers are recognized by the current parser.
    #
    # The two Va+b records are intentionally not assigned
    # to either binary component until their ORBIT semantics
    # are explicitly established.

    assert len(
        rv_data
    ) == 108

    
# ==========================================================
# GL 765 time normalization regression
# ==========================================================

def test_gl765_rv_times_are_normalized_to_decimal_years():

    filepath = (
        ROOT
        / "GL765"
        / "GL765.inp"
    )

    parsed = parse_orbit_file(
        str(filepath)
    )

    rv_data = (
        parsed["rv_data"]
    )

    # The source RV epochs are around 45533--49604
    # (JD-2400000 convention). They must not remain in
    # that numerical representation after parsing.

    assert (
        rv_data[:, 0].min()
        < 2100.0
    )

    assert (
        rv_data[:, 0].max()
        > 1900.0
    )


# ==========================================================
# Optional standalone diagnostic
# ==========================================================

def _print_benchmark_diagnostics():

    benchmark_files = [

        ROOT
        / "FIN379"
        / "FIN379.inp",

        ROOT
        / "GL765"
        / "GL765.inp",

        ROOT
        / "ZetaAqr"
        / "ZetaAqr.in2",
    ]

    for filepath in benchmark_files:

        print()

        print(
            "=" * 60
        )

        print(
            filepath.name
        )

        print(
            "=" * 60
        )

        parsed = parse_orbit_file(
            str(filepath)
        )

        elements = (
            parsed["elements"]
        )

        visual_data = (
            parsed["visual_data"]
        )

        rv_data = (
            parsed["rv_data"]
        )

        metadata = (
            parsed["metadata"]
        )

        print()

        print(
            "Metadata:"
        )

        print(
            metadata
        )

        print()

        print(
            "Orbital elements:"
        )

        print(
            elements
        )

        print()

        print(
            "Visual data shape:"
        )

        print(
            visual_data.shape
        )

        print()

        print(
            "Visual finite:"
        )

        print(
            np.all(
                np.isfinite(
                    visual_data
                )
            )
        )

        print()

        print(
            "RV data shape:"
        )

        print(
            rv_data.shape
        )

        if len(rv_data) > 0:

            print()

            print(
                "RV finite:"
            )

            print(
                np.all(
                    np.isfinite(
                        rv_data
                    )
                )
            )

            print()

            print(
                "RV components:"
            )

            print(
                np.unique(
                    rv_data[:, 3]
                )
            )

        print()

        print(
            "Visual time range:"
        )

        print(
            visual_data[:, 0].min(),
            visual_data[:, 0].max()
        )

        if len(rv_data) > 0:

            print()

            print(
                "RV time range:"
            )

            print(
                rv_data[:, 0].min(),
                rv_data[:, 0].max()
            )


if __name__ == "__main__":

    _print_benchmark_diagnostics()