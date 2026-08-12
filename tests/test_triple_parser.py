from pathlib import Path

import pytest

from pyseshat.triple.triple_parser import (
    parse_triple_file
)


def write_file(
    tmp_path,
    name,
    content
):

    filename = (
        tmp_path
        /
        name
    )

    filename.write_text(
        content,
        encoding="utf-8"
    )

    return filename


# ==========================================================
# Legacy .in2 parser
# ==========================================================

def test_parse_legacy_in2_file(
    tmp_path
):

    filename = write_file(

        tmp_path,

        "example.in2",

        """
C Legacy ORBITT2 example

Object: Legacy Triple
R.A.: 12.3456
Dec.: -12.3456
WDS: 12345-1234

P 30.0
T 2010.0
e 0.30
a 0.50
W 120.0
w 80.0
i 70.0

P2 3.0
T2 2012.0
e2 0.10
a2 0.05
W2 250.0
w2 140.0
i2 85.0

B 0.75

2010.0 120.0 0.50 0.01 x I2
2012.0 250.0 0.05 0.01 x I1
"""
    )

    parsed = parse_triple_file(
        filename
    )

    elements = parsed[
        "elements"
    ]

    observations = parsed[
        "observations"
    ]

    assert (
        parsed["metadata"]["format"]
        ==
        "in2"
    )

    assert (
        parsed["metadata"]["object_name"]
        ==
        "Legacy Triple"
    )

    assert (
        parsed["metadata"]["ra"]
        ==
        "12.3456"
    )

    assert (
        parsed["metadata"]["dec"]
        ==
        "-12.3456"
    )

    assert (
        parsed["metadata"]["wds_code"]
        ==
        "12345-1234"
    )

    # Legacy ordering:
    #
    # P...i  -> outer
    # P2...i2 -> inner

    assert elements.outer.P == pytest.approx(
        30.0
    )

    assert elements.inner.P == pytest.approx(
        3.0
    )

    assert elements.B == pytest.approx(
        0.75
    )

    assert elements.f == pytest.approx(
        0.0
    )

    assert len(
        observations
    ) == 2

    assert (
        observations[0].orbit_id
        ==
        "I2"
    )

    assert (
        observations[1].orbit_id
        ==
        "I1"
    )

    assert (
        parsed["rv_observations"]
        ==
        []
    )


# ==========================================================
# Official ORBIT3 header format
# ==========================================================

def test_parse_orbit3_header_format(
    tmp_path
):

    filename = write_file(

        tmp_path,

        "example.in3",

        """
HIP 111805
12.0000
+30.0000
12000+3000

P 1.50119
T 1986.093
e 0.0220
a 0.0385
W 334.5
w 232.9
i 85.8
K1 13.13
K2 19.21
f -0.33

P2 30.127
T2 2010.179
e2 0.324
a2 0.3361
W2 154.25
w2 84.92
i2 88.28
K3 6.057
K4 8.60
V0 -22.575
"""
    )

    parsed = parse_triple_file(
        filename
    )

    metadata = parsed[
        "metadata"
    ]

    assert (
        metadata["format"]
        ==
        "in3"
    )

    assert (
        metadata["object_name"]
        ==
        "HIP 111805"
    )

    assert (
        metadata["ra"]
        ==
        "12.0000"
    )

    assert (
        metadata["dec"]
        ==
        "+30.0000"
    )

    assert (
        metadata["wds_code"]
        ==
        "12000+3000"
    )


# ==========================================================
# Modern ORBIT3 parameters
# ==========================================================

def test_parse_orbit3_parameters(
    tmp_path
):

    filename = write_file(

        tmp_path,

        "example.in3",

        """
HIP 111805
12.0000
+30.0000
12000+3000

*P 1.50119
*T 1986.093
e 0.0220
*a 0.0385
W 334.5
w 232.9
i 85.8
K1 13.13
*K2 19.21
f -0.33

P2 30.127
*T2 2010.179
e2 0.324
a2 0.3361
W2 154.25
*w2 84.92
i2 88.28
K3 6.057
K4 8.60
V0 -22.575
"""
    )

    parsed = parse_triple_file(
        filename
    )

    elements = parsed[
        "elements"
    ]

    # ORBIT3:
    #
    # P...i -> inner
    # P2...i2 -> outer

    assert elements.inner.P == pytest.approx(
        1.50119
    )

    assert elements.inner.T0 == pytest.approx(
        1986.093
    )

    assert elements.inner.e == pytest.approx(
        0.0220
    )

    assert elements.inner.a == pytest.approx(
        0.0385
    )

    assert elements.inner.Omega == pytest.approx(
        334.5
    )

    assert elements.inner.omega == pytest.approx(
        232.9
    )

    assert elements.inner.i == pytest.approx(
        85.8
    )

    assert elements.K1 == pytest.approx(
        13.13
    )

    assert elements.K2 == pytest.approx(
        19.21
    )

    assert elements.f == pytest.approx(
        -0.33
    )

    assert elements.outer.P == pytest.approx(
        30.127
    )

    assert elements.outer.T0 == pytest.approx(
        2010.179
    )

    assert elements.outer.e == pytest.approx(
        0.324
    )

    assert elements.outer.a == pytest.approx(
        0.3361
    )

    assert elements.outer.Omega == pytest.approx(
        154.25
    )

    assert elements.outer.omega == pytest.approx(
        84.92
    )

    assert elements.outer.i == pytest.approx(
        88.28
    )

    assert elements.K3 == pytest.approx(
        6.057
    )

    assert elements.K4 == pytest.approx(
        8.60
    )

    assert elements.V0 == pytest.approx(
        -22.575
    )


# ==========================================================
# ORBIT3 fit flags
# ==========================================================

def test_orbit3_fit_flags(
    tmp_path
):

    filename = write_file(

        tmp_path,

        "example.in3",

        """
Test
00.0000
+00.0000
12000+0000

*P 1.0
T 2000.0
*e 0.2
a 0.1
W 10.0
w 20.0
i 30.0
*K1 10.0
K2 20.0
*f -0.3

P2 10.0
*T2 2001.0
e2 0.3
*a2 0.5
W2 40.0
w2 50.0
i2 60.0
K3 5.0
*K4 7.0
*V0 -20.0
"""
    )

    elements = parse_triple_file(
        filename
    )["elements"]


    #
    # A leading "*" means that the parameter is FIXED.
    # Therefore it must NOT be included in the fitted
    # parameter mask.
    #

    assert not elements.fit_flags[
        "P"
    ]

    assert elements.fit_flags[
        "T0"
    ]

    assert not elements.fit_flags[
        "e"
    ]

    assert elements.fit_flags[
        "a"
    ]


    #
    # Inner spectroscopic parameters.
    #

    assert not elements.fit_flags[
        "K1"
    ]

    assert elements.fit_flags[
        "K2"
    ]

    assert not elements.fit_flags[
        "f"
    ]


    #
    # Outer orbit.
    #

    assert elements.fit_flags[
        "P2"
    ]

    assert not elements.fit_flags[
        "T02"
    ]

    assert elements.fit_flags[
        "e2"
    ]

    assert not elements.fit_flags[
        "a2"
    ]

    assert elements.fit_flags[
        "Omega2"
    ]

    assert elements.fit_flags[
        "omega2"
    ]

    assert elements.fit_flags[
        "i2"
    ]


    #
    # Outer spectroscopic parameters.
    #

    assert elements.fit_flags[
        "K3"
    ]

    assert not elements.fit_flags[
        "K4"
    ]

    assert not elements.fit_flags[
        "V0"
    ]


# ==========================================================
# Parameter errors
# ==========================================================

def test_parse_parameter_errors(
    tmp_path
):

    filename = write_file(

        tmp_path,

        "example.in3",

        """
Test
00.0000
+00.0000
12000+0000

P 10.0 0.1
T 2000.0 1.5
e 0.2 0.01
a 0.5 0.02
W 100.0 1.0
w 50.0 2.0
i 70.0 0.5

P2 5.0 0.1
T2 2001.0 1.0
e2 0.1 0.01
a2 0.1 0.01
W2 20.0 1.0
w2 30.0 1.0
i2 80.0 0.5

K1 10.0 0.2
K2 12.0 0.3
f -0.4 0.05

K3 5.0 0.1
K4 6.0 0.2
V0 -20.0 0.5
"""
    )

    parsed = parse_triple_file(
        filename
    )

    errors = parsed[
        "parameter_errors"
    ]

    assert errors[
        "P"
    ] == pytest.approx(
        0.1
    )

    assert errors[
        "T"
    ] == pytest.approx(
        1.5
    )

    assert errors[
        "K1"
    ] == pytest.approx(
        0.2
    )

    assert errors[
        "f"
    ] == pytest.approx(
        0.05
    )

    assert errors[
        "V0"
    ] == pytest.approx(
        0.5
    )


# ==========================================================
# Radial velocities
# ==========================================================

def test_parse_single_radial_velocity_records(
    tmp_path
):

    filename = write_file(

        tmp_path,

        "example.in3",

        """
Test
00.0000
+00.0000
12000+0000

P 10.0
T 2000.0
e 0.2
a 0.5
W 100.0
w 50.0
i 70.0

P2 5.0
T2 2001.0
e2 0.1
a2 0.1
W2 20.0
w2 30.0
i2 80.0

2000.0 15.5 0.5 Va
2001.0 -12.0 0.7 Vb
2002.0 8.0 0.4 Vc
"""
    )

    parsed = parse_triple_file(
        filename
    )

    rv = parsed[
        "rv_observations"
    ]

    assert len(
        rv
    ) == 3

    assert rv[0][
        "component"
    ] == "Va"

    assert rv[0][
        "velocity"
    ] == pytest.approx(
        15.5
    )

    assert rv[0][
        "sigma"
    ] == pytest.approx(
        0.5
    )

    assert rv[1][
        "component"
    ] == "Vb"

    assert rv[1][
        "velocity"
    ] == pytest.approx(
        -12.0
    )

    assert rv[2][
        "component"
    ] == "Vc"


# ==========================================================
# Positional records
# ==========================================================

def test_parse_i1_and_i2_records(
    tmp_path
):

    filename = write_file(

        tmp_path,

        "example.in3",

        """
Test
00.0000
+00.0000
12000+0000

P 10.0
T 2000.0
e 0.2
a 0.5
W 100.0
w 50.0
i 70.0

P2 5.0
T2 2001.0
e2 0.1
a2 0.1
W2 20.0
w2 30.0
i2 80.0

2000.0 120.0 0.50 0.01 source I1
2001.0 250.0 0.05 0.02 source I2
"""
    )

    parsed = parse_triple_file(
        filename
    )

    observations = parsed[
        "observations"
    ]

    assert len(
        observations
    ) == 2

    assert observations[0].orbit_id == "I1"

    assert observations[0].theta == pytest.approx(
        120.0
    )

    assert observations[0].rho == pytest.approx(
        0.50
    )

    assert observations[1].orbit_id == "I2"

    assert observations[1].sigma == pytest.approx(
        0.02
    )


# ==========================================================
# Missing required parameters
# ==========================================================

def test_missing_required_parameter_raises(
    tmp_path
):

    filename = write_file(

        tmp_path,

        "invalid.in3",

        """
Invalid
00.0000
+00.0000
12000+0000

P 10.0
T 2000.0
e 0.2
a 0.5
W 100.0
w 50.0
"""
    )

    with pytest.raises(
        ValueError,
        match="Missing required triple orbital parameters"
    ):

        parse_triple_file(
            filename
        )