import numpy as np

from pyseshat.triple.triple_elements import (
    TripleOrbitalElements
)


TRIPLE_PARAMETER_ORDER = [

    # outer orbit

    "P",
    "T0",
    "e",
    "a",
    "Omega",
    "omega",
    "i",

    # inner orbit

    "P2",
    "T02",
    "e2",
    "a2",
    "Omega2",
    "omega2",
    "i2",

    # scale

    "B"
]


def triple_to_array(
    elements: TripleOrbitalElements
):
    """
    Convert TripleOrbitalElements
    into parameter vector.
    """

    return np.array([

        elements.outer.P,
        elements.outer.T0,
        elements.outer.e,
        elements.outer.a,
        elements.outer.Omega,
        elements.outer.omega,
        elements.outer.i,

        elements.inner.P,
        elements.inner.T0,
        elements.inner.e,
        elements.inner.a,
        elements.inner.Omega,
        elements.inner.omega,
        elements.inner.i,

        elements.B
    ])


def array_to_triple(
    p,
    template
):
    """
    Create TripleOrbitalElements
    from parameter vector.
    """

    outer = template.outer.copy()

    inner = template.inner.copy()

    outer.P      = p[0]
    outer.T0     = p[1]
    outer.e      = p[2]
    outer.a      = p[3]
    outer.Omega  = p[4]
    outer.omega  = p[5]
    outer.i      = p[6]

    inner.P      = p[7]
    inner.T0     = p[8]
    inner.e      = p[9]
    inner.a      = p[10]
    inner.Omega  = p[11]
    inner.omega  = p[12]
    inner.i      = p[13]

    return TripleOrbitalElements(
        outer=outer,
        inner=inner,
        B=p[14]
    )