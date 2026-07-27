from pyseshat.evolution import (
    IsochroneFitResult,
    fit_isochrone,
)


def test_public_api():

    assert IsochroneFitResult is not None
    assert callable(fit_isochrone)