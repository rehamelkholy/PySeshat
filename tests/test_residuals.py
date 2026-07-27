from pyseshat.evolution.residuals import StellarResiduals


def test_empty_residuals():

    r = StellarResiduals()

    assert r.available == ()


def test_available_fields():

    r = StellarResiduals(
        log_teff=0.02,
        log_g=-0.10,
    )

    assert r.available == (
        "log_teff",
        "log_g",
    )