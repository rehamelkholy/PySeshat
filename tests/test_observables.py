from pyseshat.evolution.observables import (
    StellarObservables,
)


def test_default_observables():

    obs = StellarObservables()

    assert obs.log_teff is None

    assert obs.log_luminosity is None

    assert obs.log_radius is None

    assert obs.log_g is None


def test_store_observables():

    obs = StellarObservables(

        log_teff=3.75,
        sigma_log_teff=0.02,

        log_luminosity=0.15,
        sigma_log_luminosity=0.05,

        log_g=4.45,
        sigma_log_g=0.10,

    )

    assert obs.log_teff == 3.75

    assert obs.sigma_log_teff == 0.02

    assert obs.log_g == 4.45