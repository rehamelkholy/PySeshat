import numpy as np

import pytest

from pyseshat.evolution.observables import (
    StellarObservables,
)

from pyseshat.inference.base_fit_result import (
    BaseFitResult,
)

from pyseshat.photometry.evolution_adapter import (
    stellar_observables_from_fit_result,
    stellar_observables_from_photometric_parameters,
    stellar_observables_from_solution,
)

from pyseshat.photometry.parameters import (
    PhotometricParameters,
)


def make_parameters():

    return PhotometricParameters(

        primary_temperature=6000.0,

        secondary_temperature=5000.0,

        primary_radius=1.2,

        secondary_radius=0.9,

        ebv=0.10,

    )


def make_uncertainties():

    return {

        "primary_temperature": 60.0,

        "secondary_temperature": 50.0,

        "primary_radius": 0.06,

        "secondary_radius": 0.045,

        "ebv": 0.01,

    }


def test_stellar_observables_from_photometric_parameters():

    primary, secondary = (
        stellar_observables_from_photometric_parameters(
            make_parameters()
        )
    )

    assert isinstance(
        primary,
        StellarObservables,
    )

    assert isinstance(
        secondary,
        StellarObservables,
    )

    assert primary.log_teff == pytest.approx(
        np.log10(6000.0)
    )

    assert secondary.log_teff == pytest.approx(
        np.log10(5000.0)
    )

    assert primary.log_radius == pytest.approx(
        np.log10(1.2)
    )

    assert secondary.log_radius == pytest.approx(
        np.log10(0.9)
    )

    assert primary.sigma_log_teff is None

    assert secondary.sigma_log_teff is None

    assert primary.sigma_log_luminosity is None

    assert secondary.sigma_log_luminosity is None


def test_stellar_observables_from_solution():

    parameters = make_parameters()

    uncertainties = make_uncertainties()

    primary, secondary = (
        stellar_observables_from_solution(

            parameters=parameters,

            uncertainties=uncertainties,

        )
    )

    assert isinstance(
        primary,
        StellarObservables,
    )

    assert isinstance(
        secondary,
        StellarObservables,
    )

    assert primary.log_teff == pytest.approx(
        np.log10(6000.0)
    )

    assert secondary.log_teff == pytest.approx(
        np.log10(5000.0)
    )

    assert primary.log_radius == pytest.approx(
        np.log10(1.2)
    )

    assert secondary.log_radius == pytest.approx(
        np.log10(0.9)
    )

    assert primary.sigma_log_teff is not None

    assert secondary.sigma_log_teff is not None

    assert primary.sigma_log_luminosity is not None

    assert secondary.sigma_log_luminosity is not None

    assert primary.sigma_log_radius is not None

    assert secondary.sigma_log_radius is not None


def test_stellar_observables_from_fit_result():

    parameters = make_parameters()

    uncertainties = make_uncertainties()

    fit_result = BaseFitResult(

        elements=parameters,

        optimizer_result=None,

        residuals=np.zeros(2),

        covariance=np.eye(5),

        uncertainties=uncertainties,

    )

    primary, secondary = (
        stellar_observables_from_fit_result(
            fit_result
        )
    )

    assert isinstance(
        primary,
        StellarObservables,
    )

    assert isinstance(
        secondary,
        StellarObservables,
    )

    assert primary.log_teff == pytest.approx(
        np.log10(6000.0)
    )

    assert secondary.log_teff == pytest.approx(
        np.log10(5000.0)
    )

    assert primary.log_radius == pytest.approx(
        np.log10(1.2)
    )

    assert secondary.log_radius == pytest.approx(
        np.log10(0.9)
    )

    assert primary.sigma_log_teff == pytest.approx(
        stellar_observables_from_solution(

            parameters,

            uncertainties,

        )[0].sigma_log_teff
    )

    assert secondary.sigma_log_teff == pytest.approx(
        stellar_observables_from_solution(

            parameters,

            uncertainties,

        )[1].sigma_log_teff
    )

    assert primary.sigma_log_luminosity == pytest.approx(
        stellar_observables_from_solution(

            parameters,

            uncertainties,

        )[0].sigma_log_luminosity
    )

    assert secondary.sigma_log_luminosity == pytest.approx(
        stellar_observables_from_solution(

            parameters,

            uncertainties,

        )[1].sigma_log_luminosity
    )


def test_fit_result_adapter_preserves_fitted_parameters():

    parameters = make_parameters()

    uncertainties = make_uncertainties()

    fit_result = BaseFitResult(

        elements=parameters,

        optimizer_result=None,

        residuals=np.zeros(2),

        covariance=np.eye(5),

        uncertainties=uncertainties,

    )

    primary, secondary = (
        stellar_observables_from_fit_result(
            fit_result
        )
    )

    direct_primary, direct_secondary = (
        stellar_observables_from_solution(

            parameters,

            uncertainties,

        )
    )

    assert primary == direct_primary

    assert secondary == direct_secondary