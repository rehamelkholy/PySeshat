import numpy as np

from pyseshat.photometry.parameters import (
    PhotometricParameters
)

from pyseshat.inference.base_fit_result import (
    BaseFitResult
)

from pyseshat.inference.photometric_optimizer import (
    build_parameter_bounds,
    fit_photometry
)


# ==========================================================
# Bounds
# ==========================================================

def test_build_parameter_bounds():

    lower, upper = build_parameter_bounds()

    assert lower.shape == (5,)
    assert upper.shape == (5,)

    assert np.all(lower < upper)

    np.testing.assert_allclose(

        lower,

        np.array(

            [

                2500.0,

                2500.0,

                0.01,

                0.01,

                0.0

            ]

        )

    )

    np.testing.assert_allclose(

        upper,

        np.array(

            [

                50000.0,

                50000.0,

                100.0,

                100.0,

                10.0

            ]

        )

    )


# ==========================================================
# Wrapper around run_least_squares()
# ==========================================================

def test_fit_photometry(monkeypatch):

    captured = {}

    def fake_run_least_squares(**kwargs):

        captured.update(kwargs)

        return BaseFitResult(

            elements=kwargs["solution_builder"](
                kwargs["initial_parameters"]
            ),

            optimizer_result=None,

            residuals=np.zeros(2),

            covariance=np.eye(5),

            uncertainties={}

        )

    monkeypatch.setattr(

        "pyseshat.inference.photometric_optimizer."
        "run_least_squares",

        fake_run_least_squares

    )

    parameters = PhotometricParameters(

        primary_temperature=6000.0,

        secondary_temperature=5000.0,

        primary_radius=1.2,

        secondary_radius=0.9,

        ebv=0.10

    )

    result = fit_photometry(

        parameters=parameters,

        atmosphere_grid=None,

        passbands={},

        zero_points={},

        measurements=[],

        primary_logg=4.5,

        secondary_logg=4.5,

        metallicity=0.0,

        distance=10.0

    )

    assert isinstance(

        result,

        BaseFitResult

    )

    np.testing.assert_allclose(

        captured["initial_parameters"],

        parameters.fitted_parameters()

    )

    assert (

        captured["parameter_names"]

        ==

        parameters.fitted_parameter_names()

    )

    assert callable(

        captured["residual_function"]

    )

    assert callable(

        captured["solution_builder"]

    )

    assert captured["fit_result_class"] is BaseFitResult

    assert captured["method"] == "trf"
    assert captured["loss"] == "linear"
    assert captured["f_scale"] == 1.0

    lower, upper = captured["bounds"]

    assert lower.shape == (5,)
    assert upper.shape == (5,)


# ==========================================================
# User-supplied bounds override defaults
# ==========================================================

def test_fit_photometry_custom_bounds(monkeypatch):

    captured = {}

    custom_bounds = (

        np.zeros(5),

        np.ones(5)

    )

    def fake_run_least_squares(**kwargs):

        captured.update(kwargs)

        return BaseFitResult(

            elements=kwargs["solution_builder"](
                kwargs["initial_parameters"]
            ),

            optimizer_result=None,

            residuals=np.zeros(2),

            covariance=np.eye(5),

            uncertainties={}

        )

    monkeypatch.setattr(

        "pyseshat.inference.photometric_optimizer."
        "run_least_squares",

        fake_run_least_squares

    )

    parameters = PhotometricParameters(

        6000,

        5000,

        1.2,

        0.9,

        0.1

    )

    fit_photometry(

        parameters=parameters,

        atmosphere_grid=None,

        passbands={},

        zero_points={},

        measurements=[],

        primary_logg=4.5,

        secondary_logg=4.5,

        metallicity=0.0,

        distance=10.0,

        bounds=custom_bounds

    )

    assert captured["bounds"] is custom_bounds