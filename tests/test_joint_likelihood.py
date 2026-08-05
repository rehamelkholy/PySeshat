import pytest

from pyseshat.inference.joint_problem import (
    JointInferenceProblem,
)

from pyseshat.inference.joint_likelihood import (
    joint_log_likelihood,
)


def test_joint_log_likelihood_combines_terms(
    monkeypatch,
):
    """
    Verify that the joint likelihood is the sum of the
    orbital and photometric likelihoods.
    """

    class DummyOrbitProblem:
        pass

    class DummyPhotometricProblem:
        pass

    problem = JointInferenceProblem(

        orbit_problem=DummyOrbitProblem(),

        photometric_problem=DummyPhotometricProblem(),

    )

    orbit_called = False
    photometry_called = False

    def fake_orbit(
        fitted_values,
        problem,
    ):
        nonlocal orbit_called

        orbit_called = True

        assert fitted_values == [1.0, 2.0]

        assert isinstance(
            problem,
            DummyOrbitProblem,
        )

        return -12.5

    def fake_photometry(
        problem,
    ):
        nonlocal photometry_called

        photometry_called = True

        assert isinstance(
            problem,
            DummyPhotometricProblem,
        )

        return -3.25

    monkeypatch.setattr(

        "pyseshat.inference.joint_likelihood."
        "log_likelihood_from_problem",

        fake_orbit,

    )

    monkeypatch.setattr(

        "pyseshat.inference.joint_likelihood."
        "photometric_log_likelihood_from_problem",

        fake_photometry,

    )

    result = joint_log_likelihood(

        orbit_fitted_values=[1.0, 2.0],

        joint_problem=problem,

    )

    assert orbit_called

    assert photometry_called

    assert result == pytest.approx(

        -15.75

    )


def test_joint_log_likelihood_requires_problem():

    with pytest.raises(

        TypeError

    ):

        joint_log_likelihood(

            orbit_fitted_values=[],

            joint_problem=None,

        )