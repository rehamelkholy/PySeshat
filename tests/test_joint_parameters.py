import numpy as np

from pyseshat.inference.joint_parameters import (
    JointParameters,
)

from pyseshat.orbit.orbital_elements import (
    OrbitalElements,
)

from pyseshat.photometry.parameters import (
    PhotometricParameters,
)


def make_orbital_elements():

    elements = OrbitalElements(

        P=10.0,
        T0=2000.0,
        e=0.2,
        a=0.5,
        Omega=120.0,
        omega=45.0,
        i=80.0,

        K1=15.0,
        K2=18.0,
        V0=12.0,

    )

    elements.fit_flags = {

        "P": True,
        "T0": True,
        "e": True,
        "a": False,
        "Omega": False,
        "omega": False,
        "i": False,
        "K1": False,
        "K2": False,
        "V0": False,

    }

    return elements


def make_photometric_parameters():

    return PhotometricParameters(

        primary_temperature=6000.0,

        secondary_temperature=5200.0,

        primary_radius=1.30,

        secondary_radius=0.90,

        ebv=0.05,

    )


def test_fitted_vector():

    joint = JointParameters(

        orbital=make_orbital_elements(),

        photometric=make_photometric_parameters(),

    )

    vector = joint.fitted_vector()

    expected = np.concatenate(

        (

            joint.orbital.fitted_parameters(),

            joint.photometric.fitted_parameters(),

        )

    )

    assert np.allclose(

        vector,

        expected,

    )


def test_update_from_vector():

    joint = JointParameters(

        orbital=make_orbital_elements(),

        photometric=make_photometric_parameters(),

    )

    vector = joint.fitted_vector().copy()

    vector += 1.0

    updated = joint.update_from_vector(

        vector,

    )

    assert np.allclose(

        updated.fitted_vector(),

        vector,

    )


def test_original_is_unchanged():

    joint = JointParameters(

        orbital=make_orbital_elements(),

        photometric=make_photometric_parameters(),

    )

    original = joint.fitted_vector().copy()

    vector = original + 1.0

    _ = joint.update_from_vector(

        vector,

    )

    assert np.allclose(

        joint.fitted_vector(),

        original,

    )


def test_size_properties():

    joint = JointParameters(

        orbital=make_orbital_elements(),

        photometric=make_photometric_parameters(),

    )

    assert (

        joint.size

        ==

        joint.n_orbital

        +

        joint.n_photometric

    )