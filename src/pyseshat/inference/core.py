from pyseshat.orbit.orbital_elements import OrbitalElements


def update_elements(
    elements: OrbitalElements,
    fitted_values,
):
    """
    Create updated orbital elements object by replacing
    fitted parameters only.
    """

    full = elements.to_array().copy()

    full[elements.fitted_mask()] = fitted_values

    updated = OrbitalElements.from_array(
        full,
        fit_flags=elements.fit_flags.copy(),
    )

    updated.normalize_angles()

    return updated