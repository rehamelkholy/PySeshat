import numpy as np


def compute_precession_rate(
    ra_deg,
    dec_deg
):
    """
    ORBIT precession correction rate.

    Returns
    -------
    degrees/year
    """

    return (
        0.0057
        *
        np.sin(
            np.deg2rad(ra_deg)
        )
        /
        np.cos(
            np.deg2rad(dec_deg)
        )
    )

def apply_theta_precession(
    theta,
    epoch,
    ra_deg,
    dec_deg
):
    """
    Apply ORBIT-style precession
    correction to position angle.
    """

    pr = compute_precession_rate(
        ra_deg,
        dec_deg
    )

    return (
        theta
        +
        (2000.0 - epoch)
        * pr
    )

def remove_theta_precession(
    theta,
    epoch,
    ra_deg,
    dec_deg
):
    """
    Undo ORBIT-style precession
    correction.
    """

    pr = compute_precession_rate(
        ra_deg,
        dec_deg
    )

    return (
        theta
        -
        (2000.0 - epoch)
        * pr
    )