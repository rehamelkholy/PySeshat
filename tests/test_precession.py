from pyseshat.analysis.precession import (
    compute_precession_rate,
    apply_theta_precession,
    remove_theta_precession
)

ra = 90.0
dec = 30.0

pr = compute_precession_rate(
    ra,
    dec
)

print("Rate:", pr)

theta = 100.0
epoch = 1990.0

theta_corr = apply_theta_precession(
    theta,
    epoch,
    ra,
    dec
)

theta_back = remove_theta_precession(
    theta_corr,
    epoch,
    ra,
    dec
)

print("Original:", theta)
print("Corrected:", theta_corr)
print("Recovered:", theta_back)