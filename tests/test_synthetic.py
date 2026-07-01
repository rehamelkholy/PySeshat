import numpy as np

from pyseshat.experiments.synthetic import (
    perturb_values,
    perturb_xy
)

rng = np.random.default_rng(42)

values = np.ones(10)

sigma = 0.1

perturbed = perturb_values(
    values,
    sigma,
    rng=rng
)

print()

print("VALUES")
print("=" * 60)

print(perturbed)

print()

print("MEAN DIFFERENCE")
print("=" * 60)

print(np.mean(perturbed - values))

print()

print("STD OF DIFFERENCE")
print("=" * 60)

print(np.std(perturbed - values))

print()

x = np.zeros(5)
y = np.zeros(5)

sx = np.full(5, 0.01)
sy = np.full(5, 0.02)

x2, y2 = perturb_xy(
    x,
    y,
    sx,
    sy,
    rng=np.random.default_rng(1)
)

print()

print("PERTURBED X")
print("=" * 60)

print(x2)

print()

print("PERTURBED Y")
print("=" * 60)

print(y2)