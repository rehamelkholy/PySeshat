from pyseshat.experiments.monte_carlo import (
    MonteCarloExperiment
)


def dummy_fit():

    return None


mc = MonteCarloExperiment(

    fit_function=dummy_fit,

    n_trials=50
)

print()

print("TRIALS")
print("=" * 60)

print(mc.n_trials)

print()

print("FIT FUNCTION")
print("=" * 60)

print(mc.fit_function.__name__)

print()

print("RUN")
print("=" * 60)

try:

    mc.run()

except NotImplementedError as e:

    print(type(e).__name__)

    print(e)