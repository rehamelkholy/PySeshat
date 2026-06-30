import numpy as np

from pyseshat.inference.base_fit_result import (
    BaseFitResult
)


class DummyOptimizer:

    success = True

    message = "Dummy optimizer."


result = BaseFitResult(

    elements=None,

    optimizer_result=DummyOptimizer(),

    residuals=np.array(
        [
            1.0,
            -2.0,
            3.0
        ]
    ),

    covariance=np.eye(1),

    uncertainties={}
)

print()

print(
    "SUCCESS"
)

print(
    "=" * 60
)

print(
    result.success
)

print()

print(
    "MESSAGE"
)

print(
    "=" * 60
)

print(
    result.message
)

print()

print(
    "SUMMARY"
)

print(
    "=" * 60
)

print(
    result.summary(
        n_parameters=1
    )
)