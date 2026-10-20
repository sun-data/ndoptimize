from typing import Any
import numpy as np
import pytest
import ndoptimize


def _evaluate(
    model: ndoptimize.Model,
    x: np.ndarray,
    parameters: np.ndarray,
) -> np.ndarray:
    """Evaluate a model for each row of `parameters`, one at a time."""
    f = np.empty((len(parameters), len(x)))
    jacobian = np.empty((len(x), parameters.shape[-1]))
    for i in range(len(parameters)):
        model(x, parameters[i], f[i], jacobian)
    return f


def test_evaluate() -> None:
    """Evaluating many sets of parameters at once matches evaluating them
    one at a time, with the points shared or not."""
    rng = np.random.default_rng(0)
    x = np.linspace(-50, 50, 21)[:, np.newaxis]
    parameters = np.stack(
        [
            rng.uniform(1, 2, 10),
            rng.normal(0, 5, 10),
            rng.uniform(5, 20, 10),
            rng.normal(0, 0.01, 10),
            rng.uniform(0, 1, 10),
        ],
        axis=-1,
    )
    model = ndoptimize.models.gaussian_linear
    expected = _evaluate(model, x, parameters)

    result = ndoptimize.evaluate(model=model, x=x, parameters=parameters)
    assert np.array_equal(result, expected)

    result_x = ndoptimize.evaluate(
        model=model,
        x=np.broadcast_to(x, (10,) + x.shape),
        parameters=parameters[0],
    )
    assert np.array_equal(result_x, np.broadcast_to(expected[0], (10, len(x))))


def test_evaluate_broadcasting() -> None:
    """The batch axes of the points and the parameters broadcast."""
    x = np.linspace(-50, 50, 21)[:, np.newaxis]
    x = np.stack([x, 2 * x])[:, np.newaxis]
    parameters = np.array([[1, 0, 10], [2, 5, 15], [3, -5, 20.0]])

    result = ndoptimize.evaluate(
        model=ndoptimize.models.gaussian,
        x=x,
        parameters=parameters,
    )

    assert result.shape == (2, 3, 21)
    for i in range(2):
        assert np.array_equal(
            result[i],
            _evaluate(ndoptimize.models.gaussian, x[i, 0], parameters),
        )


def test_evaluate_least_squares() -> None:
    """The model evaluated at the best-fit parameters of noiseless data is
    the data."""
    x = np.linspace(-50, 50, 21)[:, np.newaxis]
    data = ndoptimize.evaluate(ndoptimize.models.gaussian, x, [2, 3, 12.0])
    result = ndoptimize.least_squares(
        model=ndoptimize.models.gaussian,
        x=x,
        data=data,
        variance=1,
        guess=[1, 0, 10],
    )
    f = ndoptimize.evaluate(ndoptimize.models.gaussian, x, result.parameters)
    assert np.allclose(f, data, atol=1e-6)


@pytest.mark.parametrize(
    argnames="kwargs",
    argvalues=[
        dict(x=np.linspace(-50, 50, 21)),
        dict(x=np.zeros((21, 2))),
        dict(model=ndoptimize.models.gaussian_binned),
        dict(parameters=np.array([1, 0.0])),
        dict(parameters=1.0),
    ],
)
def test_evaluate_invalid(kwargs: dict[str, Any]) -> None:
    arguments: dict[str, Any] = dict(
        model=ndoptimize.models.gaussian,
        x=np.linspace(-50, 50, 21)[:, np.newaxis],
        parameters=np.array([1, 0, 15.0]),
    )
    with pytest.raises(ValueError):
        ndoptimize.evaluate(**(arguments | kwargs))
