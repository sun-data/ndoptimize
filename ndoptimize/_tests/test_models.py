from typing import Callable
import numpy as np
import pytest
import ndoptimize


def _gaussian(v: np.ndarray, a: float, c: float, w: float) -> np.ndarray:
    """A Gaussian evaluated with numpy."""
    return a * np.exp(-0.5 * ((v - c) / w) ** 2)


@pytest.mark.parametrize(
    argnames="model,parameters,function",
    argvalues=[
        (
            ndoptimize.models.gaussian,
            np.array([2.0, 3.0, 15.0]),
            lambda v, p: _gaussian(v, *p),
        ),
        (
            ndoptimize.models.gaussian_linear,
            np.array([2.0, 3.0, 15.0, 0.01, 1.5]),
            lambda v, p: _gaussian(v, *p[:3]) + p[3] * v + p[4],
        ),
        (
            ndoptimize.models.two_gaussians_linear,
            np.array([2.0, 3.0, 15.0, 0.01, 1.5, 1.0, -20.0, 4.0]),
            lambda v, p: _gaussian(v, *p[:3]) + p[3] * v + p[4] + _gaussian(v, *p[5:]),
        ),
    ],
)
def test_models(
    model: ndoptimize._model.Model,
    parameters: np.ndarray,
    function: Callable[[np.ndarray, np.ndarray], np.ndarray],
) -> None:
    v = np.linspace(-50, 50, 21)
    x = v[:, np.newaxis]
    f = np.empty(v.size)
    jacobian = np.empty((v.size, parameters.size))

    model(x, parameters, f, jacobian)

    assert np.allclose(f, function(v, parameters))

    step = 1e-6
    for k in range(parameters.size):
        dp = np.zeros(parameters.size)
        dp[k] = step
        derivative = (function(v, parameters + dp) - function(v, parameters - dp)) / (
            2 * step
        )
        assert np.allclose(jacobian[:, k], derivative, rtol=1e-5, atol=1e-8)
