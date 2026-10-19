from typing import Any, Callable
import copy
import math
import pickle
import numpy as np
import pytest
import scipy.integrate
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
    model: ndoptimize.Model,
    parameters: np.ndarray,
    function: Callable[[np.ndarray, np.ndarray], np.ndarray],
) -> None:
    v = np.linspace(-50, 50, 21)
    x = v[:, np.newaxis]
    f = np.empty(v.size)
    jacobian = np.empty((v.size, parameters.size))

    assert len(model.parameters) == parameters.size

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


@pytest.mark.parametrize(
    argnames="shape_parameters,shape_f,shape_jacobian",
    argvalues=[
        ((2,), (21,), (21, 2)),
        ((3,), (20,), (21, 3)),
        ((3,), (21,), (21, 5)),
    ],
)
def test_model_shapes(
    shape_parameters: tuple[int, ...],
    shape_f: tuple[int, ...],
    shape_jacobian: tuple[int, ...],
) -> None:
    """Calling a model with arrays of the wrong shapes raises an error rather
    than reading or writing outside them."""
    with pytest.raises(ValueError):
        ndoptimize.models.gaussian(
            np.zeros((21, 1)),
            np.ones(shape_parameters),
            np.empty(shape_f),
            np.empty(shape_jacobian),
        )


def test_model_parameters() -> None:
    def constant(
        x: np.ndarray,
        parameters: np.ndarray,
        f: np.ndarray,
        jacobian: np.ndarray,
    ) -> None:
        """A constant."""
        for i in range(x.shape[0]):
            f[i] = parameters[0]
            jacobian[i, 0] = 1

    with pytest.raises(TypeError):
        ndoptimize.model(parameters="offset")
    with pytest.raises(ValueError):
        ndoptimize.model(parameters=("offset", "offset"))

    model = ndoptimize.model(parameters=["offset"], cache=False)(constant)
    assert model.parameters == ("offset",)
    assert model.__doc__ == "A constant."
    assert repr(model) == "Model(parameters=('offset',))"

    f = np.empty(3)
    jacobian = np.empty((3, 1))
    model(np.zeros((3, 1)), np.array([2.0]), f, jacobian)
    assert np.all(f == 2)
    assert np.all(jacobian == 1)


def test_model_uncached() -> None:
    """A model which is not defined in a file can be compiled without the
    cache."""
    source = (
        "@ndoptimize.model(parameters=('slope',), cache=False)\n"
        "def proportional(x, parameters, f, jacobian):\n"
        "    for i in range(x.shape[0]):\n"
        "        f[i] = parameters[0] * x[i, 0]\n"
        "        jacobian[i, 0] = x[i, 0]\n"
    )
    namespace: dict = dict(ndoptimize=ndoptimize)
    exec(source, namespace)
    model = namespace["proportional"]

    x = np.arange(4.0)[:, np.newaxis]
    f = np.empty(4)
    jacobian = np.empty((4, 1))
    model(x, np.array([2.0]), f, jacobian)

    assert np.array_equal(f, 2 * x[:, 0])
    assert np.array_equal(jacobian, x)


def test_model_reference() -> None:
    """A model is copied and pickled by reference, like a function."""
    model = ndoptimize.models.gaussian
    assert copy.copy(model) is model
    assert copy.deepcopy(model) is model
    assert pickle.loads(pickle.dumps(model)) is model


_edges = np.array(
    [
        [-160.0, -150.0],
        [-50.0, -40.0],
        [-12.0, -2.0],
        [-3.0, 8.0],
        [2.0, 3.0],
        [20.0, 35.0],
        [150.0, 160.0],
    ]
)
"""
Bins in both tails, far enough out that the difference of the error functions
cancels, on either side of the center, and across it.
"""


def _gaussian_scalar(v: float, a: float, c: float, w: float) -> float:
    """A Gaussian at a single point, for numerical integration."""
    return a * math.exp(-0.5 * ((v - c) / w) ** 2)


def test_gaussian_binned() -> None:
    """
    The binned Gaussian is the average of the Gaussian over each bin, to full
    precision even far in the tails, and its Jacobian is its derivative.
    """
    model = ndoptimize.models.gaussian_binned
    parameters = np.array([2.0, 3.0, 10.0])
    f = np.empty(len(_edges))
    jacobian = np.empty((len(_edges), 3))

    model(_edges, parameters, f, jacobian)

    expected = np.array(
        [
            scipy.integrate.quad(
                _gaussian_scalar,
                lower,
                upper,
                args=tuple(parameters),
                epsabs=0,
                epsrel=1e-13,
            )[0]
            / (upper - lower)
            for lower, upper in _edges
        ]
    )
    assert np.all(expected[[0, -1]] < 1e-40)
    assert np.allclose(f, expected, rtol=1e-11, atol=0)

    step = 1e-6
    for k in range(parameters.size):
        dp = np.zeros(parameters.size)
        dp[k] = step
        f_plus, f_minus = np.empty(len(_edges)), np.empty(len(_edges))
        model(_edges, parameters + dp, f_plus, np.empty((len(_edges), 3)))
        model(_edges, parameters - dp, f_minus, np.empty((len(_edges), 3)))
        derivative = (f_plus - f_minus) / (2 * step)
        assert np.allclose(jacobian[:, k], derivative, rtol=1e-5, atol=1e-10)


def test_gaussian_binned_narrow() -> None:
    """A bin of zero width gives the Gaussian at that point, and narrow bins
    approach it."""
    parameters = np.array([2.0, 3.0, 10.0])
    v = np.array([-7.0, 3.0, 15.0])

    f = np.empty(v.size)
    jacobian = np.empty((v.size, 3))
    ndoptimize.models.gaussian(v[:, np.newaxis], parameters, f, jacobian)

    for size in (0, 1e-6):
        x = np.stack([v - size / 2, v + size / 2], axis=-1)
        f_binned = np.empty(v.size)
        jacobian_binned = np.empty((v.size, 3))
        ndoptimize.models.gaussian_binned(x, parameters, f_binned, jacobian_binned)
        assert np.allclose(f_binned, f, rtol=1e-9)
        assert np.allclose(jacobian_binned, jacobian, rtol=1e-6, atol=1e-12)


def test_gaussian_binned_fit() -> None:
    """
    Fitting the binned Gaussian to the averages of a line over bins as wide as
    the line recovers the line, while fitting the Gaussian at the centers of
    the bins overestimates its width.
    """
    truth = np.array([5.0, 2.0, 10.0])
    edges = np.linspace(-60, 60, 13)
    x = np.stack([edges[:-1], edges[1:]], axis=-1)
    data = np.empty(len(x))
    ndoptimize.models.gaussian_binned(x, truth, data, np.empty((len(x), 3)))

    kwargs: dict[str, Any] = dict(data=data, variance=1e-4, guess=[4, 0, 8])
    result = ndoptimize.least_squares(
        model=ndoptimize.models.gaussian_binned,
        x=x,
        **kwargs,
    )
    assert result.converged
    assert np.allclose(result.parameters, truth, rtol=1e-6)

    result_sampled = ndoptimize.least_squares(
        model=ndoptimize.models.gaussian,
        x=x.mean(axis=-1, keepdims=True),
        **kwargs,
    )
    assert result_sampled.parameters[2] > 1.02 * truth[2]
