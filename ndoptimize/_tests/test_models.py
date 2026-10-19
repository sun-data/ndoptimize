from typing import Callable
import copy
import pickle
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
