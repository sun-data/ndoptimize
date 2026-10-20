"""
Evaluation of a model for many sets of parameters at once.
"""

import math
import numpy as np
import numpy.typing as npt
import numba
from numba import types
from ._model import ModelFunction, Model, signature_model
from ._least_squares import _input

__all__ = [
    "evaluate",
]

_signature_kernel = types.float64[:, ::1](
    types.FunctionType(signature_model),
    _input(3),
    _input(2),
    types.int64,
)
"""The Numba signature of :func:`_evaluate`."""


@numba.njit(_signature_kernel, parallel=True, cache=True, error_model="numpy")
def _evaluate(
    model: ModelFunction,
    x: np.ndarray,
    parameters: np.ndarray,
    num_threads: int,
) -> np.ndarray:
    """
    Evaluate a model for each row of `parameters`.

    Parameters
    ----------
    model
        The compiled model function, :attr:`ndoptimize.Model.function`.
    x
        The independent variable, ``(num_sets, num_points, num_inputs)``.
    parameters
        The parameters, ``(num_sets, num_parameters)``.
    num_threads
        The number of chunks to split the sets of parameters into, the result
        of :func:`numba.get_num_threads`, which disables caching if it is
        called inside a compiled function.
    """
    num_set, num_point, num_input = x.shape
    num_parameter = parameters.shape[1]

    f = np.empty((num_set, num_point))

    num_chunk = max(1, min(num_threads, num_set))
    for chunk in numba.prange(num_chunk):  # type: ignore[attr-defined]
        xi = np.empty((num_point, num_input))
        p = np.empty(num_parameter)
        jacobian = np.empty((num_point, num_parameter))

        for i in range(chunk, num_set, num_chunk):
            for j in range(num_point):
                for q in range(num_input):
                    xi[j, q] = x[i, j, q]
            for k in range(num_parameter):
                p[k] = parameters[i, k]
            model(xi, p, f[i], jacobian)

    return f


def evaluate(
    model: Model,
    x: npt.ArrayLike,
    parameters: npt.ArrayLike,
) -> np.ndarray:
    """
    Evaluate a model for many sets of parameters at once, such as the results
    of :func:`ndoptimize.least_squares`.

    The sets run in parallel.

    Parameters
    ----------
    model
        The model to evaluate, made with :func:`ndoptimize.model`.
    x
        The points at which to evaluate the model, an array of shape
        ``(..., num_points, num_inputs)``, where ``num_inputs`` is the number
        of names in :attr:`ndoptimize.Model.inputs`.
    parameters
        The parameters of the model, an array of shape
        ``(..., num_parameters)``, where ``num_parameters`` is the number of
        names in :attr:`ndoptimize.Model.parameters`.

    Returns
    -------
    The model evaluated at `x`, an array of shape ``(..., num_points)``,
    where the batch axes of `x` and `parameters`, all but their last one (two
    for `x`), are broadcast against each other.

    Examples
    --------

    Evaluate Gaussians of three different widths.

    .. jupyter-execute::

        import numpy as np
        import matplotlib.pyplot as plt
        import ndoptimize

        x = np.linspace(-50, 50, 101)
        parameters = np.array([
            [1, 0, 5],
            [1, 0, 10],
            [1, 0, 20],
        ])

        f = ndoptimize.evaluate(
            model=ndoptimize.models.gaussian,
            x=x[:, np.newaxis],
            parameters=parameters,
        )

        fig, ax = plt.subplots(constrained_layout=True)
        ax.plot(x, f.T, label=[f"width {w}" for w in parameters[:, 2]]);
        ax.legend();
    """
    return _evaluate_arrays(
        model=model,
        x=np.asarray(x, dtype=float),
        parameters=np.asarray(parameters, dtype=float),
    )


def _evaluate_arrays(
    model: Model,
    x: np.ndarray,
    parameters: np.ndarray,
) -> np.ndarray:
    """:func:`evaluate` for arguments which are already arrays of floats."""
    model._check_shapes(x, parameters, "parameters")
    num_point, num_input = x.shape[-2:]
    num_parameter = len(model.parameters)

    shape = np.broadcast_shapes(x.shape[:-2], parameters.shape[:-1])
    num_set = math.prod(shape)

    x = np.broadcast_to(x, shape + (num_point, num_input))
    parameters = np.broadcast_to(parameters, shape + (num_parameter,))

    f = _evaluate(
        model.function,
        x.reshape(num_set, num_point, num_input),
        parameters.reshape(num_set, num_parameter),
        numba.get_num_threads(),
    )

    return f.reshape(shape + (num_point,))
