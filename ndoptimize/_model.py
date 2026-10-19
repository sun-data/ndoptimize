"""
The interface between a model function and the fitting kernels.
"""

from typing import Callable
import numpy as np
import numba
from numba import types

__all__ = [
    "Model",
    "signature_model",
    "model",
]

Model = Callable[[np.ndarray, np.ndarray, np.ndarray, np.ndarray], None]
"""
The type of a model function, ``model(x, parameters, f, jacobian)``.
"""

signature_model = types.void(
    types.float64[:, :],
    types.float64[:],
    types.float64[:],
    types.float64[:, :],
)
"""
The Numba signature of a model function.
"""


def model(function: Model) -> Model:
    r"""
    Compile a function as a model which :func:`ndoptimize.least_squares` can fit.

    A model is a function ``model(x, parameters, f, jacobian)`` which writes
    the model evaluated at the points ``x``, an array of shape
    ``(num_points, num_inputs)``, for the given ``parameters``, an array of
    shape ``(num_parameters,)``, into ``f``, an array of shape
    ``(num_points,)``, and its derivatives with respect to the parameters
    into ``jacobian``, an array of shape ``(num_points, num_parameters)``.
    It returns nothing.

    The function is compiled with Numba in `nopython` mode, with this fixed
    signature.
    The fitting kernel takes the model as a typed function pointer, so a
    single compiled kernel serves every model and is cached on disk.

    Scalar functions such as :func:`math.exp` should come from :mod:`math`,
    since they are the ones Numba compiles for every target.

    Parameters
    ----------
    function
        The model function to compile.

    Examples
    --------

    A straight line, :math:`f(x) = a + b x`.

    .. jupyter-execute::

        import ndoptimize

        @ndoptimize.model
        def line(x, parameters, f, jacobian):
            a, b = parameters[0], parameters[1]
            for i in range(x.shape[0]):
                f[i] = a + b * x[i, 0]
                jacobian[i, 0] = 1
                jacobian[i, 1] = x[i, 0]
    """
    return numba.njit(signature_model, cache=True, error_model="numpy")(function)
