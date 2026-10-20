"""
The interface between a model function and the fitting kernels.
"""

from typing import Callable, Sequence
import dataclasses
import functools
import numpy as np
import numba
from numba import types

__all__ = [
    "ModelFunction",
    "signature_model",
    "Model",
    "model",
]

ModelFunction = Callable[[np.ndarray, np.ndarray, np.ndarray, np.ndarray], None]
"""
The type of a model function, ``function(x, parameters, f, jacobian)``.
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


@dataclasses.dataclass(eq=False)
class Model:
    """
    A model which :func:`ndoptimize.least_squares` can fit: a compiled model
    function and the names of its parameters and inputs, made with
    :func:`model`.

    Calling a model evaluates it, after checking the shapes of the arrays,
    since the compiled function does not check its indices.
    """

    function: ModelFunction
    """
    The compiled model function, ``function(x, parameters, f, jacobian)``,
    described in :func:`model`.
    """

    parameters: tuple[str, ...]
    """
    The names of the parameters, in the order of the ``parameters`` array
    and of the columns of the ``jacobian`` array.
    """

    inputs: tuple[str, ...]
    """
    The names of the inputs, in the order of the columns of the ``x`` array.
    """

    def __post_init__(self) -> None:
        functools.update_wrapper(self, self.function, updated=())

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(parameters={self.parameters}, "
            f"inputs={self.inputs})"
        )

    def __get__(self, instance: object, owner: None | type = None) -> "Model":
        """
        Return this model unchanged when it is an attribute of a class, like
        a static method, rather than binding it to an instance.

        This also makes it a method descriptor, which documentation tools
        treat as a function.
        """
        return self

    def __reduce__(self) -> str:
        """
        Pickle and copy this model by reference, like a function: a copy is
        this model itself, and a pickled model is the one of the same name in
        its module when it is unpickled.
        """
        return self.function.__qualname__

    def __call__(
        self,
        x: np.ndarray,
        parameters: np.ndarray,
        f: np.ndarray,
        jacobian: np.ndarray,
    ) -> None:
        """
        Evaluate the model and its Jacobian.

        Parameters
        ----------
        x
            The points at which to evaluate the model,
            ``(num_points, num_inputs)``, where ``num_inputs`` is the number
            of names in :attr:`inputs`.
        parameters
            The parameters of the model, ``(num_parameters,)``.
        f
            The output array for the model, ``(num_points,)``.
        jacobian
            The output array for the derivatives of the model with respect
            to the parameters, ``(num_points, num_parameters)``.
        """
        num_point = np.shape(x)[0] if np.ndim(x) == 2 else -1
        num_input = len(self.inputs)
        num_parameter = len(self.parameters)
        if (
            np.ndim(x) != 2
            or np.shape(x)[1] != num_input
            or np.shape(parameters) != (num_parameter,)
            or np.shape(f) != (num_point,)
            or np.shape(jacobian) != (num_point, num_parameter)
        ):
            raise ValueError(
                f"For a model with {num_input} inputs and {num_parameter} "
                f"parameters, `x` must have shape (num_points, {num_input}), "
                f"`parameters` ({num_parameter},), `f` (num_points,) and "
                f"`jacobian` (num_points, {num_parameter}), got {np.shape(x)}, "
                f"{np.shape(parameters)}, {np.shape(f)} and "
                f"{np.shape(jacobian)}."
            )
        self.function(x, parameters, f, jacobian)


def _names(names: Sequence[str], argument: str) -> tuple[str, ...]:
    """
    Check that `names`, the argument of :func:`model` named `argument`, is a
    sequence of unique names, and return it as a tuple.
    """
    if isinstance(names, str):
        raise TypeError(
            f"`{argument}` must be a sequence of names, got the string {names!r}."
        )
    names = tuple(names)
    if len(set(names)) != len(names):
        raise ValueError(f"The names in `{argument}`, {names}, must be unique.")
    return names


def model(
    parameters: Sequence[str],
    inputs: Sequence[str],
    cache: bool = True,
) -> Callable[[ModelFunction], Model]:
    r"""
    Compile a function as a model which :func:`ndoptimize.least_squares` can
    fit, with the given names for its parameters and inputs.

    A model function, ``function(x, parameters, f, jacobian)``, writes the
    model evaluated at the points ``x``, an array of shape
    ``(num_points, num_inputs)`` with a column for each input, for the given
    ``parameters``, an array of shape ``(num_parameters,)``, into ``f``, an
    array of shape ``(num_points,)``, and its derivatives with respect to the
    parameters into ``jacobian``, an array of shape
    ``(num_points, num_parameters)``.
    It returns nothing.

    The function is compiled with Numba in `nopython` mode, with this fixed
    signature.
    The fitting kernel takes the function as a typed function pointer, so a
    single compiled kernel serves every model and is cached on disk.

    Scalar functions such as :func:`math.exp` should come from :mod:`math`,
    since they are the ones Numba compiles for every target.

    Parameters
    ----------
    parameters
        The names of the parameters, in the order of the ``parameters``
        array and of the columns of the ``jacobian`` array.
        :func:`ndoptimize.least_squares` checks the initial guess against
        their number, since the compiled function does not check its
        indices.
    inputs
        The names of the inputs, in the order of the columns of the ``x``
        array, such as ``("x",)`` for a function of one variable, or
        ``("lower", "upper")`` for a function of the edges of a bin.
        :func:`ndoptimize.least_squares` and :func:`ndoptimize.evaluate`
        check the last axis of ``x`` against their number, for the same
        reason.
    cache
        Whether to cache the compiled function on disk, so that later
        processes need not compile it again.
        Numba compiles the global arrays a function reads into it as
        constants, and the cache does not notice when they change, so a model
        which reads globals computed at run time should not be cached.
        Neither can a model which is not defined in a file, such as one
        defined in the interactive interpreter.

    Examples
    --------

    A straight line, :math:`f(x) = a + b x`.

    .. jupyter-execute::

        import ndoptimize

        @ndoptimize.model(parameters=("intercept", "slope"), inputs=("x",))
        def line(x, parameters, f, jacobian):
            a, b = parameters[0], parameters[1]
            for i in range(x.shape[0]):
                f[i] = a + b * x[i, 0]
                jacobian[i, 0] = 1
                jacobian[i, 1] = x[i, 0]

        line
    """
    parameters = _names(parameters, "parameters")
    inputs = _names(inputs, "inputs")

    def decorator(function: ModelFunction) -> Model:
        compiled = numba.njit(signature_model, cache=cache, error_model="numpy")
        return Model(
            function=compiled(function),
            parameters=parameters,
            inputs=inputs,
        )

    return decorator
