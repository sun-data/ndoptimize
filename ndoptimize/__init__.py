"""
Numba-accelerated minimization of objective functions.
"""

from ._model import Model, model
from . import models
from ._least_squares import LeastSquaresResult, least_squares

__all__ = [
    "Model",
    "model",
    "models",
    "LeastSquaresResult",
    "least_squares",
]
