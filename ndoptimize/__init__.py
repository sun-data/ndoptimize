"""
Numba-accelerated minimization of objective functions.
"""

from ._model import model
from . import models
from ._least_squares import LeastSquaresResult, least_squares

__all__ = [
    "model",
    "models",
    "LeastSquaresResult",
    "least_squares",
]
