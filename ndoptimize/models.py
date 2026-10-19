"""
Common model functions, compiled with :func:`ndoptimize.model`.
"""

import math
import numpy as np
from . import _model

__all__ = [
    "gaussian",
    "gaussian_linear",
    "two_gaussians_linear",
]


@_model.model
def gaussian(
    x: np.ndarray,
    parameters: np.ndarray,
    f: np.ndarray,
    jacobian: np.ndarray,
) -> None:
    r"""
    A Gaussian,

    .. math::

        f(x) = A \exp \left( -\frac{(x - \mu)^2}{2 \sigma^2} \right),

    with parameters :math:`(A, \mu, \sigma)`, the amplitude, center and
    width, of the first component of :math:`x`.

    Parameters
    ----------
    x
        The points at which to evaluate the model, ``(num_points, num_inputs)``.
    parameters
        The amplitude, center, and width.
    f
        The output array for the model, ``(num_points,)``.
    jacobian
        The output array for the derivatives of the model with respect to
        the parameters, ``(num_points, 3)``.
    """
    amplitude, center, width = parameters[0], parameters[1], parameters[2]
    for i in range(x.shape[0]):
        z = (x[i, 0] - center) / width
        e = math.exp(-0.5 * z * z)
        f[i] = amplitude * e
        jacobian[i, 0] = e
        jacobian[i, 1] = amplitude * e * z / width
        jacobian[i, 2] = amplitude * e * z * z / width


@_model.model
def gaussian_linear(
    x: np.ndarray,
    parameters: np.ndarray,
    f: np.ndarray,
    jacobian: np.ndarray,
) -> None:
    r"""
    A Gaussian plus a straight line,

    .. math::

        f(x) = A \exp \left( -\frac{(x - \mu)^2}{2 \sigma^2} \right) + m x + b,

    with parameters :math:`(A, \mu, \sigma, m, b)`, the amplitude, center,
    width, slope and intercept, of the first component of :math:`x`.
    The line takes up a continuum or background under a spectral line.

    Parameters
    ----------
    x
        The points at which to evaluate the model, ``(num_points, num_inputs)``.
    parameters
        The amplitude, center, width, slope and intercept.
    f
        The output array for the model, ``(num_points,)``.
    jacobian
        The output array for the derivatives of the model with respect to
        the parameters, ``(num_points, 5)``.
    """
    amplitude, center, width = parameters[0], parameters[1], parameters[2]
    slope, intercept = parameters[3], parameters[4]
    for i in range(x.shape[0]):
        v = x[i, 0]
        z = (v - center) / width
        e = math.exp(-0.5 * z * z)
        f[i] = amplitude * e + slope * v + intercept
        jacobian[i, 0] = e
        jacobian[i, 1] = amplitude * e * z / width
        jacobian[i, 2] = amplitude * e * z * z / width
        jacobian[i, 3] = v
        jacobian[i, 4] = 1


@_model.model
def two_gaussians_linear(
    x: np.ndarray,
    parameters: np.ndarray,
    f: np.ndarray,
    jacobian: np.ndarray,
) -> None:
    r"""
    Two Gaussians plus a straight line,

    .. math::

        f(x) = A \exp \left( -\frac{(x - \mu)^2}{2 \sigma^2} \right) + m x + b
            + A_2 \exp \left( -\frac{(x - \mu_2)^2}{2 \sigma_2^2} \right),

    with parameters :math:`(A, \mu, \sigma, m, b, A_2, \mu_2, \sigma_2)`, of
    the first component of :math:`x`.
    The first five parameters are those of :func:`gaussian_linear`, so its
    fit is a natural starting point for this one.

    Parameters
    ----------
    x
        The points at which to evaluate the model, ``(num_points, num_inputs)``.
    parameters
        The amplitude, center, width, slope and intercept, then the
        amplitude, center and width of the second Gaussian.
    f
        The output array for the model, ``(num_points,)``.
    jacobian
        The output array for the derivatives of the model with respect to
        the parameters, ``(num_points, 8)``.
    """
    amplitude, center, width = parameters[0], parameters[1], parameters[2]
    slope, intercept = parameters[3], parameters[4]
    amplitude_2, center_2, width_2 = parameters[5], parameters[6], parameters[7]
    for i in range(x.shape[0]):
        v = x[i, 0]
        z = (v - center) / width
        e = math.exp(-0.5 * z * z)
        z_2 = (v - center_2) / width_2
        e_2 = math.exp(-0.5 * z_2 * z_2)
        f[i] = amplitude * e + slope * v + intercept + amplitude_2 * e_2
        jacobian[i, 0] = e
        jacobian[i, 1] = amplitude * e * z / width
        jacobian[i, 2] = amplitude * e * z * z / width
        jacobian[i, 3] = v
        jacobian[i, 4] = 1
        jacobian[i, 5] = e_2
        jacobian[i, 6] = amplitude_2 * e_2 * z_2 / width_2
        jacobian[i, 7] = amplitude_2 * e_2 * z_2 * z_2 / width_2
