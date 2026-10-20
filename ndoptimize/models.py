"""
Common models, made with :func:`ndoptimize.model`.
"""

import math
import numpy as np
from . import _model

__all__ = [
    "gaussian",
    "gaussian_binned",
    "gaussian_linear",
    "two_gaussians_linear",
]


@_model.model(
    parameters=("amplitude", "center", "width"),
    inputs=("x",),
)
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
    width, of a single input, :math:`x`.

    Parameters
    ----------
    x
        The points at which to evaluate the model, ``(num_points, 1)``.
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


@_model.model(
    parameters=("amplitude", "center", "width"),
    inputs=("lower", "upper"),
)
def gaussian_binned(
    x: np.ndarray,
    parameters: np.ndarray,
    f: np.ndarray,
    jacobian: np.ndarray,
) -> None:
    r"""
    A Gaussian averaged over each bin,

    .. math::

        f(x_0, x_1) = \frac{1}{x_1 - x_0} \int_{x_0}^{x_1}
            A \exp \left( -\frac{(x - \mu)^2}{2 \sigma^2} \right) dx
            = \sqrt{\frac{\pi}{2}} \frac{A \sigma}{x_1 - x_0} \left[
                \text{erf} \left( \frac{x_1 - \mu}{\sqrt{2} \sigma} \right)
                - \text{erf} \left( \frac{x_0 - \mu}{\sqrt{2} \sigma} \right)
            \right],

    with the same parameters as :func:`gaussian`, the amplitude, center and
    width, where its two inputs, ``lower`` and ``upper``, are the edges of
    each bin, :math:`x_0` and :math:`x_1`.
    The integral of the Gaussian is :math:`\sqrt{2 \pi} A \sigma`.

    Averaging matters when the bins are not much narrower than the line.
    Averaging over bins of width :math:`\Delta` broadens a Gaussian by about
    :math:`\Delta^2 / 12` in variance, which a fit of :func:`gaussian` at the
    centers of the bins would take as part of the width of the line.
    A bin of zero width gives the Gaussian at that point.

    Parameters
    ----------
    x
        The edges of each bin, ``(num_points, 2)``, with the lower edge in the
        first column and the upper edge in the second.
    parameters
        The amplitude, center, and width.
    f
        The output array for the model, ``(num_points,)``.
    jacobian
        The output array for the derivatives of the model with respect to
        the parameters, ``(num_points, 3)``.

    Examples
    --------

    A Gaussian averaged over bins as wide as the line, compared to the
    Gaussian itself.

    .. jupyter-execute::

        import numpy as np
        import matplotlib.pyplot as plt
        import ndoptimize

        parameters = np.array([1.0, 3.0, 10.0])  # amplitude, center and width

        # the Gaussian
        v = np.linspace(-50, 50, 201)
        f = np.empty(v.size)
        ndoptimize.models.gaussian(v[:, np.newaxis], parameters, f, np.empty((v.size, 3)))

        # its average over each bin
        edges = np.linspace(-50, 50, 11)
        x = np.stack([edges[:-1], edges[1:]], axis=-1)
        f_binned = np.empty(len(x))
        ndoptimize.models.gaussian_binned(x, parameters, f_binned, np.empty((len(x), 3)))

        fig, ax = plt.subplots(constrained_layout=True)
        ax.plot(v, f, label="gaussian");
        ax.stairs(f_binned, edges, label="gaussian_binned");
        ax.legend();
    """
    amplitude, center, width = parameters[0], parameters[1], parameters[2]
    scale = math.sqrt(2) * width
    for i in range(x.shape[0]):
        lower = x[i, 0]
        upper = x[i, 1]
        size = upper - lower
        if size == 0:
            # the limit of a narrow bin, the Gaussian at that point
            z = (lower - center) / width
            e = math.exp(-0.5 * z * z)
            f[i] = amplitude * e
            jacobian[i, 0] = e
            jacobian[i, 1] = amplitude * e * z / width
            jacobian[i, 2] = amplitude * e * z * z / width
            continue
        a = (lower - center) / scale
        b = (upper - center) / scale
        # in a tail, where both error functions are close to 1 or to -1,
        # their difference comes from the complementary error functions
        # without cancellation
        if a > 0 and b > 0:
            difference = math.erfc(a) - math.erfc(b)
        elif a < 0 and b < 0:
            difference = math.erfc(-b) - math.erfc(-a)
        else:
            difference = math.erf(b) - math.erf(a)
        e_a = math.exp(-a * a)
        e_b = math.exp(-b * b)
        factor = math.sqrt(math.pi / 2) * width / size
        f[i] = amplitude * factor * difference
        jacobian[i, 0] = factor * difference
        jacobian[i, 1] = amplitude * (e_a - e_b) / size
        jacobian[i, 2] = (
            amplitude
            * (math.sqrt(math.pi / 2) * difference + math.sqrt(2) * (a * e_a - b * e_b))
            / size
        )


@_model.model(
    parameters=("amplitude", "center", "width", "slope", "intercept"),
    inputs=("x",),
)
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
    width, slope and intercept, of a single input, :math:`x`.
    The line takes up a continuum or background under a spectral line.

    Parameters
    ----------
    x
        The points at which to evaluate the model, ``(num_points, 1)``.
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


@_model.model(
    parameters=(
        "amplitude",
        "center",
        "width",
        "slope",
        "intercept",
        "amplitude_2",
        "center_2",
        "width_2",
    ),
    inputs=("x",),
)
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
    a single input, :math:`x`.
    The first five parameters are those of :func:`gaussian_linear`, so its
    fit is a natural starting point for this one.

    Parameters
    ----------
    x
        The points at which to evaluate the model, ``(num_points, 1)``.
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
