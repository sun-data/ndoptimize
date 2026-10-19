"""
Nonlinear least squares for many independent fits at once.
"""

import dataclasses
import numpy as np
import numba
from numba import types
from ._model import Model, signature_model

__all__ = [
    "LeastSquaresResult",
    "least_squares",
]

_damping_initial = 1e-3
"""The starting value of the Levenberg-Marquardt damping parameter."""

_damping_min = 1e-7
"""The smallest value of the damping parameter."""

_damping_max = 1e7
"""The largest value of the damping parameter, at which a fit stalls."""

_ridge = 1e-12
"""A small number added to the diagonal of the normal matrix, so that it can
be factored even where a parameter has no effect on the model."""


@numba.njit(cache=True, error_model="numpy")
def _cholesky(a: np.ndarray, m: int, w: int) -> bool:
    """
    Overwrite the lower triangle of the leading ``m x m`` block of the
    symmetric matrix `a`, of bandwidth `w`, with its Cholesky factor.
    Only the band of the lower triangle is read.

    Returns :obj:`False` if `a` is not positive definite.
    """
    for j in range(m):
        s = a[j, j]
        for k in range(max(0, j - w), j):
            s -= a[j, k] * a[j, k]
        if not s > 0:
            return False
        d = np.sqrt(s)
        a[j, j] = d
        for i in range(j + 1, min(m, j + w + 1)):
            s = a[i, j]
            for k in range(max(0, i - w), j):
                s -= a[i, k] * a[j, k]
            a[i, j] = s / d
    return True


@numba.njit(cache=True, error_model="numpy")
def _solve_lower(factor: np.ndarray, b: np.ndarray, m: int, w: int) -> None:
    """
    Overwrite the first `m` elements of `b` with the solution of
    ``factor @ z = b`` for a lower-triangular `factor` of bandwidth `w`.
    """
    for i in range(m):
        s = b[i]
        for k in range(max(0, i - w), i):
            s -= factor[i, k] * b[k]
        b[i] = s / factor[i, i]


@numba.njit(cache=True, error_model="numpy")
def _solve_lower_columns(factor: np.ndarray, b: np.ndarray, m: int, w: int) -> None:
    """
    Overwrite the first `m` rows of every column of `b` with the solution of
    ``factor @ z = b`` for a lower-triangular `factor` of bandwidth `w`.
    """
    for i in range(m):
        d = factor[i, i]
        for c in range(b.shape[1]):
            s = b[i, c]
            for k in range(max(0, i - w), i):
                s -= factor[i, k] * b[k, c]
            b[i, c] = s / d


@numba.njit(cache=True, error_model="numpy")
def _solve_positive_definite(
    a: np.ndarray,
    b: np.ndarray,
    work: np.ndarray,
    out: np.ndarray,
) -> bool:
    """
    Solve ``a @ x = b`` into `out` for a small, symmetric, positive-definite
    matrix `a`, using `work` for its Cholesky factor.

    Returns :obj:`False` if `a` is not positive definite.
    """
    n = a.shape[0]
    work[:, :] = a
    if not _cholesky(work, n, n):
        return False
    out[:] = b
    _solve_lower(work, out, n, n)
    for i in range(n - 1, -1, -1):
        s = out[i]
        for k in range(i + 1, n):
            s -= work[k, i] * out[k]
        out[i] = s / work[i, i]
    return True


@numba.njit(cache=True, error_model="numpy")
def _whitened_residual(
    factor: np.ndarray,
    m: int,
    w: int,
    data_whitened: np.ndarray,
    f: np.ndarray,
    out: np.ndarray,
) -> float:
    """
    Compute the whitened residual, ``out = data_whitened - factor^-1 @ f``,
    and return its sum of squares.
    """
    for j in range(m):
        out[j] = f[j]
    _solve_lower(factor, out, m, w)
    total = 0.0
    for j in range(m):
        r = data_whitened[j] - out[j]
        out[j] = r
        total += r * r
    return total


@numba.njit(cache=True, error_model="numpy")
def _whitened_jacobian(
    factor: np.ndarray,
    m: int,
    w: int,
    jacobian: np.ndarray,
    out: np.ndarray,
) -> None:
    """Compute the whitened Jacobian, ``out = factor^-1 @ jacobian``."""
    for j in range(m):
        for k in range(jacobian.shape[1]):
            out[j, k] = jacobian[j, k]
    _solve_lower_columns(factor, out, m, w)


@numba.njit(cache=True, error_model="numpy")
def _normal(
    jacobian_whitened: np.ndarray,
    residual_whitened: np.ndarray,
    m: int,
    parameters: np.ndarray,
    prior_mean: np.ndarray,
    prior_precision: np.ndarray,
    a: np.ndarray,
    g: np.ndarray,
) -> None:
    """
    Compute the normal matrix, ``a = J^T C^-1 J + P``, and the descent
    direction, ``g = J^T C^-1 r - P (p - mean)``, from the whitened Jacobian
    and residual, where ``P`` is the diagonal prior precision.
    """
    n = a.shape[0]
    for k in range(n):
        s = 0.0
        for j in range(m):
            s += jacobian_whitened[j, k] * residual_whitened[j]
        g[k] = s - prior_precision[k] * (parameters[k] - prior_mean[k])
        for q in range(k + 1):
            s = 0.0
            for j in range(m):
                s += jacobian_whitened[j, k] * jacobian_whitened[j, q]
            a[k, q] = s
            a[q, k] = s
        a[k, k] += prior_precision[k]


@numba.njit(cache=True, error_model="numpy")
def _chi2_prior(
    parameters: np.ndarray,
    prior_mean: np.ndarray,
    prior_precision: np.ndarray,
) -> float:
    """The chi-squared of the parameters against their priors."""
    s = 0.0
    for k in range(parameters.shape[0]):
        if prior_precision[k] > 0:
            d = parameters[k] - prior_mean[k]
            s += prior_precision[k] * d * d
    return s


_signature_kernel = types.Tuple(
    (
        types.float64[:, ::1],
        types.float64[:, :, ::1],
        types.float64[::1],
        types.int64[::1],
        types.int64[::1],
        types.boolean[::1],
    )
)(
    types.FunctionType(signature_model),
    types.float64[:, :, ::1],
    types.float64[:, ::1],
    types.float64[:, ::1],
    types.float64[::1],
    types.float64[:, ::1],
    types.float64[:, ::1],
    types.float64[:, ::1],
    types.float64[:, ::1],
    types.float64[:, ::1],
    types.int64,
    types.float64,
    types.int64,
)
"""
The Numba signature of :func:`_levenberg_marquardt`.

The model is a typed function pointer, rather than a dispatcher which would be
compiled into the kernel, because Numba cannot cache a function that takes a
dispatcher argument: such a function is compiled anew in every process.
"""


@numba.njit(_signature_kernel, parallel=True, cache=True, error_model="numpy")
def _levenberg_marquardt(
    model: Model,
    x: np.ndarray,
    data: np.ndarray,
    variance: np.ndarray,
    correlation: np.ndarray,
    guess: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    prior_mean: np.ndarray,
    prior_sigma: np.ndarray,
    max_iterations: int,
    tolerance: float,
    num_threads: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Fit a model to each row of `data` using the Levenberg-Marquardt method.

    The fits are split into `num_threads` interleaved chunks, one per thread,
    so that an expensive region of the data is shared between the threads
    rather than landing on one of them.

    Parameters
    ----------
    model
        The model function, compiled with :func:`ndoptimize.model`.
    x
        The independent variable, ``(num_fits, num_points, num_inputs)``.
    data
        The data, ``(num_fits, num_points)``, NaN where missing.
    variance
        The noise variance of each data point, ``(num_fits, num_points)``.
    correlation
        The correlation of the noise of two points by their separation, in
        units of the sampling of the data, starting with 1.
    guess
        The initial guess for the parameters, ``(num_fits, num_parameters)``.
    lower
        The lower bound of each parameter, ``(num_fits, num_parameters)``.
    upper
        The upper bound of each parameter, ``(num_fits, num_parameters)``.
    prior_mean
        The mean of the Gaussian prior of each parameter,
        ``(num_fits, num_parameters)``.
    prior_sigma
        The standard deviation of the Gaussian prior of each parameter,
        ``(num_fits, num_parameters)``, infinite for no prior.
    max_iterations
        The maximum number of steps per fit.
    tolerance
        The Newton decrement below which a fit has converged.
    num_threads
        The number of chunks to split the fits into, the result of
        :func:`numba.get_num_threads`, which disables caching if it is called
        inside a compiled function.
    """
    num_fit, num_point, num_input = x.shape
    num_parameter = guess.shape[1]
    num_lag = min(correlation.shape[0], num_point)
    w = num_lag - 1

    parameters = np.full((num_fit, num_parameter), np.nan)
    covariance = np.full((num_fit, num_parameter, num_parameter), np.nan)
    chi2 = np.full(num_fit, np.nan)
    num_used = np.zeros(num_fit, dtype=np.int64)
    iterations = np.zeros(num_fit, dtype=np.int64)
    converged = np.zeros(num_fit, dtype=np.bool_)

    num_chunk = max(1, min(num_threads, num_fit))
    for chunk in numba.prange(num_chunk):  # type: ignore[attr-defined]
        xg = np.empty((num_point, num_input))
        data_whitened = np.empty(num_point)
        sigma = np.empty(num_point)
        index = np.empty(num_point, dtype=np.int64)
        factor = np.zeros((num_point, num_point))
        f = np.empty(num_point)
        jac = np.empty((num_point, num_parameter))
        residual = np.empty(num_point)
        residual_new = np.empty(num_point)
        jacobian = np.empty((num_point, num_parameter))
        a = np.empty((num_parameter, num_parameter))
        system = np.empty((num_parameter, num_parameter))
        work = np.empty((num_parameter, num_parameter))
        g = np.empty(num_parameter)
        rhs = np.empty(num_parameter)
        step = np.empty(num_parameter)
        unit = np.empty(num_parameter)
        p = np.empty(num_parameter)
        p_new = np.empty(num_parameter)
        mean = np.empty(num_parameter)
        precision = np.empty(num_parameter)
        free = np.empty(num_parameter, dtype=np.bool_)

        for i in range(chunk, num_fit, num_chunk):

            # gather the usable points
            m = 0
            for j in range(num_point):
                ok = np.isfinite(data[i, j])
                ok = ok and np.isfinite(variance[i, j]) and variance[i, j] > 0
                for q in range(num_input):
                    ok = ok and np.isfinite(x[i, j, q])
                if ok:
                    for q in range(num_input):
                        xg[m, q] = x[i, j, q]
                    data_whitened[m] = data[i, j]
                    sigma[m] = np.sqrt(variance[i, j])
                    index[m] = j
                    m += 1
            num_used[i] = m
            if m == 0:
                continue

            # factor the banded noise covariance of the usable points and
            # whiten the data with it
            for r in range(m):
                for s in range(max(0, r - w), r + 1):
                    lag = index[r] - index[s]
                    rho = correlation[lag] if lag < num_lag else 0.0
                    factor[r, s] = sigma[r] * sigma[s] * rho
            if not _cholesky(factor, m, w):
                continue
            _solve_lower(factor, data_whitened, m, w)

            for k in range(num_parameter):
                precision[k] = 1 / prior_sigma[i, k] ** 2
                mean[k] = prior_mean[i, k] if np.isfinite(prior_mean[i, k]) else 0
                p[k] = min(max(guess[i, k], lower[i, k]), upper[i, k])

            model(xg[:m], p, f[:m], jac[:m])
            chi2_data = _whitened_residual(factor, m, w, data_whitened, f, residual)
            _whitened_jacobian(factor, m, w, jac, jacobian)
            objective = chi2_data + _chi2_prior(p, mean, precision)

            damping = _damping_initial
            nu = 2.0
            done = False
            n_step = 0
            for _ in range(max_iterations + 1):
                _normal(jacobian, residual, m, p, mean, precision, a, g)

                # hold the parameters which are at a bound and are pushed
                # outward by the gradient
                for k in range(num_parameter):
                    at_lower = p[k] <= lower[i, k] and g[k] < 0
                    at_upper = p[k] >= upper[i, k] and g[k] > 0
                    free[k] = not (at_lower or at_upper)

                # the Newton decrement of the free parameters, the decrease
                # in chi-squared a Gauss-Newton step would give if the model
                # were linear
                for k in range(num_parameter):
                    rhs[k] = g[k] if free[k] else 0
                    for q in range(num_parameter):
                        system[k, q] = a[k, q] if (free[k] and free[q]) else 0
                    if free[k]:
                        system[k, k] += _ridge
                    else:
                        system[k, k] = 1
                if _solve_positive_definite(system, rhs, work, step):
                    decrement = 0.0
                    for k in range(num_parameter):
                        decrement += rhs[k] * step[k]
                    if decrement < tolerance:
                        done = True
                        break
                if n_step == max_iterations:
                    break

                # take a damped step, with more damping until the objective
                # decreases
                accepted = False
                chi2_new = np.inf
                objective_new = np.inf
                while not accepted:
                    for k in range(num_parameter):
                        if free[k]:
                            system[k, k] = (a[k, k] + _ridge) * (1 + damping)
                    if _solve_positive_definite(system, rhs, work, step):
                        for k in range(num_parameter):
                            p_new[k] = p[k] + step[k]
                            p_new[k] = min(max(p_new[k], lower[i, k]), upper[i, k])
                        model(xg[:m], p_new, f[:m], jac[:m])
                        chi2_new = _whitened_residual(
                            factor, m, w, data_whitened, f, residual_new
                        )
                        objective_new = chi2_new + _chi2_prior(p_new, mean, precision)
                        accepted = objective_new < objective
                        if np.isnan(objective) and not np.isnan(objective_new):
                            accepted = True
                    if accepted:
                        # Nielsen's rule: the better the quadratic model
                        # predicted the decrease, the less damping
                        predicted = 0.0
                        for k in range(num_parameter):
                            dk = p_new[k] - p[k]
                            predicted += 2 * g[k] * dk
                            for q in range(num_parameter):
                                predicted -= dk * a[k, q] * (p_new[q] - p[q])
                        if predicted > 0:
                            gain = (objective - objective_new) / predicted
                        else:
                            gain = -1.0
                        factor_damping = max(1 / 3, 1 - (2 * gain - 1) ** 3)
                        damping = max(damping * factor_damping, _damping_min)
                        nu = 2.0
                    elif damping >= _damping_max:
                        break
                    else:
                        damping = min(damping * nu, _damping_max)
                        nu *= 2
                n_step += 1
                if not accepted:
                    break

                p, p_new = p_new, p
                residual, residual_new = residual_new, residual
                chi2_data = chi2_new
                objective = objective_new
                _whitened_jacobian(factor, m, w, jac, jacobian)

            # the covariance of the parameters is the inverse of the normal
            # matrix, with infinite variance for a parameter which nothing
            # constrains, and a small ridge if the rest is still singular
            _normal(jacobian, residual, m, p, mean, precision, a, g)
            diagonal_max = 0.0
            for k in range(num_parameter):
                diagonal_max = max(diagonal_max, a[k, k])
            for attempt in range(2):
                for k in range(num_parameter):
                    for q in range(num_parameter):
                        constrained = a[k, k] > 0 and a[q, q] > 0
                        system[k, q] = a[k, q] if constrained else 0
                    if not a[k, k] > 0:
                        system[k, k] = 1
                    elif attempt > 0:
                        system[k, k] += _ridge * diagonal_max
                invertible = True
                for k in range(num_parameter):
                    for q in range(num_parameter):
                        unit[q] = 1 if q == k else 0
                    if not _solve_positive_definite(system, unit, work, step):
                        invertible = False
                        break
                    for q in range(num_parameter):
                        covariance[i, q, k] = step[q]
                if invertible:
                    break
            for k in range(num_parameter):
                if not a[k, k] > 0:
                    for q in range(num_parameter):
                        covariance[i, k, q] = 0
                        covariance[i, q, k] = 0
                    covariance[i, k, k] = np.inf

            for k in range(num_parameter):
                parameters[i, k] = p[k]
            chi2[i] = chi2_data
            iterations[i] = n_step
            converged[i] = done

    return parameters, covariance, chi2, num_used, iterations, converged


@dataclasses.dataclass(frozen=True)
class LeastSquaresResult:
    """
    The results of :func:`least_squares` for every fit of a batch.

    The leading axes of each array are the batch axes of the fit.
    """

    parameters: np.ndarray
    """The best-fit parameters, ``(..., num_parameters)``."""

    covariance: np.ndarray
    """
    The covariance of the best-fit parameters,
    ``(..., num_parameters, num_parameters)``.

    This is the inverse of the normal matrix, :math:`J^T C^{-1} J`, plus the
    precision of the priors, at the solution.
    A parameter which neither the data nor its prior constrains has an
    infinite variance.
    """

    chi2: np.ndarray
    """
    The chi-squared of the data at the solution, without the contribution of
    the priors.
    """

    num_points: np.ndarray
    """The number of data points used by each fit."""

    iterations: np.ndarray
    """The number of steps taken by each fit."""

    converged: np.ndarray
    """Whether each fit converged."""


def _flatten(
    x: np.ndarray,
    data: np.ndarray,
    variance: float | np.ndarray,
    guess: np.ndarray,
    lower: float | np.ndarray,
    upper: float | np.ndarray,
    prior_mean: float | np.ndarray,
    prior_sigma: float | np.ndarray,
) -> tuple[tuple[int, ...], dict[str, np.ndarray]]:
    """
    Broadcast the arguments of :func:`least_squares` against each other over
    their batch axes and flatten those axes, into writable, contiguous
    arrays of the types the kernel's signature requires.

    Returns the batch shape and the flattened arrays.
    """
    x = np.asarray(x, dtype=float)
    data = np.asarray(data, dtype=float)
    guess = np.asarray(guess, dtype=float)

    if x.ndim < 2:
        raise ValueError(
            f"`x` must have at least two axes, (..., num_points, num_inputs), "
            f"got shape {x.shape}."
        )
    num_point, num_input = x.shape[-2:]
    num_parameter = guess.shape[-1]

    if data.shape[-1:] != (num_point,):
        raise ValueError(
            f"The last axis of `data`, {data.shape}, must match the "
            f"number of points in `x`, {num_point}."
        )

    variance = np.broadcast_to(
        variance, np.broadcast_shapes(np.shape(variance), (num_point,))
    )
    lower, upper, prior_mean, prior_sigma = (
        np.broadcast_to(a, np.broadcast_shapes(np.shape(a), (num_parameter,)))
        for a in (lower, upper, prior_mean, prior_sigma)
    )

    shape = np.broadcast_shapes(
        x.shape[:-2],
        data.shape[:-1],
        variance.shape[:-1],
        guess.shape[:-1],
        lower.shape[:-1],
        upper.shape[:-1],
        prior_mean.shape[:-1],
        prior_sigma.shape[:-1],
    )
    num_fit = int(np.prod(shape))

    def flat(a: np.ndarray, *shape_trailing: int) -> np.ndarray:
        """A writable, contiguous copy of `a` with its batch axes flattened."""
        a = np.broadcast_to(a, shape + shape_trailing)
        return np.array(a, dtype=float, order="C").reshape(num_fit, *shape_trailing)

    arrays = dict(
        x=flat(x, num_point, num_input),
        data=flat(data, num_point),
        variance=flat(variance, num_point),
        guess=flat(guess, num_parameter),
        lower=flat(lower, num_parameter),
        upper=flat(upper, num_parameter),
        prior_mean=flat(prior_mean, num_parameter),
        prior_sigma=flat(prior_sigma, num_parameter),
    )

    return shape, arrays


def least_squares(
    model: Model,
    x: np.ndarray,
    data: np.ndarray,
    variance: float | np.ndarray,
    guess: np.ndarray,
    lower: float | np.ndarray = -np.inf,
    upper: float | np.ndarray = np.inf,
    prior_mean: float | np.ndarray = 0,
    prior_sigma: float | np.ndarray = np.inf,
    correlation: None | np.ndarray = None,
    max_iterations: int = 1000,
    tolerance: float = 1e-8,
) -> LeastSquaresResult:
    r"""
    Fit a model to many independent sets of data at once, using the
    Levenberg-Marquardt method.

    Every fit has the same model, but its own independent variable, data,
    noise, initial guess, bounds and priors.
    The fits run in parallel, each with its own damping and convergence test.

    Each fit minimizes

    .. math::

        \chi^2 = (y - f)^T C^{-1} (y - f)
            + \sum_k \left( \frac{p_k - \bar{p}_k}{\sigma_k} \right)^2,

    where :math:`y` is the data, :math:`f` is the model,
    :math:`p_k` are the parameters, :math:`\bar{p}_k` and :math:`\sigma_k`
    are the means and standard deviations of their Gaussian priors,
    and :math:`C` is the covariance of the noise,

    .. math::

        C_{ij} = \sigma_i \sigma_j \rho_{|i - j|},

    with :math:`\sigma_i^2` the variance of data point :math:`i`
    and :math:`\rho_k` the correlation of the noise of two points
    :math:`k` samples apart.
    Uncorrelated noise, :math:`\rho = (1,)`, is ordinary weighted least
    squares.
    Correlated noise, from resampling for instance, makes this generalized
    least squares, which is computed by whitening the residual and the
    Jacobian with the banded Cholesky factor of :math:`C`.

    Parameters
    ----------
    model
        The model to fit, a function compiled with :func:`ndoptimize.model`.
    x
        The independent variable, an array of shape
        ``(..., num_points, num_inputs)``.
    data
        The data to fit, an array of shape ``(..., num_points)``.
        Points which are NaN are left out of their fit.
    variance
        The variance of the noise of each data point, an array of shape
        ``(..., num_points)``.
        Points with a variance which is NaN or not positive are left out of
        their fit.
    guess
        The initial guess for the parameters, an array of shape
        ``(..., num_parameters)``.
    lower
        The lower bound of each parameter, an array of shape
        ``(..., num_parameters)``.
    upper
        The upper bound of each parameter, an array of shape
        ``(..., num_parameters)``.
    prior_mean
        The mean of the Gaussian prior of each parameter, an array of shape
        ``(..., num_parameters)``.
    prior_sigma
        The standard deviation of the Gaussian prior of each parameter, an
        array of shape ``(..., num_parameters)``.
        An infinite standard deviation, the default, means no prior.
    correlation
        The correlation of the noise of two data points by their separation
        along the last axis of `data`, in samples, starting with 1 for no
        separation and zero beyond its length.
        The separation counts the points left out of a fit.
        If :obj:`None` (the default), the noise is uncorrelated.
    max_iterations
        The maximum number of steps for each fit.
    tolerance
        The convergence tolerance on the Newton decrement,
        :math:`g^T A^{-1} g`, where :math:`g = J^T C^{-1} r` is the gradient
        and :math:`A = J^T C^{-1} J` is the normal matrix, which is the
        decrease in :math:`\chi^2` a Gauss-Newton step would give if the
        model were linear.
        The default, :math:`10^{-8}`, puts each parameter within about
        :math:`10^{-4}` of its uncertainty of the minimum.

    Returns
    -------
    The best-fit parameters, their covariance, chi-squared, and the
    convergence state of each fit.
    Fits with no usable data points are NaN.

    Notes
    -----

    The batch axes of the arguments, all but their last one (two for `x`),
    are broadcast against each other.

    Each step solves :math:`(A + \lambda \, \text{diag} A) \delta = g`,
    Marquardt's damping, and the damping parameter :math:`\lambda` is updated
    by the gain-ratio rule of Madsen, Nielsen, and Tingleff (2004), "Methods
    for Non-Linear Least Squares Problems", which converges in curved
    valleys of :math:`\chi^2` (where the amplitude and width of a weak line
    trade off) in about a third of the steps of the classic fixed factors.

    Steps are clipped to the bounds, and a parameter at a bound which the
    gradient pushes outward is held fixed for the step and for the
    convergence test.

    Examples
    --------

    Fit a Gaussian plus a straight line to a thousand noisy spectral lines.

    .. jupyter-execute::

        import numpy as np
        import matplotlib.pyplot as plt
        import ndoptimize

        rng = np.random.default_rng(0)

        # the true parameters: amplitude, center, width, slope and intercept
        num = 1000
        truth = np.stack([
            rng.uniform(2, 10, num),
            rng.normal(0, 5, num),
            rng.uniform(10, 20, num),
            np.zeros(num),
            rng.uniform(1, 2, num),
        ], axis=-1)

        # the noisy spectra
        velocity = np.linspace(-100, 100, 41)
        x = velocity[:, np.newaxis]
        f = np.empty((num, velocity.size))
        jacobian = np.empty((velocity.size, 5))
        for i in range(num):
            ndoptimize.models.gaussian_linear(x, truth[i], f[i], jacobian)
        variance = 0.5
        data = f + rng.normal(0, np.sqrt(variance), f.shape)

        # fit every spectrum
        result = ndoptimize.least_squares(
            model=ndoptimize.models.gaussian_linear,
            x=x,
            data=data,
            variance=variance,
            guess=[5, 0, 15, 0, 1],
        )

        # compare the fitted centers to the truth
        sigma = np.sqrt(result.covariance[..., 1, 1])
        error = (result.parameters[..., 1] - truth[..., 1]) / sigma

        fig, ax = plt.subplots(constrained_layout=True)
        ax.hist(error, bins=40, density=True, label="fits");
        z = np.linspace(-4, 4, 101)
        ax.plot(z, np.exp(-z**2 / 2) / np.sqrt(2 * np.pi), label="unit normal");
        ax.set_xlabel("(fitted center - true center) / uncertainty");
        ax.legend();
    """
    shape, a = _flatten(
        x=x,
        data=data,
        variance=variance,
        guess=guess,
        lower=lower,
        upper=upper,
        prior_mean=prior_mean,
        prior_sigma=prior_sigma,
    )

    if correlation is None:
        correlation = np.ones(1)
    correlation = np.array(correlation, dtype=float).reshape(-1)
    if correlation.size == 0 or correlation[0] != 1:
        raise ValueError(
            f"`correlation` must start with 1, the correlation of a point "
            f"with itself, got {correlation}."
        )

    # the noise correlation of every fit is a principal submatrix of this
    # one, so it is positive definite if this one is
    num_point = a["x"].shape[1]
    rho = np.zeros(num_point)
    rho[: correlation.size] = correlation[:num_point]
    index = np.arange(num_point)
    try:
        np.linalg.cholesky(rho[np.abs(index[:, np.newaxis] - index)])
    except np.linalg.LinAlgError:
        raise ValueError(
            f"`correlation`, {correlation}, does not describe a positive "
            f"definite noise covariance for {num_point} points."
        )

    result = _levenberg_marquardt(
        model,
        a["x"],
        a["data"],
        a["variance"],
        correlation,
        a["guess"],
        a["lower"],
        a["upper"],
        a["prior_mean"],
        a["prior_sigma"],
        max_iterations,
        tolerance,
        numba.get_num_threads(),
    )

    parameters, covariance, chi2, num_points, iterations, converged = (
        r.reshape(shape + r.shape[1:]) for r in result
    )

    return LeastSquaresResult(
        parameters=parameters,
        covariance=covariance,
        chi2=chi2,
        num_points=num_points,
        iterations=iterations,
        converged=converged,
    )
