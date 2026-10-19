"""
Nonlinear least squares for many independent fits at once.
"""

import dataclasses
import numpy as np
import numpy.typing as npt
import numba
from numba import types
from ._model import ModelFunction, Model, signature_model

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
"""
The fraction of each diagonal element of the normal matrix which is added to
it for the convergence test, so that it can be factored where the parameters
are degenerate.
"""

_degenerate = 1e-12
"""
The smallest ratio of the precision of a parameter, given the parameters
before it, to its precision alone, below which the normal matrix is singular
and the covariance of the parameters is NaN.
"""


@numba.njit(cache=True, error_model="numpy")
def _cholesky_band(band: np.ndarray, m: int) -> bool:
    """
    Overwrite the first `m` rows of `band`, the lower band of a symmetric
    matrix with ``band[i, d]`` its element ``(i, i - d)``, with the same band
    of its Cholesky factor.

    Returns :obj:`False` if the matrix is not positive definite.
    """
    w = band.shape[1] - 1
    for j in range(m):
        s = band[j, 0]
        for d in range(1, min(j, w) + 1):
            s -= band[j, d] * band[j, d]
        if not s > 0:
            return False
        pivot = np.sqrt(s)
        band[j, 0] = pivot
        for i in range(j + 1, min(m, j + w + 1)):
            s = band[i, i - j]
            for k in range(max(0, i - w), j):
                s -= band[i, i - k] * band[j, j - k]
            band[i, i - j] = s / pivot
    return True


@numba.njit(cache=True, error_model="numpy")
def _solve_lower_band(band: np.ndarray, b: np.ndarray, m: int) -> None:
    """
    Overwrite the first `m` elements of `b` with the solution of
    ``factor @ z = b``, for the lower-triangular `factor` stored as a band by
    :func:`_cholesky_band`.
    """
    w = band.shape[1] - 1
    for i in range(m):
        s = b[i]
        for k in range(max(0, i - w), i):
            s -= band[i, i - k] * b[k]
        b[i] = s / band[i, 0]


@numba.njit(cache=True, error_model="numpy")
def _solve_lower_band_columns(band: np.ndarray, b: np.ndarray, m: int) -> None:
    """
    Overwrite the first `m` rows of every column of `b` with the solution of
    ``factor @ z = b``, for the lower-triangular `factor` stored as a band by
    :func:`_cholesky_band`.
    """
    w = band.shape[1] - 1
    for i in range(m):
        pivot = band[i, 0]
        for c in range(b.shape[1]):
            s = b[i, c]
            for k in range(max(0, i - w), i):
                s -= band[i, i - k] * b[k, c]
            b[i, c] = s / pivot


@numba.njit(cache=True, error_model="numpy")
def _cholesky(a: np.ndarray, degenerate: float) -> bool:
    """
    Overwrite the lower triangle of the small symmetric matrix `a` with its
    Cholesky factor.

    Returns :obj:`False` if `a` is not positive definite, or if a pivot is
    no more than `degenerate` times the diagonal element it came from.
    For a normal matrix, that ratio is the precision of a parameter, given
    the parameters before it, over its precision alone.
    """
    n = a.shape[0]
    for j in range(n):
        s = a[j, j]
        for k in range(j):
            s -= a[j, k] * a[j, k]
        if not s > degenerate * a[j, j]:
            return False
        pivot = np.sqrt(s)
        a[j, j] = pivot
        for i in range(j + 1, n):
            s = a[i, j]
            for k in range(j):
                s -= a[i, k] * a[j, k]
            a[i, j] = s / pivot
    return True


@numba.njit(cache=True, error_model="numpy")
def _solve_factored(factor: np.ndarray, b: np.ndarray, out: np.ndarray) -> None:
    """
    Solve ``a @ x = b`` into `out`, given the Cholesky factor of `a` from
    :func:`_cholesky`.
    """
    n = factor.shape[0]
    for i in range(n):
        s = b[i]
        for k in range(i):
            s -= factor[i, k] * out[k]
        out[i] = s / factor[i, i]
    for i in range(n - 1, -1, -1):
        s = out[i]
        for k in range(i + 1, n):
            s -= factor[k, i] * out[k]
        out[i] = s / factor[i, i]


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
    work[:, :] = a
    if not _cholesky(work, 0.0):
        return False
    _solve_factored(work, b, out)
    return True


@numba.njit(cache=True, error_model="numpy")
def _whitened_residual(
    factor: np.ndarray,
    m: int,
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
    _solve_lower_band(factor, out, m)
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
    jacobian: np.ndarray,
    out: np.ndarray,
) -> None:
    """Compute the whitened Jacobian, ``out = factor^-1 @ jacobian``."""
    for j in range(m):
        for k in range(jacobian.shape[1]):
            out[j, k] = jacobian[j, k]
    _solve_lower_band_columns(factor, out, m)


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
        g[k] = s
        for q in range(k + 1):
            s = 0.0
            for j in range(m):
                s += jacobian_whitened[j, k] * jacobian_whitened[j, q]
            a[k, q] = s
            a[q, k] = s
        if prior_precision[k] > 0:
            g[k] -= prior_precision[k] * (parameters[k] - prior_mean[k])
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


def _input(ndim: int) -> types.Array:
    """
    The Numba type of an array input of :func:`_levenberg_marquardt`.

    It is read-only and of any layout, so that an argument broadcast over the
    fits can be passed as a view rather than copied.
    """
    return types.Array(types.float64, ndim, "A", readonly=True)


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
    _input(3),
    _input(2),
    _input(2),
    _input(1),
    _input(2),
    _input(2),
    _input(2),
    _input(2),
    _input(2),
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
    model: ModelFunction,
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
        The compiled model function, :attr:`ndoptimize.Model.function`.
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
        The maximum number of steps per fit, not negative.
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
        factor = np.empty((num_point, num_lag))
        f = np.empty(num_point)
        jac = np.zeros((num_point, num_parameter))
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
                for d in range(min(r, num_lag - 1) + 1):
                    lag = index[r] - index[r - d]
                    rho = correlation[lag] if lag < num_lag else 0.0
                    factor[r, d] = sigma[r] * sigma[r - d] * rho
            if not _cholesky_band(factor, m):
                continue
            _solve_lower_band(factor, data_whitened, m)

            for k in range(num_parameter):
                precision[k] = 1 / prior_sigma[i, k] ** 2
                mean[k] = prior_mean[i, k]
                p[k] = min(max(guess[i, k], lower[i, k]), upper[i, k])

            model(xg[:m], p, f[:m], jac[:m])
            chi2_data = _whitened_residual(factor, m, data_whitened, f, residual)
            _whitened_jacobian(factor, m, jac, jacobian)
            objective = chi2_data + _chi2_prior(p, mean, precision)

            damping = _damping_initial
            nu = 2.0
            done = False
            n_step = 0
            while True:
                _normal(jacobian, residual, m, p, mean, precision, a, g)

                # hold the parameters which have no effect on the model, and
                # those which are at a bound and are pushed outward by the
                # gradient
                for k in range(num_parameter):
                    at_lower = p[k] <= lower[i, k] and g[k] < 0
                    at_upper = p[k] >= upper[i, k] and g[k] > 0
                    free[k] = a[k, k] != 0 and not (at_lower or at_upper)

                # the Newton decrement of the free parameters, the decrease
                # in chi-squared a Gauss-Newton step would give if the model
                # were linear
                for k in range(num_parameter):
                    rhs[k] = g[k] if free[k] else 0
                    for q in range(num_parameter):
                        system[k, q] = a[k, q] if (free[k] and free[q]) else 0
                    system[k, k] = a[k, k] * (1 + _ridge) if free[k] else 1
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
                            system[k, k] = a[k, k] * (1 + damping)
                    if _solve_positive_definite(system, rhs, work, step):
                        for k in range(num_parameter):
                            p_new[k] = p[k] + step[k]
                            p_new[k] = min(max(p_new[k], lower[i, k]), upper[i, k])
                        model(xg[:m], p_new, f[:m], jac[:m])
                        chi2_new = _whitened_residual(
                            factor, m, data_whitened, f, residual_new
                        )
                        objective_new = chi2_new + _chi2_prior(p_new, mean, precision)
                        accepted = objective_new < objective
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
                if not accepted:
                    break
                n_step += 1

                p, p_new = p_new, p
                residual, residual_new = residual_new, residual
                chi2_data = chi2_new
                objective = objective_new
                _whitened_jacobian(factor, m, jac, jacobian)

            # the covariance of the parameters is the inverse of the normal
            # matrix, with infinite variance for a parameter which has no
            # effect on the model, and NaN if the rest of it is singular
            _normal(jacobian, residual, m, p, mean, precision, a, g)
            for k in range(num_parameter):
                for q in range(num_parameter):
                    constrained = a[k, k] != 0 and a[q, q] != 0
                    work[k, q] = a[k, q] if constrained else 0
                if a[k, k] == 0:
                    work[k, k] = 1
            if _cholesky(work, _degenerate):
                for k in range(num_parameter):
                    for q in range(num_parameter):
                        unit[q] = 1 if q == k else 0
                    _solve_factored(work, unit, step)
                    for q in range(num_parameter):
                        covariance[i, q, k] = step[q]
            for k in range(num_parameter):
                if a[k, k] == 0:
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


@dataclasses.dataclass(frozen=True, eq=False)
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
    A parameter which has no effect on the model and no prior has an
    infinite variance.
    If the rest of the normal matrix is singular, because the data and the
    priors cannot constrain some combination of the parameters, as in a fit
    with fewer usable points than parameters, the covariance is NaN.
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
    model: Model,
    x: np.ndarray,
    data: np.ndarray,
    variance: np.ndarray,
    guess: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    prior_mean: np.ndarray,
    prior_sigma: np.ndarray,
) -> tuple[tuple[int, ...], dict[str, np.ndarray]]:
    """
    Check the arguments of :func:`least_squares`, arrays of floats, broadcast
    them against each other over their batch axes and flatten those axes,
    into views of the arguments where possible.

    Returns the batch shape and the flattened arrays.
    """
    if x.ndim < 2:
        raise ValueError(
            f"`x` must have at least two axes, (..., num_points, num_inputs), "
            f"got shape {x.shape}."
        )
    num_point, num_input = x.shape[-2:]
    num_parameter = len(model.parameters)

    if data.shape[-1:] != (num_point,):
        raise ValueError(
            f"The last axis of `data`, {data.shape}, must match the "
            f"number of points in `x`, {num_point}."
        )
    if guess.shape[-1:] != (num_parameter,):
        raise ValueError(
            f"The last axis of `guess`, {guess.shape}, must match the "
            f"number of parameters of the model, {model.parameters}."
        )
    if not np.all(lower <= upper):
        raise ValueError("`lower` must not exceed `upper`, and neither may be NaN.")
    if not np.all(prior_sigma > 0):
        raise ValueError("`prior_sigma` must be positive.")
    if np.any(np.isfinite(prior_sigma) & ~np.isfinite(prior_mean)):
        raise ValueError("`prior_mean` must be finite where `prior_sigma` is finite.")

    variance = np.broadcast_to(
        variance, np.broadcast_shapes(variance.shape, (num_point,))
    )
    lower, upper, prior_mean, prior_sigma = (
        np.broadcast_to(a, np.broadcast_shapes(a.shape, (num_parameter,)))
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
        """
        `a` broadcast to the batch shape with its batch axes flattened, a
        view of `a` unless the flattening needs a copy.
        """
        a = np.broadcast_to(a, shape + shape_trailing)
        return a.reshape(num_fit, *shape_trailing)

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
    x: npt.ArrayLike,
    data: npt.ArrayLike,
    variance: npt.ArrayLike,
    guess: npt.ArrayLike,
    lower: npt.ArrayLike = -np.inf,
    upper: npt.ArrayLike = np.inf,
    prior_mean: npt.ArrayLike = 0,
    prior_sigma: npt.ArrayLike = np.inf,
    correlation: None | npt.ArrayLike = None,
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
        The model to fit, made with :func:`ndoptimize.model`.
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
        ``(..., num_parameters)``, where ``num_parameters`` is the number of
        names in :attr:`ndoptimize.Model.parameters`.
        It is clipped to the bounds.
    lower
        The lower bound of each parameter, an array of shape
        ``(..., num_parameters)``.
    upper
        The upper bound of each parameter, an array of shape
        ``(..., num_parameters)``, no less than `lower`.
    prior_mean
        The mean of the Gaussian prior of each parameter, an array of shape
        ``(..., num_parameters)``.
        It is ignored where `prior_sigma` is infinite.
    prior_sigma
        The standard deviation of the Gaussian prior of each parameter, a
        positive array of shape ``(..., num_parameters)``.
        An infinite standard deviation, the default, means no prior.
    correlation
        The correlation of the noise of two data points by their separation
        along the last axis of `data`, in samples, starting with 1 for no
        separation and zero beyond its length.
        The separation counts the points left out of a fit.
        It must describe a positive-definite noise covariance.
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
    convergence test, as is a parameter which has no effect on the model.

    A fit whose :math:`\chi^2` is NaN at the initial guess, where the model
    is NaN for instance, takes no steps and does not converge.

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
    if max_iterations < 0:
        raise ValueError(
            f"`max_iterations` must not be negative, got {max_iterations}."
        )

    shape, a = _flatten(
        model=model,
        x=np.asarray(x, dtype=float),
        data=np.asarray(data, dtype=float),
        variance=np.asarray(variance, dtype=float),
        guess=np.asarray(guess, dtype=float),
        lower=np.asarray(lower, dtype=float),
        upper=np.asarray(upper, dtype=float),
        prior_mean=np.asarray(prior_mean, dtype=float),
        prior_sigma=np.asarray(prior_sigma, dtype=float),
    )

    if correlation is None:
        correlation = 1
    rho = np.array(correlation, dtype=float).reshape(-1)
    if rho.size == 0 or rho[0] != 1:
        raise ValueError(
            f"`correlation` must start with 1, the correlation of a point "
            f"with itself, got {rho}."
        )

    # the noise correlation of every fit is a principal submatrix of this
    # one, so it is positive definite if this one is
    num_point = a["x"].shape[1]
    num_lag = min(rho.size, num_point)
    band = np.zeros((num_point, num_lag))
    for d in range(num_lag):
        band[d:, d] = rho[d]
    if not np.all(np.isfinite(rho)) or not _cholesky_band(band, num_point):
        raise ValueError(
            f"`correlation`, {rho}, does not describe a positive "
            f"definite noise covariance for {num_point} points."
        )

    result = _levenberg_marquardt(
        model.function,
        a["x"],
        a["data"],
        a["variance"],
        rho,
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
