from typing import Any
import numpy as np
import pytest
import scipy.optimize
import ndoptimize

_velocity = np.linspace(-112, 112, 39)
"""The independent variable of the test spectra."""

_correlation = np.array([1, 0.187, -0.048, 0.007, 0.002, 0.005])
"""The correlation by lag of the noise of resampled spectra, from IRIS."""


def _noise_covariance(sigma: np.ndarray, correlation: None | np.ndarray) -> np.ndarray:
    """The banded covariance of the noise of one spectrum."""
    if correlation is None:
        correlation = np.ones(1)
    num = sigma.size
    lag = np.abs(np.arange(num)[:, np.newaxis] - np.arange(num))
    rho = np.zeros(num)
    rho[: correlation.size] = correlation[:num]
    return sigma[:, np.newaxis] * rho[lag] * sigma


def _evaluate(model: ndoptimize._model.Model, parameters: np.ndarray) -> np.ndarray:
    """The model at the test velocities for each row of `parameters`."""
    x = _velocity[:, np.newaxis]
    result = np.empty((len(parameters), _velocity.size))
    jacobian = np.empty((_velocity.size, parameters.shape[-1]))
    for i in range(len(parameters)):
        model(x, parameters[i], result[i], jacobian)
    return result


def _spectra(
    num: int,
    correlation: None | np.ndarray,
    seed: int = 0,
    missing: float = 0.05,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Bright Gaussian-plus-line spectra with noise of the given correlation and
    some missing points: the truth, the data, the variance and a guess.

    The lines are bright so that every fit has a minimum without priors or
    bounds; a faint line in correlated noise can fit better as the wing of
    an ever broader Gaussian, which has no minimum at all.
    """
    rng = np.random.default_rng(seed)
    truth = np.stack(
        [
            rng.uniform(4, 8, num),
            rng.normal(0, 5, num),
            rng.uniform(12, 25, num),
            rng.normal(0, 0.001, num),
            rng.uniform(0.5, 2, num),
        ],
        axis=-1,
    )
    f = _evaluate(ndoptimize.models.gaussian_linear, truth)
    variance = 0.05 + np.clip(f, 0, None) / 8
    factor = np.linalg.cholesky(_noise_covariance(np.ones(_velocity.size), correlation))
    noise = rng.standard_normal(f.shape) @ factor.T * np.sqrt(variance)
    data = f + noise
    data[rng.random(data.shape) < missing] = np.nan
    guess = truth * np.array([1.2, 1, 1.1, 0, 1]) + np.array([0, 3, 0, 0, 0.1])
    return truth, data, variance, guess


def _fit_scipy(
    data: np.ndarray,
    variance: np.ndarray,
    guess: np.ndarray,
    correlation: None | np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    prior_mean: np.ndarray,
    prior_sigma: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Fit one spectrum with :func:`scipy.optimize.least_squares`, whitening the
    residual with the dense Cholesky factor of the noise covariance of the
    usable points: the parameters, their covariance and the objective.
    """
    good = np.isfinite(data)
    covariance = _noise_covariance(np.sqrt(variance), correlation)
    factor = np.linalg.cholesky(covariance[np.ix_(good, good)])
    has_prior = np.isfinite(prior_sigma)
    x = _velocity[good, np.newaxis]
    f = np.empty(good.sum())
    jacobian = np.empty((good.sum(), guess.size))

    def residual(p: np.ndarray) -> np.ndarray:
        """The whitened residual of the data and of the priors."""
        ndoptimize.models.gaussian_linear(x, p, f, jacobian)
        r = np.linalg.solve(factor, data[good] - f)
        return np.concatenate([r, ((p - prior_mean) / prior_sigma)[has_prior]])

    bounded = np.isfinite(lower).any() or np.isfinite(upper).any()
    result = scipy.optimize.least_squares(
        fun=residual,
        x0=np.clip(guess, lower, upper),
        method="trf" if bounded else "lm",
        bounds=(lower, upper),
        ftol=1e-15,
        xtol=1e-15,
        gtol=1e-15,
        x_scale="jac",
        max_nfev=10000,
    )
    jacobian = np.asarray(result.jac)
    covariance = np.linalg.inv(jacobian.T @ jacobian)
    return result.x, covariance, float(np.sum(result.fun**2))


@pytest.mark.parametrize(
    argnames="correlation",
    argvalues=[None, _correlation],
)
@pytest.mark.parametrize(
    argnames="prior",
    argvalues=[False, True],
)
@pytest.mark.parametrize(
    argnames="bounded",
    argvalues=[False, True],
)
def test_least_squares(
    correlation: None | np.ndarray,
    prior: bool,
    bounded: bool,
) -> None:
    truth, data, variance, guess = _spectra(20, correlation)

    infinity = np.full(5, np.inf)
    prior_mean = np.array([0, 0, 18, 0, 0])
    prior_sigma = np.array([np.inf, 10, 6, np.inf, np.inf]) if prior else infinity
    lower = np.array([-np.inf, -90, 3, -np.inf, -np.inf]) if bounded else -infinity
    upper = np.array([np.inf, 90, 16, np.inf, np.inf]) if bounded else infinity

    result = ndoptimize.least_squares(
        model=ndoptimize.models.gaussian_linear,
        x=_velocity[:, np.newaxis],
        data=data,
        variance=variance,
        guess=guess,
        lower=lower,
        upper=upper,
        prior_mean=prior_mean,
        prior_sigma=prior_sigma,
        correlation=correlation,
    )

    assert np.all(result.converged)
    assert np.all(result.num_points == np.isfinite(data).sum(axis=-1))
    if bounded:
        assert np.any(result.parameters[..., 2] == upper[2])

    for i in range(len(data)):
        parameters, covariance, objective = _fit_scipy(
            data=data[i],
            variance=variance[i],
            guess=guess[i],
            correlation=correlation,
            lower=lower,
            upper=upper,
            prior_mean=prior_mean,
            prior_sigma=prior_sigma,
        )
        chi2_prior = np.where(
            np.isfinite(prior_sigma),
            ((result.parameters[i] - prior_mean) / prior_sigma) ** 2,
            0,
        )
        assert result.chi2[i] + chi2_prior.sum() <= objective + 1e-6

        sigma = np.sqrt(np.diag(result.covariance[i]))
        assert np.all(np.abs(result.parameters[i] - parameters) < 1e-3 * sigma)
        difference = (result.covariance[i] - covariance) / np.outer(sigma, sigma)
        assert np.all(np.abs(difference) < 1e-3)


def test_least_squares_calibration() -> None:
    """
    The covariance of the fits is the scatter of the parameters over many
    realizations of the noise, if the noise correlation is given, and
    underestimates it if it is not.
    """
    num = 500
    rng = np.random.default_rng(1)
    truth = np.array([4, 2, 16, 0, 1.0])
    f = _evaluate(ndoptimize.models.gaussian_linear, truth[np.newaxis])[0]
    variance = 0.05 + f / 8
    covariance = _noise_covariance(np.sqrt(variance), _correlation)
    data = f + rng.standard_normal((num, f.size)) @ np.linalg.cholesky(covariance).T

    def ratio(correlation: None | np.ndarray) -> np.ndarray:
        """The scatter of the fits over their covariance, for each parameter."""
        result = ndoptimize.least_squares(
            model=ndoptimize.models.gaussian_linear,
            x=_velocity[:, np.newaxis],
            data=data,
            variance=variance,
            guess=truth,
            correlation=correlation,
        )
        assert np.all(result.converged)
        scatter = np.var(result.parameters, axis=0)
        return scatter / np.mean(
            np.diagonal(result.covariance, axis1=-2, axis2=-1), axis=0
        )

    assert np.all(np.abs(ratio(_correlation) - 1) < 0.2)
    assert np.all(ratio(None)[:3] > 1.15)


def test_least_squares_broadcasting() -> None:
    truth, data, variance, guess = _spectra(6, None)
    kwargs: dict[str, Any] = dict(
        model=ndoptimize.models.gaussian_linear,
        x=_velocity[:, np.newaxis],
        variance=variance.mean(),
        guess=guess.mean(axis=0),
    )

    result = ndoptimize.least_squares(data=data.reshape(2, 3, -1), **kwargs)
    result_flat = ndoptimize.least_squares(data=data, **kwargs)

    assert result.parameters.shape == (2, 3, 5)
    assert result.covariance.shape == (2, 3, 5, 5)
    for name in ("chi2", "num_points", "iterations", "converged"):
        assert getattr(result, name).shape == (2, 3)
    assert np.array_equal(result.parameters.reshape(6, 5), result_flat.parameters)


def test_least_squares_missing() -> None:
    truth, data, variance, guess = _spectra(3, None, missing=0)
    data[0] = np.nan
    data[1, 10:15] = np.nan
    kwargs: dict[str, Any] = dict(
        model=ndoptimize.models.gaussian_linear,
        variance=variance[1],
        guess=guess[1],
    )

    result = ndoptimize.least_squares(x=_velocity[:, np.newaxis], data=data, **kwargs)

    assert np.all(np.isnan(result.parameters[0]))
    assert result.num_points[0] == 0
    assert not result.converged[0]
    assert result.num_points[1] == _velocity.size - 5

    good = np.isfinite(data[1])
    result_good = ndoptimize.least_squares(
        model=kwargs["model"],
        x=_velocity[good, np.newaxis],
        data=data[1, good],
        variance=variance[1, good],
        guess=guess[1],
    )
    assert np.allclose(result.parameters[1], result_good.parameters, rtol=1e-10)


def test_least_squares_unconstrained() -> None:
    """A parameter which nothing constrains has an infinite variance."""
    result = ndoptimize.least_squares(
        model=ndoptimize.models.gaussian,
        x=_velocity[:, np.newaxis],
        data=np.zeros(_velocity.size),
        variance=1,
        guess=np.array([0, 0, 15.0]),
        lower=np.array([0, -np.inf, 0]),
        upper=np.array([0, np.inf, np.inf]),
    )
    assert result.converged
    assert np.all(np.isinf(np.diagonal(result.covariance)[1:]))
    assert np.isfinite(result.covariance[0, 0])


@ndoptimize.model
def _line(
    x: np.ndarray,
    parameters: np.ndarray,
    f: np.ndarray,
    jacobian: np.ndarray,
) -> None:
    """A straight line."""
    for i in range(x.shape[0]):
        f[i] = parameters[0] + parameters[1] * x[i, 0]
        jacobian[i, 0] = 1
        jacobian[i, 1] = x[i, 0]


def test_least_squares_linear() -> None:
    """A user-defined linear model gives the weighted linear least squares
    solution in a few steps, to well within the tolerance."""
    rng = np.random.default_rng(2)
    x = np.linspace(0, 1, 11)
    variance = rng.uniform(0.5, 2, x.size)
    data = 1 + 2 * x + rng.normal(0, np.sqrt(variance))

    result = ndoptimize.least_squares(
        model=_line,
        x=x[:, np.newaxis],
        data=data,
        variance=variance,
        guess=np.zeros(2),
    )

    design = np.stack([np.ones_like(x), x], axis=-1) / np.sqrt(variance)[:, np.newaxis]
    expected = np.linalg.lstsq(design, data / np.sqrt(variance), rcond=None)[0]
    covariance = np.linalg.inv(design.T @ design)
    sigma = np.sqrt(np.diag(covariance))
    assert np.all(np.abs(result.parameters - expected) < 1e-4 * sigma)
    assert np.allclose(result.covariance, covariance, rtol=1e-10)
    assert result.converged
    assert result.iterations <= 3


def test_least_squares_far_guess() -> None:
    """Fits started far from the solution, which take damped and rejected
    steps, reach the same minimum as fits started near it."""
    truth, data, variance, guess = _spectra(10, _correlation)
    kwargs: dict[str, Any] = dict(
        model=ndoptimize.models.gaussian_linear,
        x=_velocity[:, np.newaxis],
        data=data,
        variance=variance,
        lower=np.array([-np.inf, -np.inf, 3, -np.inf, -np.inf]),
        correlation=_correlation,
    )

    result = ndoptimize.least_squares(guess=guess, **kwargs)
    result_far = ndoptimize.least_squares(guess=np.array([2, 10, 30, 0, 1]), **kwargs)

    assert np.all(result_far.converged)
    assert result_far.iterations.sum() > result.iterations.sum()
    sigma = np.sqrt(np.diagonal(result.covariance, axis1=-2, axis2=-1))
    assert np.all(np.abs(result_far.parameters - result.parameters) < 1e-3 * sigma)


def test_least_squares_max_iterations() -> None:
    truth, data, variance, guess = _spectra(5, None)

    result = ndoptimize.least_squares(
        model=ndoptimize.models.gaussian_linear,
        x=_velocity[:, np.newaxis],
        data=data,
        variance=variance,
        guess=guess,
        max_iterations=1,
    )

    assert not np.any(result.converged)
    assert np.all(result.iterations == 1)


@pytest.mark.parametrize(
    argnames="kwargs",
    argvalues=[
        dict(correlation=np.array([0.5, 0.1])),
        dict(correlation=np.array([1, 0.9])),
        dict(data=np.zeros(5)),
        dict(x=_velocity),
    ],
)
def test_least_squares_invalid(kwargs: dict[str, Any]) -> None:
    arguments: dict[str, Any] = dict(
        model=ndoptimize.models.gaussian,
        x=_velocity[:, np.newaxis],
        data=np.zeros(_velocity.size),
        variance=1,
        guess=np.array([1, 0, 15.0]),
    )
    with pytest.raises(ValueError):
        ndoptimize.least_squares(**(arguments | kwargs))
