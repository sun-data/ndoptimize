# ndoptimize

[![tests](https://github.com/sun-data/ndoptimize/actions/workflows/tests.yml/badge.svg)](https://github.com/sun-data/ndoptimize/actions/workflows/tests.yml)
[![codecov](https://codecov.io/gh/sun-data/ndoptimize/graph/badge.svg)](https://codecov.io/gh/sun-data/ndoptimize)
[![typing](https://github.com/sun-data/ndoptimize/actions/workflows/typing.yml/badge.svg)](https://github.com/sun-data/ndoptimize/actions/workflows/typing.yml)
[![Black](https://github.com/sun-data/ndoptimize/actions/workflows/black.yml/badge.svg)](https://github.com/sun-data/ndoptimize/actions/workflows/black.yml)
[![Ruff](https://github.com/sun-data/ndoptimize/actions/workflows/ruff.yml/badge.svg)](https://github.com/sun-data/ndoptimize/actions/workflows/ruff.yml)
[![Documentation Status](https://readthedocs.org/projects/ndoptimize/badge/?version=latest)](https://ndoptimize.readthedocs.io/en/latest/?badge=latest)
[![PyPI version](https://badge.fury.io/py/ndoptimize.svg)](https://badge.fury.io/py/ndoptimize)

`ndoptimize` minimizes objective functions for many independent problems at
once, similar to
[`scipy.optimize`](https://docs.scipy.org/doc/scipy/reference/optimize.html),
but compiled and parallelized using
[Numba](https://numba.readthedocs.io/en/stable/).

Its main function, `ndoptimize.least_squares()`, fits a model to every
spectrum of a spectral cube, or any other batch of small datasets, using the
Levenberg-Marquardt method. Compared to calling
`scipy.optimize.least_squares()` in a loop, it offers:

- **Speed.** The fits run in parallel on every CPU core, roughly half a
  million fits of a five-parameter model to 39 points per second on a
  48-thread machine.
- **Correlated noise.** The noise of neighboring data points may be
  correlated, as it is in resampled data, which makes the fit generalized
  least squares, with parameter uncertainties to match.
- **Priors and bounds.** Each parameter of each fit may have a Gaussian prior
  and bounds, to keep faint fits stable.
- **Missing data.** Data points which are NaN are left out of their fit.
- **Broadcasting.** The data, noise, initial guess, bounds and priors are
  broadcast against each other, so each fit can have its own.

## Installation

```bash
pip install ndoptimize
```

## Example

```python
import numpy as np
import ndoptimize

velocity = np.linspace(-100, 100, 41)
data = ...      # spectra, shape (..., 41)
variance = ...  # their noise variance, the same shape

result = ndoptimize.least_squares(
    model=ndoptimize.models.gaussian_linear,
    x=velocity[:, np.newaxis],
    data=data,
    variance=variance,
    guess=[5, 0, 15, 0, 1],  # amplitude, center, width, slope, intercept
)

result.parameters  # shape (..., 5)
result.covariance  # shape (..., 5, 5)

# the best-fit model at each point of each spectrum, shape (..., 41)
fit = ndoptimize.evaluate(
    model=ndoptimize.models.gaussian_linear,
    x=velocity[:, np.newaxis],
    parameters=result.parameters,
)
```

Models are short Numba functions which compute the model and its Jacobian,
compiled with the `ndoptimize.model` decorator, which also names their
parameters.
See the [documentation](https://ndoptimize.readthedocs.io/en/latest) for
details.
