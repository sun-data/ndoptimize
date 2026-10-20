# ndoptimize

[![tests](https://github.com/sun-data/ndoptimize/actions/workflows/tests.yml/badge.svg)](https://github.com/sun-data/ndoptimize/actions/workflows/tests.yml)
[![codecov](https://codecov.io/gh/sun-data/ndoptimize/graph/badge.svg)](https://codecov.io/gh/sun-data/ndoptimize)
[![typing](https://github.com/sun-data/ndoptimize/actions/workflows/typing.yml/badge.svg)](https://github.com/sun-data/ndoptimize/actions/workflows/typing.yml)
[![Black](https://github.com/sun-data/ndoptimize/actions/workflows/black.yml/badge.svg)](https://github.com/sun-data/ndoptimize/actions/workflows/black.yml)
[![Ruff](https://github.com/sun-data/ndoptimize/actions/workflows/ruff.yml/badge.svg)](https://github.com/sun-data/ndoptimize/actions/workflows/ruff.yml)
[![Documentation Status](https://readthedocs.org/projects/ndoptimize/badge/?version=latest)](https://ndoptimize.readthedocs.io/en/latest/?badge=latest)
[![PyPI version](https://badge.fury.io/py/ndoptimize.svg)](https://badge.fury.io/py/ndoptimize)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23484609.svg)](https://doi.org/10.5281/zenodo.23484609)

`ndoptimize` minimizes objective functions for many independent problems at
once, similar to
[`scipy.optimize`](https://docs.scipy.org/doc/scipy/reference/optimize.html),
but compiled and parallelized using
[Numba](https://numba.readthedocs.io/en/stable/).

Its main function,
[`ndoptimize.least_squares()`](https://ndoptimize.readthedocs.io/en/latest/_autosummary/ndoptimize.least_squares.html),
fits a model to every spectrum of a spectral cube, or any other batch of small
datasets, using the Levenberg-Marquardt method.
Compared to calling `scipy.optimize.least_squares()` in a loop, it offers:

- **Speed.** The fits run in parallel on every CPU core, roughly half a
  million fits of a five-parameter model to 39 points per second on a
  48-thread machine.
- **Correlated noise.** The noise of neighboring data points may be
  correlated, as it is in resampled data, which makes the fit generalized
  least squares, with parameter uncertainties to match.
- **Priors and bounds.** Each parameter of each fit may have a Gaussian prior
  and bounds, to keep faint fits stable.
- **Missing data.** Data points which are NaN are left out of their fit.
- **Broadcasting.** The independent variable, data, noise variance, initial
  guess, bounds and priors are broadcast against each other, so each fit can
  have its own. The correlation of the noise is shared by every fit.
- **Custom models.** A model is a short Numba function which computes its
  values and their derivatives. Every model shares the same fitting kernel,
  which is compiled once and cached on disk.

## Differences from `scipy.optimize.least_squares`

For the same objective, the two find the same minimum, to a thousandth of a
standard deviation in the tests, but they differ in the following ways.

- **Derivatives.** A model computes its own Jacobian. There is no
  finite-difference approximation.
- **Noise and priors.** The variance of the data, the correlation of its
  noise, and the Gaussian priors are arguments, and the result includes the
  covariance of the parameters. `scipy.optimize.least_squares()` minimizes the
  sum of squares of whatever residual it is given, and returns the Jacobian at
  the solution, from which the covariance can be computed.
- **Bounds.** Steps are clipped to the bounds, and a parameter which the
  gradient pushes against its bound is held fixed for that step. SciPy's
  `"lm"` method does not support bounds, and its `"trf"` and `"dogbox"`
  methods handle them with trust regions.
- **Convergence.** A single tolerance on the decrease in chi-squared that a
  Gauss-Newton step predicts replaces `ftol`, `xtol` and `gtol`.
- **Model arguments.** A model takes only the points `x` and the parameters,
  so any other data it needs, such as the edges of each bin, goes in more
  columns of `x`, each one an input that the model names. SciPy passes extra
  data through `args` and `kwargs`.
- **Results.** `chi2` is the sum of the squares of the whitened residuals,
  without the priors, while SciPy's `cost` is half the sum of the squares of
  whatever residual it is given. `max_iterations` and `iterations` count
  accepted steps, while SciPy's `max_nfev` and `nfev` count evaluations of
  the residual.
- **Not supported:** robust loss functions and sparse Jacobians.

The full documentation is hosted on
[Read the Docs](https://ndoptimize.readthedocs.io/en/latest/).

## Installation

`ndoptimize` is published on PyPI and can be installed using `pip`.

```bash
pip install ndoptimize
```

## Quickstart

Fit a Gaussian to each of a thousand noisy spectral lines at once, and
evaluate the best-fit models on a finer grid.

```python
import numpy as np
import ndoptimize

rng = np.random.default_rng(0)

# a thousand noisy Gaussian lines with random centers
velocity = np.linspace(-100, 100, 41)
center = rng.normal(0, 10, size=(1000, 1))
noise = 0.5
data = 5 * np.exp(-np.square((velocity - center) / 15) / 2)
data = data + rng.normal(0, noise, size=data.shape)

# start each fit at the peak of its line
guess = np.stack([
    data.max(axis=-1),                # amplitude
    velocity[data.argmax(axis=-1)],   # center
    np.full(len(data), 10),           # width
], axis=-1)

# fit every line at once
result = ndoptimize.least_squares(
    model=ndoptimize.models.gaussian,
    x=velocity[:, np.newaxis],  # shape (num_points, num_inputs)
    data=data,
    variance=noise**2,
    guess=guess,
)

result.parameters  # shape (1000, 3)
result.covariance  # shape (1000, 3, 3)

# the best-fit model of every line on a finer grid, shape (1000, 201)
velocity_fine = np.linspace(-100, 100, 201)
fit = ndoptimize.evaluate(
    model=ndoptimize.models.gaussian,
    x=velocity_fine[:, np.newaxis],
    parameters=result.parameters,
)
```

The documentation plots three of the lines with their fits:

![quickstart](https://ndoptimize.readthedocs.io/en/latest/_images/index_0_0.png)

## Writing a model

A model is a function, `model(x, parameters, f, jacobian)`, which writes its
value at the points `x`, an array of shape `(num_points, num_inputs)`, into
`f`, and its derivatives with respect to the `parameters` into `jacobian`.
The
[`ndoptimize.model`](https://ndoptimize.readthedocs.io/en/latest/_autosummary/ndoptimize.model-decorator.html)
decorator compiles it and names its parameters and its inputs, the columns of
`x`.
`least_squares()` and `evaluate()` check that `x` has one column for each
input, since the compiled model does not check its indices.
The compiled model is cached on disk, so define it in a file or a notebook.
A model defined in the interactive interpreter needs
`@ndoptimize.model(..., cache=False)`.

```python
import math
import numpy as np
import ndoptimize

@ndoptimize.model(parameters=("amplitude", "scale"), inputs=("x",))
def exponential(
    x: np.ndarray,
    parameters: np.ndarray,
    f: np.ndarray,
    jacobian: np.ndarray,
) -> None:
    """An exponential decay, amplitude * exp(-x / scale)."""
    amplitude, scale = parameters[0], parameters[1]
    for i in range(x.shape[0]):
        e = math.exp(-x[i, 0] / scale)
        f[i] = amplitude * e
        jacobian[i, 0] = e
        jacobian[i, 1] = amplitude * e * x[i, 0] / scale**2
```

[`ndoptimize.models`](https://ndoptimize.readthedocs.io/en/latest/_autosummary/ndoptimize.models.html)
has some common models: a Gaussian, a Gaussian averaged over each bin, a
Gaussian plus a straight line, and two Gaussians plus a straight line.

## Citation

If you use ndoptimize in your research, please cite it.
The citation metadata is kept in [`CITATION.cff`](https://github.com/sun-data/ndoptimize/blob/main/CITATION.cff),
which the "Cite this repository" button on GitHub can export as BibTeX or APA.

Every release of ndoptimize is archived on Zenodo with its own DOI.
The concept DOI, [10.5281/zenodo.23484609](https://doi.org/10.5281/zenodo.23484609),
always resolves to the latest version,
and the Zenodo page lists the DOI of every version.
Please include the version of ndoptimize that you used,
which is given by `importlib.metadata.version("ndoptimize")`.
The BibTeX entry below uses the concept DOI.
To cite a specific version instead,
replace `doi` with the DOI of that version.

```bibtex
@software{ndoptimize,
  author = {Smart, Roy T.},
  title = {ndoptimize},
  version = {X.Y.Z},
  doi = {10.5281/zenodo.23484609},
  url = {https://github.com/sun-data/ndoptimize},
}
```
