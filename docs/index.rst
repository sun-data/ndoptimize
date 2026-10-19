Introduction
============

:mod:`ndoptimize` minimizes objective functions for many independent
problems at once, similar to the :mod:`scipy.optimize` package, but compiled
and parallelized using the `Numba <https://numba.readthedocs.io/en/stable/>`_
just-in-time compiler.

Its main function, :func:`ndoptimize.least_squares`, fits a model to every
spectrum of a spectral cube, or any other batch of small datasets, using the
Levenberg-Marquardt method.
Compared to calling :func:`scipy.optimize.least_squares` in a loop, it
offers:

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

The package provides:

- :func:`ndoptimize.least_squares`, which fits a model to many sets of data
  at once.
- :func:`ndoptimize.evaluate`, which evaluates a model for many sets of
  parameters at once, such as the best-fit ones.
- :func:`ndoptimize.model`, a decorator which compiles a function as a model,
  an :class:`ndoptimize.Model`, and names its parameters.
- :mod:`ndoptimize.models`, some common models: a Gaussian, a Gaussian
  averaged over each bin, a Gaussian plus a straight line, and two Gaussians
  plus a straight line.


Differences from :func:`scipy.optimize.least_squares`
=====================================================

For the same objective, the two find the same minimum, to a thousandth of a
standard deviation in the tests, but they differ in the following ways.

- **Derivatives.** A model computes its own Jacobian. There is no
  finite-difference approximation.
- **Noise and priors.** The variance of the data, the correlation of its
  noise, and the Gaussian priors are arguments, and the result includes the
  covariance of the parameters. :func:`scipy.optimize.least_squares`
  minimizes the sum of squares of whatever residual it is given, and returns
  the Jacobian at the solution, from which the covariance can be computed.
- **Bounds.** Steps are clipped to the bounds, and a parameter which the
  gradient pushes against its bound is held fixed for that step. SciPy's
  ``"lm"`` method does not support bounds, and its ``"trf"`` and
  ``"dogbox"`` methods handle them with trust regions.
- **Convergence.** A single tolerance on the decrease in chi-squared that a
  Gauss-Newton step predicts replaces ``ftol``, ``xtol`` and ``gtol``.
- **Model arguments.** A model takes only the points ``x`` and the
  parameters, so any other data it needs, such as the edges of each bin, goes
  in more columns of ``x``. SciPy passes extra data through ``args`` and
  ``kwargs``.
- **Results.** :attr:`~ndoptimize.LeastSquaresResult.chi2` is the sum of the
  squares of the whitened residuals, without the priors, while SciPy's
  ``cost`` is half the sum of the squares of whatever residual it is given.
  ``max_iterations`` and :attr:`~ndoptimize.LeastSquaresResult.iterations`
  count accepted steps, while SciPy's ``max_nfev`` and ``nfev`` count
  evaluations of the residual.
- **Not supported:** robust loss functions and sparse Jacobians.


Installation
============
:mod:`ndoptimize` is published on PyPI and can be installed using::

    pip install ndoptimize


Quickstart
==========

Fit a Gaussian to each of a thousand noisy spectral lines at once, and
evaluate the best-fit models on a finer grid.

..
    The README embeds the figure of the next cell from Read the Docs as
    _images/index_0_0.png, a name which jupyter-sphinx derives from the order
    of the cells, so keep it the first jupyter-execute cell of this page.

.. jupyter-execute::

    import numpy as np
    import matplotlib.pyplot as plt
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

    # the best-fit model of every line on a finer grid
    velocity_fine = np.linspace(-100, 100, 201)
    fit = ndoptimize.evaluate(
        model=ndoptimize.models.gaussian,
        x=velocity_fine[:, np.newaxis],
        parameters=result.parameters,
    )

    # plot the first three lines and their fits
    fig, ax = plt.subplots(constrained_layout=True)
    for i in range(3):
        ax.scatter(velocity, data[i], color=f"C{i}", s=10);
        ax.plot(velocity_fine, fit[i], color=f"C{i}");
    ax.set_xlabel("velocity (km/s)");
    ax.set_ylabel("intensity");

The result holds the best-fit parameters of every line, their covariance,
chi-squared, and the convergence state of each fit, as described in
:class:`ndoptimize.LeastSquaresResult`.

.. jupyter-execute::

    result.parameters.shape, result.covariance.shape


Writing a model
===============

A model is a function, ``model(x, parameters, f, jacobian)``, which writes its
value at the points ``x``, an array of shape ``(num_points, num_inputs)``,
into ``f``, and its derivatives with respect to the ``parameters`` into
``jacobian``.
The :func:`ndoptimize.model` decorator compiles it and names its parameters.
The compiled model is cached on disk, so define it in a file or a notebook.
A model defined in the interactive interpreter needs
``@ndoptimize.model(..., cache=False)``.

Here is an exponential decay, fit to a hundred noisy decays, with bounds which
keep the amplitude from going negative and the scale at least 0.1.

.. jupyter-execute::

    import math

    @ndoptimize.model(parameters=("amplitude", "scale"))
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

    time = np.linspace(0, 10, 50)
    scale = rng.uniform(1, 5, size=(100, 1))
    data = 3 * np.exp(-time / scale) + rng.normal(0, 0.1, size=(100, time.size))

    result = ndoptimize.least_squares(
        model=exponential,
        x=time[:, np.newaxis],
        data=data,
        variance=0.1**2,
        guess=[1, 1],
        lower=[0, 0.1],
    )

    # compare the fitted scales to the truth
    fig, ax = plt.subplots(constrained_layout=True)
    ax.errorbar(
        x=scale[:, 0],
        y=result.parameters[:, 1],
        yerr=np.sqrt(result.covariance[:, 1, 1]),
        fmt="o",
        markersize=3,
    );
    ax.plot([1, 5], [1, 5], color="gray");
    ax.set_xlabel("true scale");
    ax.set_ylabel("fitted scale");


Citation
========

If you use :mod:`ndoptimize` in your research, please cite it.
The citation metadata is kept in
`CITATION.cff <https://github.com/sun-data/ndoptimize/blob/main/CITATION.cff>`_,
which the "Cite this repository" button on the
`GitHub page <https://github.com/sun-data/ndoptimize>`_
can export as BibTeX or APA.

Every release of :mod:`ndoptimize` is archived on Zenodo with its own DOI.
The concept DOI,
`10.5281/zenodo.23484609 <https://doi.org/10.5281/zenodo.23484609>`_,
always resolves to the latest version,
and the Zenodo page lists the DOI of every version.
Please include the version of :mod:`ndoptimize` that you used,
which is given by ``importlib.metadata.version("ndoptimize")``.
The BibTeX entry below uses the concept DOI.
To cite a specific version instead,
replace ``doi`` with the DOI of that version.

.. code-block:: bibtex

    @software{ndoptimize,
      author = {Smart, Roy T.},
      title = {ndoptimize},
      version = {X.Y.Z},
      doi = {10.5281/zenodo.23484609},
      url = {https://github.com/sun-data/ndoptimize},
    }


API Reference
=============

An in-depth description of the interfaces defined by this package.

.. autosummary::
    :toctree: _autosummary
    :template: module_custom.rst
    :recursive:

    ndoptimize


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
