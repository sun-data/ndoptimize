Introduction
============

:mod:`ndoptimize` minimizes objective functions for many independent
problems at once, similar to the :mod:`scipy.optimize` package, but compiled
and parallelized using the `Numba <https://numba.readthedocs.io/en/stable/>`_
just-in-time compiler.

Its main function, :func:`ndoptimize.least_squares`, fits a model to every
spectrum of a spectral cube, or any other batch of small datasets, using the
Levenberg-Marquardt method.
Each fit has its own independent variable, data, noise, initial guess,
bounds and Gaussian priors, and the noise of neighboring data points may be
correlated, as it is in resampled data, which makes the fit generalized least
squares.
The fits run in parallel on every CPU core, roughly half a million small
fits per second on a 48-thread machine.

Models are short Numba functions which compute the model and its Jacobian,
compiled with the :func:`ndoptimize.model` decorator, which also names their
parameters.
Some common ones are provided in :mod:`ndoptimize.models`.


Installation
============
:mod:`ndoptimize` is published on PyPI and can be installed using::

    pip install ndoptimize


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
