from __future__ import absolute_import
from __future__ import print_function

import logging
import pytest

import numpy as np
from numpy.testing import assert_equal
import pytest

import nengo
from nengo.utils.numpy import filt, filtfilt, lti, meshgrid_nd
from nengo.utils.testing import Plotter

logger = logging.getLogger(__name__)


def test_filt():
    dt = 1e-3
    tend = 3.
    t = dt * np.arange(tend / dt)
    nt = len(t)

    tau = 0.1 / dt

    u = np.random.normal(size=nt)

    tk = np.arange(0, 30 * tau)
    k = 1. / tau * np.exp(-tk / tau)
    x = np.convolve(u, k, mode='full')[:nt]

    y = filt(u, tau)

    with Plotter(nengo.Simulator) as plt:
        plt.plot(t, x)
        plt.plot(t, y, '--')
        plt.savefig('utils.test_filtering.test_filt.pdf')
        plt.close()

    assert np.allclose(x, y, atol=1e-3, rtol=1e-2)


def test_filtfilt():
    dt = 1e-3
    tend = 3.
    t = dt * np.arange(tend / dt)
    nt = len(t)

    tau = 0.03 / dt

    u = np.random.normal(size=nt)
    x = filt(u, tau)
    x = filt(x[::-1], tau, x0=x[-1])[::-1]
    y = filtfilt(u, tau)

    with Plotter(nengo.Simulator) as plt:
        plt.plot(t, x)
        plt.plot(t, y, '--')
        plt.savefig('utils.test_filtering.test_filtfilt.pdf')
        plt.close()

    assert np.allclose(x, y)


def test_lti_lowpass():
    dt = 1e-3
    tend = 3.
    t = dt * np.arange(tend / dt)
    nt = len(t)

    tau = 1e-2

    d = -np.expm1(-dt / tau)
    a = [d]
    b = [1, d - 1]

    u = np.random.normal(size=(nt, 10))
    x = filt(u, tau / dt)
    y = lti(u, (a, b))
    assert np.allclose(x, y)


def test_meshgrid_nd():
    a = [0, 0, 1]
    b = [1, 2, 3]
    c = [23, 42]
    expected = [
        np.array([[[0, 0], [0, 0], [0, 0]],
                  [[0, 0], [0, 0], [0, 0]],
                  [[1, 1], [1, 1], [1, 1]]]),
        np.array([[[1, 1], [2, 2], [3, 3]],
                  [[1, 1], [2, 2], [3, 3]],
                  [[1, 1], [2, 2], [3, 3]]]),
        np.array([[[23, 42], [23, 42], [23, 42]],
                  [[23, 42], [23, 42], [23, 42]],
                  [[23, 42], [23, 42], [23, 42]]])]
    actual = meshgrid_nd(a, b, c)
    assert_equal(expected, actual)


if __name__ == "__main__":
    nengo.log(debug=True)
    pytest.main([__file__, '-v'])
