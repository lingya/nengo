import shutil
import tempfile

import numpy as np
from numpy.testing import assert_equal
import pytest

from nengo.cache import DecoderCache


@pytest.fixture(scope='function')
def cache_dir(request):
    d = tempfile.mkdtemp()
    def fin():
        shutil.rmtree(d)
    request.addfinalizer(fin)
    return d


class DecoderSolverMock(object):
    def __init__(self, name='solver_mock'):
        self.n_calls = 0
        self.__module__ = __name__
        self.__name__ = name

    def __call__(self, A, Y, rng=np.random, E=None):
        self.n_calls += 1
        if E is None:
            return np.random.rand(A.shape[1], Y.shape[1]), {}
        else:
            return np.random.rand(A.shape[1], E.shape[2]), {}


def test_decoder_cache(cache_dir):
    solver_mock = DecoderSolverMock()

    M = 100
    N = 10
    D = 2
    activities = np.ones((M, D))
    targets = np.ones((M, N))
    rng = np.random.RandomState(42)

    cache = DecoderCache(cache_dir)
    decoders1, solver_info1 = cache(solver_mock)(activities, targets, rng)
    assert solver_mock.n_calls == 1
    decoders2, solver_info2 = cache(solver_mock)(activities, targets, rng)
    assert solver_mock.n_calls == 1  # check the result is read from cache
    assert_equal(decoders1, decoders2)
    assert solver_info1 == solver_info2

    another_solver = DecoderSolverMock('another_solver')
    cache(another_solver)(activities, targets, rng)
    assert another_solver.n_calls == 1

    # TODO test using E
