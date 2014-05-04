import os
import shutil
import tempfile

import numpy as np
from numpy.testing import assert_equal
import pytest

from nengo.cache import DecoderCache, Fingerprint


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
        self.name = name

    def get_solver(mock):
        class Solver(object):
            def __init__(self, name):
                self.name = name

            def __call__(self, A, Y, rng=np.random, E=None):
                mock.n_calls += 1
                if E is None:
                    return np.random.rand(A.shape[1], Y.shape[1]), {'info': 'v'}
                else:
                    return np.random.rand(A.shape[1], E.shape[1]), {'info': 'v'}
        return Solver(mock.name)


def test_decoder_cache(cache_dir):
    solver_mock = DecoderSolverMock()

    M = 100
    N = 10
    D = 2
    activities = np.ones((M, D))
    targets = np.ones((M, N))
    rng = np.random.RandomState(42)

    # Basic test, that results are cached.
    cache = DecoderCache(cache_dir=cache_dir)
    decoders1, solver_info1 = cache.wrap_solver(solver_mock.get_solver())(
        activities, targets, rng)
    assert solver_mock.n_calls == 1
    decoders2, solver_info2 = cache.wrap_solver(solver_mock.get_solver())(
        activities, targets, rng)
    assert solver_mock.n_calls == 1  # check the result is read from cache
    assert_equal(decoders1, decoders2)
    assert solver_info1 == solver_info2

    decoders3, solver_info3 = cache.wrap_solver(solver_mock.get_solver())(
        2 * activities, targets, rng)
    assert solver_mock.n_calls == 2
    assert np.any(decoders1 != decoders3)

    # Test that the cache does not load results of another solver.
    another_solver = DecoderSolverMock('another_solver')
    cache.wrap_solver(another_solver.get_solver())(activities, targets, rng)
    assert another_solver.n_calls == 1


def test_decoder_cache_invalidation(cache_dir):
    solver_mock = DecoderSolverMock()

    M = 100
    N = 10
    D = 2
    activities = np.ones((M, D))
    targets = np.ones((M, N))
    rng = np.random.RandomState(42)

    # Basic test, that results are cached.
    cache = DecoderCache(cache_dir=cache_dir)
    cache.wrap_solver(solver_mock.get_solver())(activities, targets, rng)
    assert solver_mock.n_calls == 1
    cache.invalidate()
    cache.wrap_solver(solver_mock.get_solver())(activities, targets, rng)
    assert solver_mock.n_calls == 2


def test_decoder_cache_shrinking(cache_dir):
    solver_mock = DecoderSolverMock()
    another_solver = DecoderSolverMock('another_solver')

    M = 100
    N = 10
    D = 2
    activities = np.ones((M, D))
    targets = np.ones((M, N))
    rng = np.random.RandomState(42)

    cache = DecoderCache(cache_dir=cache_dir)
    cache.wrap_solver(solver_mock.get_solver())(activities, targets, rng)

    # Ensure differing time stamps (depending on the file system the timestamp
    # resolution might be as bad as 1 day).
    for filename in os.listdir(cache.cache_dir):
        path = os.path.join(cache.cache_dir, filename)
        timestamp = os.stat(path).st_atime
        timestamp -= 60 * 60 * 24 * 2  # 2 days
        os.utime(path, (timestamp, timestamp))

    cache.wrap_solver(another_solver.get_solver())(activities, targets, rng)

    assert cache.get_size() > 0

    cache.shrink(1)

    # check that older cached result was removed
    assert solver_mock.n_calls == 1
    cache.wrap_solver(another_solver.get_solver())(activities, targets, rng)
    cache.wrap_solver(solver_mock.get_solver())(activities, targets, rng)
    assert solver_mock.n_calls == 2
    assert another_solver.n_calls == 1


def test_decoder_cache_with_E_argument_to_solver(cache_dir):
    solver_mock = DecoderSolverMock()

    M = 100
    N = 10
    N2 = 5
    D = 2
    activities = np.ones((M, D))
    targets = np.ones((M, N))
    rng = np.random.RandomState(42)
    E = np.ones((D, N2))

    cache = DecoderCache(cache_dir=cache_dir)
    decoders1, solver_info1 = cache.wrap_solver(solver_mock.get_solver())(
        activities, targets, rng, E=E)
    assert solver_mock.n_calls == 1
    decoders2, solver_info2 = cache.wrap_solver(solver_mock.get_solver())(
        activities, targets, rng, E=E)
    assert solver_mock.n_calls == 1  # check the result is read from cache
    assert_equal(decoders1, decoders2)
    assert solver_info1 == solver_info2


class DummyA(object):
    def __init__(self, attr=0):
        self.attr = attr

class DummyB(object):
    def __init__(self, attr=0):
        self.attr = attr

def dummy_fn(arg):
    pass

@pytest.mark.parametrize('reference, equal, different', (
    (True, True, False),             # bool
    (False, False, True),            # bool
    (1, 1, 2),                       # int
    (1L, 1L, 2L),                    # long
    (1.0, 1.0, 2.0),                 # float
    (1.0 + 2.0j, 1 + 2j, 2.0 + 1j),  # complex
    (b'a', b'a', b'b'),              # bytes
    ('a', 'a', 'b'),              # string
    (u'a', u'a', u'b'),              # unicode string
    (np.eye(2), np.eye(2), np.array([[0, 1], [1, 0]])),      # array
    ({'a': 1, 'b': 2}, {'b': 2, 'a': 1}, {'a': 2, 'b': 1}),  # dict
    ((1, 2), (1, 2), (2, 1)),        # tuple
    ([1, 2], [1, 2], [2, 1]),        # list
    (DummyA(), DummyA(), DummyB()),  # object instance
    (DummyA(1), DummyA(1), DummyA(2)),    # object instance
    (DummyA(1), DummyA(1), DummyA(2)),    # object instance
))
def test_fingerprinting(reference, equal, different):
    assert str(Fingerprint(reference)) == str(Fingerprint(equal))
    assert str(Fingerprint(reference)) != str(Fingerprint(different))


def test_fails_for_functions():
    # Functions are difficult to handle because they could be lambda
    # expressions which are hard to tell apart.
    with pytest.raises(NotImplementedError):
        Fingerprint(dummy_fn)
