import hashlib
import inspect
import logging
import os
import os.path
try:
    import cPickle as pickle
except ImportError:
    import pickle
import struct
import warnings

import numpy as np

logger = logging.getLogger(__name__)


class DecoderCache(object):
    _DECODER_EXT = '.npy'
    _SOLVER_INFO_EXT = '.pkl'
    DEFAULT_DIR = os.path.expanduser(os.path.join('~', '.nengo-cache'))

    def __init__(self, read_only=False, cache_dir=None):
        self.read_only = read_only
        if cache_dir is None:
            cache_dir = self.DEFAULT_DIR
        self.cache_dir = cache_dir
        if not os.path.exists(self.cache_dir):
            os.mkdir(self.cache_dir)

    # TODO test this function
    def invalidate(self):
        for filename in os.listdir(self.cache_dir):
            is_cache_file = filename.endswith(self._DECODER_EXT) or \
                filename.endswith(self._SOLVER_INFO_EXT)
            if is_cache_file:
                os.unlink(os.path.join(self.cache_dir, filename))

    def __call__(self, solver):
        def cached_solver(activities, targets, rng=None, E=None):
            args, _, _, defaults = inspect.getargspec(solver)
            args = args[-len(defaults):]
            if rng is None and 'rng' in args:
                rng = defaults[args.index('rng')]
            if E is None and 'E' in args:
                E = defaults[args.index('E')]

            key = self._get_cache_key(solver, activities, targets, rng, E)
            decoder_path = self._get_decoder_path(key)
            solver_info_path = self._get_solver_info_path(key)
            if os.path.exists(decoder_path):
                logger.info(
                    "Cache hit [{0}]: Loading stored decoders.".format(key))
                decoders = np.load(decoder_path)
                if os.path.exists(solver_info_path):
                    logger.info(
                        "Cache hit [{0}]: Loading stored solver info.".format(
                            key))
                    with open(solver_info_path, 'rb') as f:
                        solver_info = pickle.load(f)
                else:
                    warnings.warn(
                        "Loaded cached decoders [{0}], but could not find "
                        "cached solver info. It will be empty.".format(key),
                        RuntimeWarning)
                    solver_info = {}
            else:
                logger.info("Cache miss [{0}].".format(key))
                decoders, solver_info = solver(
                    activities, targets, rng=rng, E=E)
                if not self.read_only:
                    np.save(decoder_path, decoders)
                    with open(solver_info_path, 'wb') as f:
                        pickle.dump(solver_info, f)
            return decoders, solver_info
        return cached_solver

    def _get_cache_key(self, solver, activities, targets, rng, E):
        h = hashlib.sha1()

        h.update(solver.__module__)
        h.update(solver.__name__)

        h.update(activities.data)
        h.update(targets.data)

        # rng format doc: http://docs.scipy.org/doc/numpy/reference/generated/numpy.random.RandomState.get_state.html#numpy.random.RandomState.get_state
        state = rng.get_state()
        h.update(state[0])  # string 'MT19937'
        h.update(state[1].data)  # 1-D array of 624 unsigned integer keys
        h.update(struct.pack('q', state[2]))  # integer pos
        h.update(struct.pack('q', state[3]))  # integer has_gauss
        h.update(struct.pack('d', state[4]))  # float cached_gaussian

        if E is not None:
            h.update(E.data)
        return h.hexdigest()

    def _get_decoder_path(self, key):
        return os.path.join(self.cache_dir, key + self._DECODER_EXT)

    def _get_solver_info_path(self, key):
        return os.path.join(self.cache_dir, key + self._SOLVER_INFO_EXT)


def null_decoder_caching_handler(solver):
    return solver
