"""Caching capabilities for a faster build process."""

import collections
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
from types import LambdaType
import warnings

import numpy as np

import nengo.utils.appdirs
import nengo.version

logger = logging.getLogger(__name__)


class Fingerprint(object):
    """Fingerprint of an object instance.

    A finger print is equal for two instances if and only if they are of the
    same type and have the same attributes.

    The fingerprint will be used as identification for caching.

    Parameters
    ----------
    obj : object
        Object to fingerprint.
    """
    def __init__(self, obj):
        self.fingerprint = hashlib.sha1()
        self._process(obj)

    def _process(self, obj):
        if obj is None:
            self.fingerprint.update(b'None')
        elif isinstance(obj, bool):
            self.fingerprint.update(struct.pack('?', obj))
        elif isinstance(obj, int):
            self.fingerprint.update(struct.pack('i', obj))
        elif isinstance(obj, long):
            self.fingerprint.update(struct.pack('l', obj))
        elif isinstance(obj, float):
            self.fingerprint.update(struct.pack('d', obj))
        elif isinstance(obj, complex):
            self.fingerprint.update(struct.pack('dd', obj.real, obj.imag))
        elif isinstance(obj, bytes):
            self.fingerprint.update(repr(obj))
        elif isinstance(obj, str) or isinstance(obj, unicode):
            self.fingerprint.update(repr(obj).encode())
        elif isinstance(obj, np.ndarray):
            self.fingerprint.update(obj.data)
        elif isinstance(obj, collections.Mapping):
            for k, v in obj.items():
                self._process(k)
                self._process(v)
        elif isinstance(obj, collections.Iterable):
            for item in obj:
                self._process(item)
        elif inspect.isfunction(obj):
            raise NotImplementedError(
                "Fingerprinting not supported for {t}.".format(t=type(obj)))
        elif self._is_class_instance(obj):
            self._process(obj.__class__.__module__)
            self._process(obj.__class__.__name__)
            self._process(obj.__dict__)
        else:
            raise NotImplementedError(
                "Fingerprinting not supported for {t}.".format(t=type(obj)))

    @staticmethod
    def _is_class_instance(obj):
        return hasattr(obj, '__class__') and hasattr(obj, '__dict__')

    def __str__(self):
        return self.fingerprint.hexdigest()


class DecoderCache(object):
    """Cache for decoders.

    Hashes the arguments to the decoder solver and stores the result in a file
    which will be reused in later calls with the same arguments.

    Be aware that decoders should not use any global state, but only values
    passed and attributes of the object instance. Otherwise the wrong solver
    results might get loaded from the cache.

    Parameters
    ----------
    read_only : bool
        Indicates that already existing items in the cache will be used, but no
        new items will be written to the disk in case of a cache miss.
    cache_dir : str or None
        Path to the directory in which the cache will be stored. It will be
        created if it does not exists. Will use the value returned by
        :func:`get_default_dir`, if `None`.
    """

    _DECODER_EXT = '.npy'
    _SOLVER_INFO_EXT = '.pkl'

    def __init__(self, read_only=False, cache_dir=None):
        self.read_only = read_only
        if cache_dir is None:
            cache_dir = self.get_default_dir()
        self.cache_dir = cache_dir
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)

    def get_size(self):
        """Returns the size of the cache in bytes.

        Returns
        -------
        int
        """
        size = 0
        for filename in os.listdir(self.cache_dir):
            size += os.stat(os.path.join(self.cache_dir, filename)).st_size
        return size

    def shrink(self, limit=100):
        """Reduces the number of cached decoder matrices to meet a limit.

        Parameters
        ----------
        limit : int
            Maximum number of decoder matrices to keep.
        """
        filelist = []
        for filename in os.listdir(self.cache_dir):
            key, ext = os.path.splitext(filename)
            if ext == self._SOLVER_INFO_EXT:
                continue
            path = os.path.join(self.cache_dir, filename)
            stat = os.stat(path)
            filelist.append((stat.st_atime, key))
        filelist.sort()

        excess = len(filelist) - limit
        for _, key in filelist:
            if excess <= 0:
                break
            excess -= 1

            decoder_path = os.path.join(
                self.cache_dir, key + self._DECODER_EXT)
            solver_info_path = os.path.join(
                self.cache_dir, key + self._SOLVER_INFO_EXT)

            os.unlink(decoder_path)
            if os.path.exists(solver_info_path):
                os.unlink(solver_info_path)

    def invalidate(self):
        """Invalidates the cache (i.e. removes all stored decoder matrices)."""
        for filename in os.listdir(self.cache_dir):
            is_cache_file = filename.endswith(self._DECODER_EXT) or \
                filename.endswith(self._SOLVER_INFO_EXT)
            if is_cache_file:
                os.unlink(os.path.join(self.cache_dir, filename))

    @classmethod
    def get_default_dir(cls):
        """Returns the default location of the cache.

        Returns
        -------
        str
        """
        return os.path.join(nengo.utils.appdirs.user_cache_dir(
            nengo.version.name, nengo.version.author), 'decoders')

    def wrap_solver(self, solver):
        """Takes a decoder solver and wraps it to use caching.

        Parameters
        ----------
        solver : func
            Decoder solver to wrap for caching.

        Returns
        -------
        func
            Wrapped decoder solver.
        """
        def cached_solver(activities, targets, rng=None, E=None):
            try:
                args, _, _, defaults = inspect.getargspec(solver)
            except TypeError:
                args, _, _, defaults = inspect.getargspec(solver.__call__)
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
        return str(Fingerprint((
            solver, activities, targets, rng.get_state(), E)))

    def _get_decoder_path(self, key):
        return os.path.join(self.cache_dir, key + self._DECODER_EXT)

    def _get_solver_info_path(self, key):
        return os.path.join(self.cache_dir, key + self._SOLVER_INFO_EXT)


class NoDecoderCache(object):
    """Provides the same interface as :class:`DecoderCache`, but does not
    perform any caching."""

    def wrap_solver(self, solver):
        return solver

    def get_size(self):
        return 0

    def shrink(self, limit=0):
        pass

    def invalidate(self):
        pass
