"""
Simulator.py

Reference simulator for nengo models.
"""

from __future__ import print_function

from collections import Mapping
import logging
import sys

import numpy as np

from nengo.builder import Model, Builder, SignalDict
import nengo.cache
from nengo.utils.graphs import toposort
from nengo.utils.simulator import operator_depencency_graph

logger = logging.getLogger(__name__)


class ProbeDict(Mapping):
    """Map from Probe -> ndarray

    This is more like a view on the dict that the simulator manipulates.
    However, for speed reasons, the simulator uses Python lists,
    and we want to return NumPy arrays. Additionally, this mapping
    is readonly, which is more appropriate for its purpose.
    """

    def __init__(self, raw):
        self.raw = raw

    def __getitem__(self, key):
        rval = self.raw[key]
        if isinstance(rval, list):
            rval = np.asarray(rval)
            rval.flags.writeable = False
        return rval

    def __str__(self):
        return str(self.raw)

    def __repr__(self):
        return repr(self.raw)

    def __iter__(self):
        return iter(self.raw)

    def __len__(self):
        return len(self.raw)


class Simulator(object):
    """Reference simulator for Nengo models."""

    def __init__(self, network, dt=0.001, seed=None, model=None, caching=None):
        """Initialize the simulator with a network and (optionally) a model.

        Most of the time, you will pass in a network and sometimes a dt::

            sim1 = nengo.Simulator(my_network)  # Uses default 0.001s dt
            sim2 = nengo.Simulator(my_network, dt=0.01)  # Uses 0.01s dt

        For more advanced use cases, you can initialize the model yourself,
        and also pass in a network that will be built into the same model
        that you pass in::

            sim = nengo.Simulator(my_network, model=my_model)

        If you want full control over the build process, then you can build
        your network into the model manually. If you do this, then you must
        explicitly pass in ``None`` for the network::

            sim = nengo.Simulator(None, model=my_model)

        Parameters
        ----------
        network : nengo.Network instance or None
            A network object to the built and then simulated.
            If a fully built ``model`` is passed in, then you can skip
            building the network by passing in network=None.
        dt : float
            The length of a simulator timestep, in seconds.
        seed : int
            A seed for all stochastic operators used in this simulator.
            Note that there are not stochastic operators implemented
            currently, so this parameters does nothing.
        model : nengo.builder.Model instance or None
            A model object that contains build artifacts to be simulated.
            Usually the simulator will build this model for you; however,
            if you want to build the network manually, or to inject some
            build artifacts in the Model before building the network,
            then you can pass in a ``nengo.builder.Model`` instance.
        caching : bool or 'ro' or None
            Whether to use decoder caching. If 'ro' is passed, the cache will
            only be read, but no new entries will be written. ``None`` enables
            the cache only if a network seed is set and the simulator is not
            created from a unit test.
        """
        self.dt = dt
        if model is None:
            in_test = hasattr(sys, '_called_from_test')
            network_seed_set = network is not None and network.seed is not None
            if caching:
                decoder_cache = nengo.cache.DecoderCache(caching == 'ro')
            elif caching is None and network_seed_set and not in_test:
                decoder_cache = nengo.cache.DecoderCache()
            else:
                decoder_cache = nengo.cache.NoDecoderCache()

            self.model = Model(dt=self.dt,
                               label="%s, dt=%f" % (network.label, dt),
                               seed=network.seed, decoder_cache=decoder_cache)
        else:
            self.model = model

        if network is not None:
            # Build the network into the model
            Builder.build(
                network, model=self.model)

        self.model.decoder_cache.shrink()

        # Use model seed as simulator seed if the seed is not provided
        # Note: seed is not used right now, but one day...
        self.seed = self.model.seed if seed is None else seed

        # -- map from Signal.base -> ndarray
        self.signals = SignalDict(__time__=np.asarray(0.0, dtype=np.float64))
        for op in self.model.operators:
            op.init_signals(self.signals, self.dt)

        self.dg = operator_depencency_graph(self.model.operators)
        self._step_order = [node for node in toposort(self.dg)
                            if hasattr(node, 'make_step')]
        self._steps = [node.make_step(self.signals, self.dt)
                       for node in self._step_order]

        self.n_steps = 0

        # Add built states to the probe dictionary
        self._probe_outputs = self.model.params

        # Provide a nicer interface to probe outputs
        self.data = ProbeDict(self._probe_outputs)

    def step(self):
        """Advance the simulator by `self.dt` seconds.
        """
        old_err = np.seterr(invalid='raise', divide='ignore')
        try:
            for step_fn in self._steps:
                step_fn()
        finally:
            np.seterr(**old_err)

        # -- probes signals -> probe buffers
        for probe in self.model.probes:
            period = (1 if probe.sample_every is None
                      else int(probe.sample_every / self.dt))
            if self.n_steps % period == 0:
                tmp = self.signals[self.model.sig[probe]['in']].copy()
                self._probe_outputs[probe].append(tmp)

        self.signals['__time__'] += self.dt
        self.n_steps += 1

    def run(self, time_in_seconds):
        """Simulate for the given length of time."""
        steps = int(np.round(float(time_in_seconds) / self.dt))
        logger.debug("Running %s for %f seconds, or %d steps",
                     self.model.label, time_in_seconds, steps)
        self.run_steps(steps)

    def run_steps(self, steps):
        """Simulate for the given number of `dt` steps."""
        for i in range(steps):
            if i % 1000 == 0:
                logger.debug("Step %d", i)
            self.step()

    def trange(self, dt=None):
        dt = self.dt if dt is None else dt
        n_steps = int(np.ceil(self.n_steps * self.dt / dt))
        return dt * np.arange(0, n_steps)
