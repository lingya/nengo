"""Tests for nengo.objects.Probe"""

import logging
import time

import numpy as np
import pytest

import nengo

logger = logging.getLogger(__name__)


def test_multirun(Simulator):
    """Test probing the time on multiple runs"""
    rng = np.random.RandomState(2239)

    # set rtol a bit higher, since OCL model.t accumulates error over time
    rtol = 0.0001

    model = nengo.Model("Multi-run")

    sim = Simulator(model)
    t_stops = sim.model.dt * rng.randint(low=100, high=2000, size=10)

    t_sum = 0
    for t_stop in t_stops:
        sim.run(t_stop)
        sim_t = sim.trange()
        t = sim.model.dt * np.arange(len(sim_t))
        assert np.allclose(sim_t, t, rtol=rtol)

        t_sum += t_stop
        assert np.allclose(sim_t[-1], t_sum - sim.model.dt, rtol=rtol)


def test_dts(Simulator, n_probes=10, simtime=2.483):
    """Test probes with different timesteps"""

    rng = np.random.RandomState(48392)
    dts = 0.001 * rng.randint(low=1, high=100, size=n_probes)

    def input_fn(t):
        """Just returns range(1, 10)"""
        _ = t
        return list(range(1, 10))

    model = nengo.Model('test_probe_dts', seed=2891)
    probes = []
    for i, dt in enumerate(dts):
        node_i = nengo.Node(label='x%d' % i, output=input_fn)
        probes.append(nengo.Probe(node_i, 'output', sample_every=dt))

    sim = Simulator(model)
    timer = time.time()
    sim.run(simtime)
    timer = time.time() - timer
    logger.debug("Ran %d probes for %f sec simtime in %0.3f sec",
                 n_probes, simtime, timer)

    for i, probe in enumerate(probes):
        t = sim.model.dt * np.arange(int(np.ceil(simtime / dts[i])))
        x = np.asarray([input_fn(tt) for tt in t])
        p_data = sim.data(probe)
        assert len(x) == len(p_data)
        assert np.allclose(p_data[1:], x[:-1])  # 1-step delay


def test_large(Simulator, n_probes=10, simtime=2.483):
    """Test with a lot of big probes. Can also be used for speed."""

    def input_fn(t):
        """Just returns range(1, 10)"""
        _ = t
        return list(range(1, 10))

    model = nengo.Model('test_large_probes', seed=3249)

    probes = []
    for i in range(n_probes):
        node_i = nengo.Node(label='x%d' % i, output=input_fn)
        probes.append(nengo.Probe(node_i, 'output'))

    sim = Simulator(model)
    timer = time.time()
    sim.run(simtime)
    timer = time.time() - timer
    logger.debug("Ran %d probes for %f sec simtime in %0.3f sec",
                 n_probes, simtime, timer)

    t = sim.model.dt * np.arange(int(np.round(simtime / sim.model.dt)))
    x = np.asarray([input_fn(ti) for ti in t])
    for probe in probes:
        assert np.allclose(sim.data(probe)[1:], x[:-1])  # 1-step delay


def test_defaults(Simulator):
    """Tests that probing with no attr sets the right attr."""
    model = nengo.Model('test_defaults')
    node = nengo.Node(output=0.5)
    ens = nengo.Ensemble(nengo.LIF(20), 1)
    conn = nengo.Connection(node, ens)
    node_p = nengo.Probe(node)
    assert node_p.attr == 'output'
    ens_p = nengo.Probe(ens)
    assert ens_p.attr == 'decoded_output'
    with pytest.raises(TypeError):
        nengo.Probe(conn)
    # Let's just make sure it runs too...
    sim = Simulator(model)
    sim.run(0.01)


if __name__ == "__main__":
    nengo.log(debug=True)
    pytest.main([__file__, '-v'])
