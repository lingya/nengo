import numpy as np
import pytest

import nengo
from nengo import builder
from nengo.tests.helpers import rms

import logging
logger = logging.getLogger(__name__)


def test_lif_builtin():
    """Test that the dynamic model approximately matches the rates."""
    rng = np.random.RandomState(85243)

    dt = 1e-3
    t_final = 1.0

    N = 10
    lif = nengo.LIF(N)
    lif.set_gain_bias(rng.uniform(80, 100, size=N), rng.uniform(-1, 1, size=N))

    x = np.arange(-2, 2, .1).reshape(-1, 1)
    J = lif.gain * x + lif.bias

    voltage = np.zeros_like(J)
    reftime = np.zeros_like(J)

    spikes = np.zeros((t_final / dt,) + J.shape)
    for i, spikes_i in enumerate(spikes):
        lif.step_math0(dt, J, voltage, reftime, spikes_i)

    math_rates = lif.rates(x)
    sim_rates = spikes.sum(0)
    assert np.allclose(sim_rates, math_rates, atol=1, rtol=0.02)


def test_lif_base(nl_nodirect):
    """Test that the dynamic model approximately matches the rates"""
    rng = np.random.RandomState(85243)

    dt = 0.001
    n = 5000
    x = 0.5

    m = nengo.Model("")

    ins = nengo.Node(x)
    lif = nl_nodirect(n)
    lif.set_gain_bias(max_rates=rng.uniform(low=10, high=200, size=n),
                      intercepts=rng.uniform(low=-1, high=1, size=n))
    lif.add_to_model(m)
    nengo.Connection(ins, lif, transform=np.ones((n, 1)))
    spike_probe = nengo.Probe(lif, 'output')

    sim = nengo.Simulator(m, dt=dt)

    t_final = 1.0
    sim.run(t_final)
    spikes = sim.data(spike_probe).sum(0)

    math_rates = lif.rates(x)
    sim_rates = spikes / t_final
    logger.debug("ME = %f", (sim_rates - math_rates).mean())
    logger.debug("RMSE = %f",
                 rms(sim_rates - math_rates) / (rms(math_rates) + 1e-20))
    assert np.sum(math_rates > 0) > 0.5 * n, (
        "At least 50% of neurons must fire")
    assert np.allclose(sim_rates, math_rates, atol=1, rtol=0.02)


if __name__ == "__main__":
    nengo.log(debug=True)
    pytest.main([__file__, '-v'])
