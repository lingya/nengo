import logging

import numpy as np
import pytest

import nengo
# from nengo.builder import ShapeMismatch
from nengo.utils.testing import Plotter

logger = logging.getLogger(__name__)


def test_ensemblearray(Simulator):

    dims = 4
    n_neurons = 100

    rng = np.random.RandomState(82)
    x = rng.uniform(-1, 1, size=dims)

    model = nengo.Network(seed=28)
    with model:
        u = nengo.Node(x)
        a = nengo.networks.EnsembleArray(n_neurons, dims)
        nengo.Connection(u, a.input)
        ap = nengo.Probe(a.output, synapse=0.03)

    sim = Simulator(model)
    sim.run(1.)

    t = sim.trange()
    values = sim.data[ap]
    y = values[-100:]

    with Plotter(Simulator) as plt:
        plt.plot(t, np.ones((len(t), 1)) * x)
        plt.plot(t, values)
        plt.savefig('test_optimizer.test_ensemblearray.pdf')
        plt.close()

    assert np.allclose(x, y, atol=0.05, rtol=0)


if __name__ == "__main__":
    nengo.log(debug=True)
    pytest.main([__file__, '-v'])
