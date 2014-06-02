import logging

import numpy as np
import pytest

import nengo
from nengo.builder import SimNeurons
from nengo.utils.testing import Plotter

logger = logging.getLogger(__name__)


def test_lif(Simulator):
    n_ens = 4
    n_neurons = 100

    rng = np.random.RandomState(90)
    x = rng.uniform(-1, 1, size=n_ens)

    m = nengo.Network(seed=42)
    with m:
        u = nengo.Node(output=x)
        ensembles = [nengo.Ensemble(n_neurons, 1, neuron_type=nengo.LIF())
                     for i in range(n_ens)]
        probes = [nengo.Probe(ens, synapse=0.03) for ens in ensembles]

        for i, ens in enumerate(ensembles):
            nengo.Connection(u[i], ens)

    sim = Simulator(m)
    simneurons = [op for op in sim.model.all_ops if isinstance(op, SimNeurons)]
    assert len(simneurons) == 1

    sim.run(1.0)
    values = np.squeeze(np.array([sim.data[p] for p in probes])).T
    y = values[-100:]

    with Plotter(Simulator) as plt:
        t = sim.trange()
        plt.plot(t, np.ones((len(t), 1)) * x, '--')
        plt.plot(t, values)
        plt.savefig('test_optimizer.test_lif.pdf')
        plt.close()

    assert np.allclose(y, x, atol=0.05, rtol=0)


if __name__ == "__main__":
    nengo.log(debug=True)
    pytest.main([__file__, '-v'])
