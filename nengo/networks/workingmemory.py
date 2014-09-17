import numpy as np

import nengo
from nengo.networks import EnsembleArray

mem_config = nengo.Config(nengo.Ensemble, nengo.Connection)
mem_config[nengo.Connection].synapse = 0.1


def InputGatedMemory(n_neurons, dimensions, fdbk_scale=1.0, gate_gain=10,
                     difference_gain=1.0, reset_gain=3, net=None):
    """Stores a given vector in memory, with input controlled by a gate."""
    if net is None:
        net = nengo.Network(label="Input Gated Memory")

    n_total_neurons = n_neurons * dimensions

    with net:
        # integrator to store value
        with mem_config:
            net.mem = EnsembleArray(n_neurons, dimensions,
                                    neuron_nodes=True, label="mem")
            nengo.Connection(net.mem.output, net.mem.input,
                             transform=fdbk_scale)

        # calculate difference between stored value and input
        net.diff = EnsembleArray(n_neurons, dimensions,
                                 neuron_nodes=True, label="diff")

        nengo.Connection(net.mem.output, net.diff.input,
                         transform=-1)

        # feed difference into integrator
        with mem_config:
            nengo.Connection(net.diff.output, net.mem.input,
                             transform=difference_gain)

        # gate difference (if gate==0, update stored value,
        # otherwise retain stored value)
        net.gate = nengo.Node(size_in=1)
        nengo.Connection(net.gate, net.diff.neuron_input,
                         transform=np.ones((n_total_neurons, 1)) * -gate_gain)

        # reset input (if reset=1, remove all values stored, and set to 0)
        net.reset_node = nengo.Node(size_in=1)
        nengo.Connection(net.reset_node, net.mem.neuron_input,
                         transform=np.ones((n_total_neurons, 1)) * -reset_gain)

    net.input = net.diff.input
    net.output = net.mem.output
    return net


def FeedbackGatedMemory(n_neurons, dimensions, fdbk_synapse=0.1,
                        conn_synapse=0.005, fdbk_scale=1.0, gate_gain=2.0,
                        reset_gain=3, net=None):
    """Stores a given vector in memory, with input controlled by a gate."""
    if net is None:
        net = nengo.Network(label="Feedback Gated Memory")

    n_total_neurons = n_neurons * dimensions

    with net:
        # gate control signal (if gate==0, update stored value, otherwise
        # retain stored value)
        net.gate = nengo.Node(size_in=1)

        # integrator to store value
        net.mem = EnsembleArray(n_neurons, dimensions,
                                neuron_nodes=True, label="mem")

        # ensemble to gate feedback
        net.fdbk = EnsembleArray(n_neurons, dimensions,
                                 neuron_nodes=True, label="fdbk")

        # ensemble to gate input
        net.in_gate = EnsembleArray(n_neurons, dimensions,
                                    neuron_nodes=True, label="in_gate")

        # calculate gating control signal
        net.ctrl = nengo.Ensemble(n_neurons, 1, label="ctrl")

        # Connection from mem to fdbk, and from fdbk to mem
        nengo.Connection(net.mem.output, net.fdbk.input,
                         synapse=fdbk_synapse - conn_synapse)
        nengo.Connection(net.fdbk.output, net.mem.input,
                         transform=fdbk_scale,
                         synapse=conn_synapse)

        # Connection from input to in_gate, and from in_gate to mem
        nengo.Connection(net.in_gate.output, net.mem.input,
                         synapse=conn_synapse)

        # Connection from gate to ctrl
        nengo.Connection(net.gate, net.ctrl, synapse=None)

        # Connection from ctrl to fdbk and in_gate
        nengo.Connection(net.ctrl, net.fdbk.neuron_input,
                         function=lambda x: [1 - x[0]],
                         transform=np.ones((n_total_neurons, 1)) * -gate_gain)
        nengo.Connection(net.ctrl, net.in_gate.neuron_input,
                         transform=np.ones((n_total_neurons, 1)) * -gate_gain)

        # reset input (if reset=1, remove all values stored, and set to 0)
        net.reset = nengo.Node(size_in=1)
        nengo.Connection(net.reset, net.mem.neuron_input,
                         transform=np.ones((n_total_neurons, 1)) * -reset_gain)

    net.input = net.in_gate.input
    net.output = net.mem.output
    return net
