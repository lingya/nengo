import nengo


def Integrator(recurrent_tau, n_neurons, dimensions, net=None):
    if net is None:
        net = nengo.Network(label="Integrator")
    with net:
        net.integrator_input = nengo.Node(size_in=dimensions)
        net.ensemble = nengo.Ensemble(n_neurons, dimensions=dimensions)
        nengo.Connection(net.ensemble, net.ensemble, synapse=recurrent_tau)
        nengo.Connection(net.integrator_input, net.ensemble, transform=recurrent_tau)
    return net
