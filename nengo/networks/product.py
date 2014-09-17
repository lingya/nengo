import numpy as np

import nengo
from nengo.networks.ensemblearray import EnsembleArray
from nengo.utils.distributions import Choice


prod_config = nengo.Config(nengo.Ensemble)
prod_config[nengo.Ensemble].encoders = Choice(
    [[1, 1], [1, -1], [-1, 1], [-1, -1]])


def Product(n_neurons, dimensions, radius_in=1.0, net=None):
    """Computes the element-wise product of two equally sized vectors."""
    if net is None:
        net = nengo.Network(label="Product")

    with net:
        net.A = nengo.Node(size_in=dimensions, label="A")
        net.B = nengo.Node(size_in=dimensions, label="B")
        net.output = nengo.Node(size_in=dimensions, label="output")

        # radius_in * radius_in should be the biggest possible value,
        # but we'll add in a bit of a fudge factor too
        prod_config[nengo.Ensemble].radius = radius_in * radius_in * 1.1
        with prod_config:
            net.product = EnsembleArray(
                n_neurons, n_ensembles=dimensions, ens_dimensions=2)
        nengo.Connection(net.A, net.product.input[::2], synapse=None)
        nengo.Connection(net.B, net.product.input[1::2], synapse=None)
        nengo.Connection(
            net.product.add_output('product', lambda x: x[0] * x[1]),
            net.output, synapse=None)

    return net


def dot_product_transform(dimensions, scale=1.0):
    """Returns a transform for output to compute the scaled dot product."""
    return scale*np.ones((1, dimensions))
