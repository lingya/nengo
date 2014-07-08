import numpy as np

import nengo
from nengo.spaopt.unitea import UnitEA


class DotProduct(nengo.Network):
    """Computes the dot product of two (scaled) unit vectors.

    Requires Scipy.
    """

    def __init__(self, n_neurons, dimensions, radius=1.0, eval_points=None,
                 **ens_kwargs):
        self.config[nengo.Ensemble].update(ens_kwargs)
        self.A = nengo.Node(size_in=dimensions, label="A")
        self.B = nengo.Node(size_in=dimensions, label="B")
        self.output = nengo.Node(size_in=1, label="output")
        self.dimensions = dimensions
        self.radius = radius

        self.product = UnitEA(
            n_neurons, dimensions, dimensions, 2,
            radius=radius, eval_points=eval_points)

        nengo.Connection(
            self.A, self.product.input[::2], synapse=None)
        nengo.Connection(
            self.B, self.product.input[1::2], synapse=None)

        nengo.Connection(
            self.product.add_output('product', lambda x: x[0] * x[1]),
            self.output, synapse=None, transform=[dimensions * [1.0]])
