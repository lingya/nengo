import numpy as np

import nengo
from nengo.spaopt.product import Product


class DotProduct(nengo.Network):
    """Computes the dot product of two (scaled) unit vectors.

    Requires Scipy.
    """

    def __init__(self, n_neurons, dimensions, radius=1.0, eval_points=None,
                 **ens_kwargs):
        self.A = nengo.Node(size_in=dimensions, label="A")
        self.B = nengo.Node(size_in=dimensions, label="B")
        self.output = nengo.Node(size_in=1, label="output")
        self.dimensions = dimensions
        self.radius = radius

        self.product = Product(
            n_neurons, dimensions, radius, eval_points, **ens_kwargs)

        nengo.Connection(self.A, self.product.A, synapse=None)
        nengo.Connection(self.B, self.product.B, synapse=None)

        nengo.Connection(
            self.product.output, self.output, synapse=None,
            transform=self.product.dot_product_transform())
