import numpy as np

import nengo
from nengo.networks.ensemblearray import EnsembleArray
from nengo.utils.compat import is_number
from nengo.utils.optimization import sp_subvector_optimal_radius


# TODO unittest pure product
class Product(nengo.Network):
    """Computes the element-wise product of two (scaled) unit vectors.

    Requires Scipy.
    """

    def __init__(self, n_neurons, dimensions, radius=1.0, eval_points=None,
                 **ens_kwargs):
        self.A = nengo.Node(size_in=dimensions, label="A")
        self.B = nengo.Node(size_in=dimensions, label="B")
        self.output = nengo.Node(size_in=dimensions, label="output")
        self.dimensions = dimensions
        self.radius = radius

        if eval_points is None:
            eval_points = max(1000, 2 * n_neurons)
        if is_number(eval_points):
            n_points = eval_points
        else:
            n_points = len(eval_points)

        scaled_r = radius * sp_subvector_optimal_radius(
            dimensions, 1, 2, n_points)

        if is_number(eval_points):
            xs = np.linspace(
                -scaled_r, scaled_r, int(np.sqrt(n_points)))
            xs, ys = np.meshgrid(xs, xs)
            eval_points = np.vstack((xs.flat, ys.flat)).T

        self.product = EnsembleArray(
            n_neurons, n_ensembles=dimensions, ens_dimensions=2,
            radius=scaled_r, eval_points=eval_points, **ens_kwargs)

        nengo.Connection(
            self.A, self.product.input[::2], synapse=None)
        nengo.Connection(
            self.B, self.product.input[1::2], synapse=None)

        nengo.Connection(
            self.product.add_output('product', lambda x: x[0] * x[1]),
            self.output, synapse=None)

    def dot_product_transform(self, scale=1.0):
        """Returns a transform for output to compute the scaled dot product."""
        return scale * np.ones((1, self.dimensions))
