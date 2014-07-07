import numpy as np

import nengo
from nengo.networks.ensemblearray import EnsembleArray
from nengo.utils.compat import is_number
from nengo.utils.distributions import SqrtBeta, Uniform
from nengo.utils.optimization import sp_subvector_optimal_radius


class DotProduct(nengo.Network):
    """Computes the dot product of two (scaled) unit vectors.

    Requires Scipy.
    """

    def __init__(self, n_neurons, dimensions, radius=1, eval_points=None,
                 rng=np.random, **ens_kwargs):
        self.config[nengo.Ensemble].update(ens_kwargs)
        self.A = nengo.Node(size_in=dimensions, label="A")
        self.B = nengo.Node(size_in=dimensions, label="B")
        self.output = nengo.Node(size_in=1, label="output")
        self.dimensions = dimensions
        self.radius = radius

        if eval_points is None:
            eval_points = 1000

        if is_number(eval_points):
            num_eval_points = eval_points
        else:
            num_eval_points = len(eval_points)

        scaled_r = radius * sp_subvector_optimal_radius(
            dimensions, 1, 2, num_eval_points)

        if is_number(eval_points):
            xs = np.linspace(
                -scaled_r, scaled_r, int(np.sqrt(num_eval_points)))
            xs, ys = np.meshgrid(xs, xs)
            eval_points = np.vstack((xs.flat, ys.flat)).T

        self.product = EnsembleArray(
            n_neurons, n_ensembles=dimensions, ens_dimensions=2,
            radius=scaled_r, eval_points=eval_points)

        nengo.Connection(
            self.A, self.product.input[::2], synapse=None)
        nengo.Connection(
            self.B, self.product.input[1::2], synapse=None)

        nengo.Connection(
            self.product.add_output('product', lambda x: x[0] * x[1]),
            self.output, synapse=None, transform=[dimensions * [1.0]])
