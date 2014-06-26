import numpy as np
import scipy.optimize
from scipy.special import beta, betainc

import nengo
from nengo.networks.ensemblearray import EnsembleArray
from nengo.utils.compat import is_number
from nengo.utils.distributions import SqrtBeta, Uniform


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

        res = scipy.optimize.minimize(
            lambda x: self._error_estimate(dimensions, num_eval_points, x),
            0.5)

        scaled_r = radius * res.x

        if is_number(eval_points):
            eval_points = Uniform(-scaled_r, scaled_r).sample(
                (num_eval_points, 2), rng=rng)

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

    @staticmethod
    def _in_range_error(num_eval_points, x):
        return (x / (np.sqrt(num_eval_points) - 1)) ** 2 / 3.0

    @staticmethod
    def _out_of_range_error(dimensions, num_eval_points, x):
        dist = SqrtBeta(dimensions - 1)
        sq_x = x * x

        normalization = 1.0 - dist.cdf(x)
        b = (dimensions - 1) / 2.0
        aligned_integral = beta(1.5, b) * (1.0 - betainc(1.5, b, sq_x))
        cross_integral = beta(1.0, b) * (1.0 - betainc(1.0, b, sq_x))

        numerator = (sq_x * normalization + (
            aligned_integral - 2.0 * x * cross_integral) / beta(0.5, b))
        return numerator / normalization

    @classmethod
    def _error_estimate(cls, dimensions, num_eval_points, x):
        dist = SqrtBeta(dimensions - 1)
        return (dist.cdf(x) * cls._in_range_error(num_eval_points, x) +
                (1.0 - dist.cdf(x)) * cls._out_of_range_error(
                    dimensions, num_eval_points, x))
