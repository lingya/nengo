import numpy as np

import nengo
from nengo.networks.ensemblearray import EnsembleArray
from nengo.utils.compat import is_number
from nengo.utils.optimization import sp_subvector_optimal_radius


# TODO unittest
class UnitEA(EnsembleArray):

    def __init__(self, n_neurons, dimensions, n_ensembles, ens_dimensions=1,
                 label=None, radius=1.0, eval_points=None, **ens_kwargs):
        if dimensions % n_ensembles != 0:
            raise ValueError(
                "'dimensions' has to be divisible by 'n_ensembles'.")

        if eval_points is None:
            eval_points = max(
                np.clip(500 * ens_dimensions, 750, 2500), 2 * n_neurons)
        if is_number(eval_points):
            n_points = eval_points
        else:
            n_points = len(eval_points)

        scaled_r = radius * sp_subvector_optimal_radius(
            dimensions, dimensions // n_ensembles, ens_dimensions, n_points)

        super(UnitEA, self).__init__(
            n_neurons, n_ensembles, ens_dimensions, label=label,
            radius=scaled_r, eval_points=eval_points, **ens_kwargs)
