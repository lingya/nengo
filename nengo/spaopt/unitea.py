import numpy as np

import nengo
from nengo.networks.ensemblearray import EnsembleArray
from nengo.params import Default
from nengo.utils.compat import is_number
from nengo.utils.optimization import sp_subvector_optimal_radius


# TODO unittest
class UnitEA(EnsembleArray):

    def __init__(self, n_neurons, dimensions, n_ensembles, ens_dimensions=1,
                 label=None, radius=1.0, n_eval_points=Default,
                 eval_points=Default, **ens_kwargs):
        if dimensions % n_ensembles != 0:
            raise ValueError(
                "'dimensions' has to be divisible by 'n_ensembles'.")

        if eval_points is Default:
            if n_eval_points is Default:
                n_eval_points = max(
                    np.clip(500 * ens_dimensions, 750, 2500), 2 * n_neurons)

        scaled_r = radius * sp_subvector_optimal_radius(
            dimensions, dimensions // n_ensembles, ens_dimensions,
            n_eval_points if n_eval_points is not Default
            else len(eval_points))

        super(UnitEA, self).__init__(
            n_neurons, n_ensembles, ens_dimensions, label=label,
            radius=scaled_r, n_eval_points=n_eval_points,
            eval_points=eval_points, **ens_kwargs)
