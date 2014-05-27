from __future__ import absolute_import

import numpy as np
from numpy.testing import assert_equal
import pytest

import nengo
import nengo.utils.ensemble


@pytest.mark.parametrize('dimensions', [1, 2])
def test_tuning_curves(dimensions):
    model = nengo.Network(label='test_tuning_curves')
    with model:
        ens = nengo.Ensemble(10, neuron_type=nengo.Direct(), dimensions=dimensions)
    sim = nengo.Simulator(model)

    eval_points, activities = nengo.utils.ensemble.tuning_curves(ens, sim)
    # eval_points is passed through in direct mode neurons
    assert_equal(eval_points, activities)


def test_tuning_curves_along_pref_direction():
    model = nengo.Network(label='test_tuning_curves')
    with model:
        ens = nengo.Ensemble(nengo.Direct(30), dimensions=10, radius=1.5)
    sim = nengo.Simulator(model)

    x, activities = nengo.utils.ensemble.tuning_curves_along_pref_direction(
        ens, sim)
    assert x.ndim == 1 and x.size > 0
    assert np.all(-1.5 <= x) and np.all(x <= 1.5)
    # eval_points is passed through in direct mode neurons
    assert_equal(x, activities)


if __name__ == '__main__':
    nengo.log(debug=True)
    pytest.main([__file__, '-v'])
