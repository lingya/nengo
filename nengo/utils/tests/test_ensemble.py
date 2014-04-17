from __future__ import absolute_import

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
