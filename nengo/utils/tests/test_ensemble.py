from __future__ import absolute_import

import numpy as np
from numpy.testing import assert_equal
import pytest

import nengo
import nengo.utils.ensemble
from nengo.utils.distributions import Uniform
from nengo.utils.testing import Plotter


@pytest.mark.parametrize('dimensions', [1, 2])
def test_tuning_curves_direct_mode(Simulator, dimensions):
    model = nengo.Network(label='test_tuning_curves_direct', seed=4)
    with model:
        ens = nengo.Ensemble(10, neuron_type=nengo.Direct(), dimensions=dimensions)
    sim = Simulator(model)

    eval_points, activities = nengo.utils.ensemble.tuning_curves(
        ens, sim, apply_encoders=True)

    with Plotter(Simulator) as plt:
        plt.plot(eval_points, activities)
        plt.savefig('utils.test_ensemble.test_tuning_curves_direct_mode_%d.pdf'
                    % dimensions)
        plt.close()

    # eval_points is passed through in direct mode neurons
    assert_equal(eval_points, activities)


@pytest.mark.parametrize('dimensions', [1, 2])
def test_tuning_curves_normal_mode(Simulator, dimensions):
    max_rate = 400
    model = nengo.Network(label='test_tuning_curves', seed=4)
    with model:
        ens = nengo.Ensemble(
            10, neuron_type=nengo.LIF(), dimensions=dimensions,
            max_rates=Uniform(200, max_rate))
    sim = Simulator(model)

    eval_points, activities = nengo.utils.ensemble.tuning_curves(
        ens, sim, apply_encoders=True)

    with Plotter(Simulator) as plt:
        plt.plot(eval_points, activities)
        plt.savefig('utils.test_ensemble.test_tuning_curves_normal_mode_%d.pdf'
                    % dimensions)
        plt.close()

    assert np.all(activities >= 0)
    assert np.all(activities <= max_rate)


def test_tuning_curves_along_pref_direction_direct_mode(Simulator):
    model = nengo.Network(label='test_tuning_curves_direct', seed=4)
    with model:
        ens = nengo.Ensemble(30, neuron_type=nengo.Direct(), dimensions=10, radius=1.5)
    sim = Simulator(model)

    x, activities = nengo.utils.ensemble.tuning_curves(
        ens, sim, apply_encoders=False)

    with Plotter(Simulator) as plt:
        plt.plot(x, activities)
        plt.savefig('utils.test_ensemble.test_tuning_curves_along_pref_'
                    'direction_direct_mode.pdf')
        plt.close()

    assert x.ndim == 1 and x.size > 0
    assert np.all(-1.0 <= x) and np.all(x <= 1.0)
    # eval_points is passed through in direct mode neurons
    assert_equal(x, activities)


def test_tuning_curves_along_pref_direction_normal_mode(Simulator):
    max_rate = 400
    model = nengo.Network(label='test_tuning_curves', seed=4)
    with model:
        ens = nengo.Ensemble(
            30, neuron_type=nengo.LIF(), dimensions=10, radius=1.5,
            max_rates=Uniform(200, max_rate))
    sim = Simulator(model)

    x, activities = nengo.utils.ensemble.tuning_curves(
        ens, sim, apply_encoders=False)

    with Plotter(Simulator) as plt:
        plt.plot(x, activities)
        plt.savefig('utils.test_ensemble.test_tuning_curves_along_pref_'
                    'direction_normal_mode.pdf')
        plt.close()

    assert x.ndim == 1 and x.size > 0
    assert np.all(-1.0 <= x) and np.all(x <= 1.0)

    assert np.all(activities >= 0.0)
    assert np.all(activities <= max_rate)


if __name__ == '__main__':
    nengo.log(debug=True)
    pytest.main([__file__, '-v'])
