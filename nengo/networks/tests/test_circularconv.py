import logging

import numpy as np
import pytest

import nengo
from nengo.networks import EnsembleArray
from nengo.networks.circularconvolution import circconv
from nengo.utils.compat import range
from nengo.utils.numpy import rmse
from nengo.utils.testing import Plotter

logger = logging.getLogger(__name__)


@pytest.mark.parametrize('invert_a', [True, False])
@pytest.mark.parametrize('invert_b', [True, False])
def test_circularconv_transforms(invert_a, invert_b):
    """Test the circular convolution transforms"""
    rng = np.random.RandomState(43232)

    dims = 100
    x = rng.randn(dims)
    y = rng.randn(dims)
    z0 = circconv(x, y, invert_a=invert_a, invert_b=invert_b)

    cconv = nengo.networks.CircularConvolution(
        1, dims, invert_a=invert_a, invert_b=invert_b)
    XY = np.dot(cconv.transformA, x) * np.dot(cconv.transformB, y)
    z1 = np.dot(cconv.transform_out, XY)

    assert np.allclose(z0, z1)

def test_neural_accuracy(Simulator, dims=16, neurons_per_product=256):
    rng = np.random.RandomState(4238)
    a = rng.normal(scale=np.sqrt(1./dims), size=dims)
    b = rng.normal(scale=np.sqrt(1./dims), size=dims)
    result = circconv(a, b)

    model = nengo.Network(label="circular conv", seed=1)
    model.config[nengo.Ensemble].neuron_type = nengo.LIFRate()
    with model:
        inputA = nengo.Node(a)
        inputB = nengo.Node(b)
        cconv = nengo.networks.CircularConvolution(
            neurons_per_product, dimensions=dims)
        nengo.Connection(inputA, cconv.A, synapse=None)
        nengo.Connection(inputB, cconv.B, synapse=None)
        res_p = nengo.Probe(cconv.output)
        p_p = nengo.Probe(cconv.product.A)
    sim = Simulator(model)
    sim.run(0.01)

    rmse = np.sqrt(np.mean((result - sim.data[res_p][-1])**2))
    assert rmse < 0.1

    '''
    print 'a', np.sqrt(np.sum(a**2))
    print 'b', np.sqrt(np.sum(b**2))

    print cconv.product.all_ensembles[0].radius
    print sim.data[p_p][-1]

    print 'desired result', result
    print 'result', sim.data[res_p][-1]
    print 'rmse', rmse
    assert False
    '''

if __name__ == "__main__":
    nengo.log(debug=True)
    pytest.main([__file__, '-v'])
