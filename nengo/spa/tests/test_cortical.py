import numpy as np
import pytest

import nengo
from nengo import spa


def test_connect(Simulator):
    with spa.SPA(seed=122) as model:
        model.buffer1 = spa.Buffer(dimensions=16)
        model.buffer2 = spa.Buffer(dimensions=16)
        model.buffer3 = spa.Buffer(dimensions=16)
        model.cortical = spa.Cortical(spa.Actions('buffer2=buffer1',
                                                  'buffer3=~buffer1'))
        model.input = spa.Input(buffer1='A')

    output2, vocab = model.get_module_output('buffer2')
    output3, vocab = model.get_module_output('buffer3')

    with model:
        p2 = nengo.Probe(output2, 'output', synapse=0.03)
        p3 = nengo.Probe(output3, 'output', synapse=0.03)

    sim = Simulator(model)
    sim.run(0.2)

    match = np.dot(sim.data[p2], vocab.parse('A').v)
    assert match[199] > 0.9
    match = np.dot(sim.data[p3], vocab.parse('~A').v)
    assert match[199] > 0.9


def test_transform(Simulator):
    with spa.SPA(seed=123) as model:
        model.buffer1 = spa.Buffer(dimensions=16)
        model.buffer2 = spa.Buffer(dimensions=16)
        model.cortical = spa.Cortical(spa.Actions('buffer2=buffer1*B'))
        model.input = spa.Input(buffer1='A')

    output, vocab = model.get_module_output('buffer2')

    with model:
        p = nengo.Probe(output, 'output', synapse=0.03)

    sim = Simulator(model)
    sim.run(0.2)

    match = np.dot(sim.data[p], vocab.parse('A*B').v)
    assert match[199] > 0.7


def test_translate(Simulator):
    with spa.SPA(seed=123) as model:
        model.buffer1 = spa.Buffer(dimensions=16)
        model.buffer2 = spa.Buffer(dimensions=32)
        model.input = spa.Input(buffer1='A')
        model.cortical = spa.Cortical(spa.Actions('buffer2=buffer1'))

    output, vocab = model.get_module_output('buffer2')

    with model:
        p = nengo.Probe(output, 'output', synapse=0.03)

    sim = Simulator(model)
    sim.run(0.2)

    match = np.dot(sim.data[p], vocab.parse('A').v)
    assert match[199] > 0.8


def test_errors():
    # buffer2 does not exist
    with pytest.raises(NameError):
        with spa.SPA() as model:
            model.buffer = spa.Buffer(dimensions=16)
            model.cortical = spa.Cortical(spa.Actions('buffer2=buffer'))

    # conditional expressions not implemented
    with pytest.raises(NotImplementedError):
        with spa.SPA() as model:
            model.buffer = spa.Buffer(dimensions=16)
            model.cortical = spa.Cortical(spa.Actions(
                'dot(buffer,A) --> buffer=buffer'))

    # dot products not implemented
    with pytest.raises(NotImplementedError):
        with spa.SPA() as model:
            model.scalar = spa.Buffer(dimensions=1, subdimensions=1)
            model.cortical = spa.Cortical(spa.Actions(
                'scalar=dot(scalar, FOO)'))

    # convolution not implemented
    with pytest.raises(NotImplementedError):
        with spa.SPA() as model:
            model.unitary = spa.Buffer(dimensions=16)
            model.cortical = spa.Cortical(spa.Actions(
                'unitary=unitary*unitary'))


def test_direct(Simulator):
    with spa.SPA(seed=123) as model:
        model.buffer1 = spa.Buffer(dimensions=16)
        model.buffer2 = spa.Buffer(dimensions=32)
        model.cortical = spa.Cortical(spa.Actions('buffer1=A', 'buffer2=B',
                                                  'buffer1=C, buffer2=C'))

    output1, vocab1 = model.get_module_output('buffer1')
    output2, vocab2 = model.get_module_output('buffer2')

    with model:
        p1 = nengo.Probe(output1, 'output', synapse=0.03)
        p2 = nengo.Probe(output2, 'output', synapse=0.03)

    sim = Simulator(model)
    sim.run(0.2)

    match1 = np.dot(sim.data[p1], vocab1.parse('A+C').v)
    match2 = np.dot(sim.data[p2], vocab2.parse('A+C').v)
    assert match1[199] > 0.45
    assert match2[199] > 0.45

if __name__ == '__main__':
    nengo.log(debug=True)
    pytest.main([__file__, '-v'])
