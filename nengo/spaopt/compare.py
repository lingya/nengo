import numpy as np

import nengo
from nengo.spaopt.dotproduct import DotProduct
from nengo.spa.module import Module
from nengo.utils.compat import range


class Compare(Module):
    """A module for computing the dot product of two inputs.

    Parameters
    ----------
    dimensions : int
        Number of dimensions for the two vectors to be compared
    vocab : Vocabulary, optional
        The vocabulary to use to interpret the vectors
    neurons_per_multiply : int
        Number of neurons to use in each product computation
    output_scaling : float
        Multiplier on the dot product result
    radius : float
        Effective radius for the multiplication.  The actual radius will
        be this value times sqrt(2)
    direct : boolean
        Whether or not to use direct mode for the neurons
    """
    def __init__(self, dimensions, vocab=None, neurons_per_multiply=200,
                 output_scaling=1.0, radius=1.0, direct=False):
        super(Compare, self).__init__()
        if vocab is None:
            # use the default vocab for this number of dimensions
            vocab = dimensions

        self.output_scaling = output_scaling

        self.compare = DotProduct(
            neurons_per_multiply, dimensions, radius=radius,
            neuron_type=nengo.Diret() if direct else nengo.LIF(),
            label='compare')

        self.inputA = nengo.Node(size_in=dimensions, label='inputA')
        self.inputB = nengo.Node(size_in=dimensions, label='inputB')
        self.output = nengo.Node(size_in=dimensions, label='output')

        self.inputs = dict(A=(self.inputA, vocab), B=(self.inputB, vocab))
        self.outputs = dict(default=(self.output, vocab))

        nengo.Connection(self.inputA, self.compare.A)
        nengo.Connection(self.inputB, self.compare.B)

    def on_add(self, spa):
        Module.on_add(self, spa)

        vocab = self.outputs['default'][1]

        transform = np.array([vocab.parse('YES').v] * vocab.dimensions)

        nengo.Connection(self.compare.product.product, self.output,
                         transform=transform.T * self.output_scaling)
