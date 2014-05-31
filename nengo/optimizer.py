import collections
import logging
import warnings

import numpy as np

import nengo.builder as nb
import nengo.neurons
# from nengo.utils.compat import groupby


class ModelOptimizer(object):

    optimizers = []

    @classmethod
    def register_optimizer(cls, optimizer_fn):
        cls.optimizers.append(optimizer_fn)

    @classmethod
    def optimize(cls, network, model):

        for optimizer in cls.optimizers:
            optimizer(network, model)

        return model


def walk_network(net):

    objs = net.ensembles + net.nodes + net.connections + net.probes
    for obj in objs:
        yield obj

    for subnet in net.networks:
        yield subnet
        for obj in walk_network(subnet):
            yield obj


def optimize_ensemblearrays(network, model):
    from nengo.networks import EnsembleArray

    arrays = [obj for obj in walk_network(network)
              if isinstance(obj, EnsembleArray)]

    for array in arrays:

        n_ensembles = array.n_ensembles
        assert n_ensembles == len(array.ensembles)
        n_neurons = array.ensembles[0].n_neurons
        assert all(ens.n_neurons == n_neurons for ens in array.ensembles)
        dims = array.dimensions_per_ensemble
        assert all(ens.dimensions == dims for ens in array.ensembles)
        # TODO: check that neuron types are equal

        model.sig[array]['neuron_in'] = nb.Signal(
            np.zeros((n_ensembles, n_neurons)),
            name="%s.neuron_in" % array.label)[:]
        model.sig[array]['neuron_out'] = nb.Signal(
            np.zeros((n_ensembles, n_neurons)),
            name="%s.neuron_out" % array.label)[:]


        for ens in array.ensembles:



            # make grouped sim object
            n_neurons = sum(op.ens.n_neurons for op in ops)
            Ens = collections.namedtuple('Ens', ['label', 'n_neurons', 'neuron_type'])
            ens = Ens(label="OptimizedLIF", n_neurons=n_neurons, neuron_type=neuron_type)


            builder = nb.Builder.builders[neuron_type.__class__]
            builder(neuron_type, ens, model=model, config=None)

            # map exisiting signal-views onto new signals
            states = OptimizableNeurons[neuron_type.__class__]
            i = 0
            for op in ops:
                n = op.ens.n_neurons
                model.sig['neuron_in'][op.ens].set(model.sig['neuron_in'][ens][i:i+n])
                model.sig['neuron_out'][op.ens].set(model.sig['neuron_out'][ens][i:i+n])
                for state in states:
                    model.sig[state][op.ens].set(model.sig[state][ens][i:i+n])
                model.operators.remove(op)
                i += n


        ensembles = array.ensembles
        print ensembles


ModelOptimizer.register_optimizer(optimize_ensemblearrays)


# OptimizableNeurons = {
#     nengo.neurons.LIFRate: [],
#     nengo.neurons.LIF: ['voltage', 'refractory_time'],
#     nengo.neurons.AdaptiveLIFRate: ['adaptation'],
#     nengo.neurons.AdaptiveLIF: ['voltage', 'refractory_time', 'adaptation'],
# }

# def optimize_neurons(model):
#     neurons = [op for op in model.operators if isinstance(op, nb.SimNeurons)]
#     for neuron_type, ops in groupby(neurons, lambda op: op.ens.neuron_type):

#         if len(ops) <= 1 or neuron_type.__class__ not in OptimizableNeurons:
#             continue

#         # make grouped sim object
#         n_neurons = sum(op.ens.n_neurons for op in ops)
#         Ens = collections.namedtuple('Ens', ['label', 'n_neurons', 'neuron_type'])
#         ens = Ens(label="OptimizedLIF", n_neurons=n_neurons, neuron_type=neuron_type)

#         model.sig['neuron_in'][ens] = nb.Signal(
#             np.zeros(ens.n_neurons), name="%s.neuron_in" % ens.label)[:]
#         model.sig['neuron_out'][ens] = nb.Signal(
#             np.zeros(ens.n_neurons), name="%s.neuron_out" % ens.label)[:]

#         builder = nb.Builder.builders[neuron_type.__class__]
#         builder(neuron_type, ens, model=model, config=None)

#         # map exisiting signal-views onto new signals
#         states = OptimizableNeurons[neuron_type.__class__]
#         i = 0
#         for op in ops:
#             n = op.ens.n_neurons
#             model.sig['neuron_in'][op.ens].set(model.sig['neuron_in'][ens][i:i+n])
#             model.sig['neuron_out'][op.ens].set(model.sig['neuron_out'][ens][i:i+n])
#             for state in states:
#                 model.sig[state][op.ens].set(model.sig[state][ens][i:i+n])
#             model.operators.remove(op)
#             i += n

# ModelOptimizer.register_optimizer(optimize_neurons)


# class MultiDotInc(nb.Operator):

#     def __init__(self, A, X, Y, tag=None):
#         self.A = A
#         self.X = X
#         self.Y = Y
#         self.tag = tag

#         self.reads = [self.A, self.X]
#         self.incs = [self.Y]
#         self.sets = []
#         self.updates = []

#     def __str__(self):
#         return 'MultiDotInc(%s, %s -> %s "%s")' % (
#             str(self.A), str(self.X), str(self.Y), self.tag)

#     def make_step(self, signals, dt):
#         X = signals[self.X]
#         A = signals[self.A]
#         Y = signals[self.Y]

#         def step_multidotinc():
#             Y[...] += np.sum(A * X[...,None,:], axis=-1)
#         return step_multidotinc


# def optimize_encoders(model):

#     encoders = [op for op in model.operators
#                 if isinstance(op, nb.DotInc)]

#     # group by output base and shape
#     for (base, shape), ops in groupby(
#             encoders, lambda op: (op.Y.base, op.Y.shape)):
#         if len(ops) <= 1:
#             continue
#         if len(shape) > 1:
#             continue  # not implemented yet

#         ops.sort(key=lambda op: op.Y.offset)

#         n = shape[0]
#         groups = []
#         group = [ops[0]]
#         for i in range(1, len(ops)):
#             if ops[i].Y.offset == ops[i-1].Y.offset + n:
#                 # contiguous
#                 group.append(ops[i])
#             else:
#                 # not contiguous, start a new group
#                 groups.append(group)
#                 group = [ops[i]]

#         groups.append(group)

#         for group in groups:
#             if len(group) == 1:
#                 import pdb; pdb.set_trace()
#                 continue

#             # create a new operator
#             n_neurons = n * len(group)
#             n_dims = len(group)
#             Ens = collections.namedtuple(
#                 'Ens', ['label', 'n_neurons', 'dimensions'])
#             ens = Ens(label="OptimizedLIF",
#                       n_neurons=n_neurons, dimensions=n_dims)

#             model.sig['in'][ens] = nb.Signal(
#                 np.zeros((len(group), 1)), name="%s.signal" % ens.label)

#             A = np.array([op.A.value for op in group])
#             model.sig['encoders'][ens] = nb.Signal(
#                 A, name="%s.scaled_encoders" % ens.label)

#             model.operators.append(MultiDotInc(
#                 model.sig['encoders'][ens],
#                 model.sig['in'][ens],
#                 base[group[0].Y.offset:group[-1].Y.offset+n].reshape((len(group), n)),
#                 tag="%s encoding" % ens.label))

#             for i, op in enumerate(group):
#                 op.X.set(model.sig['in'][ens][i])
#                 model.operators.remove(op)

# ModelOptimizer.register_optimizer(optimize_encoders)
