import collections
import logging
import warnings

import numpy as np

import nengo.builder as nb


class ModelOptimizer(object):

    optimizers = []

    @classmethod
    def register_optimizer(cls, optimizer_fn):
        cls.optimizers.append(optimizer_fn)

    @classmethod
    def optimize(cls, model, *args, **kwargs):

        for optimizer in optimizers:
            model = optimizer(model)

        return model

def optimize_neurons(model):

    neurons = {}
    for op in model.operators:
        if isinstance(op, nb.SimNeurons):
            if op.neurons not in neurons:
                neurons[op.neurons] = [op]
            else:
                neurons[op.neurons].append(op)
    pass

ModelOptimizer.register_optimizer(optimize_neurons)
