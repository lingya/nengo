import numpy as np

import nengo
from nengo.utils.network import with_self


class EnsembleArray(nengo.Network):

    def __init__(self, n_neurons, n_ensembles, ens_dimensions=1, label=None,
                 optimize=False, fast=False, **ens_kwargs):
        if "dimensions" in ens_kwargs:
            raise TypeError(
                "'dimensions' is not a valid argument to EnsembleArray. "
                "To set the number of ensembles, use 'n_ensembles'. To set "
                "the number of dimensions per ensemble, use 'ens_dimensions'.")

        self.config[nengo.Ensemble].update(ens_kwargs)

        label_prefix = "" if label is None else label + "_"

        self._n_ensembles = n_ensembles
        self._ens_dimensions = ens_dimensions
        self._optimize = optimize
        self.fast = fast

        self.input = nengo.Node(size_in=self.dimensions, label="input")

        if self.optimize:

            # make a single dummy ensemble to help in building later
            self.ens_template = nengo.Ensemble(
                n_neurons, ens_dimensions, **ens_kwargs)

            # make a node to represent the ensembles
            self.ensemble = nengo.Ensemble(
                n_ensembles * n_neurons, self.dimensions,
                label="%s.ensemble" % self.label)

            nengo.Connection(self.input, self.ensemble)

        else:
            transform = np.eye(self.dimensions)

            self._ea_ensembles = []
            for i in range(n_ensembles):
                e = nengo.Ensemble(
                    n_neurons, self.ens_dimensions,
                    label=label_prefix + str(i))
                trans = transform[i * self.ens_dimensions:
                                  (i + 1) * self.ens_dimensions, :]
                nengo.Connection(self.input, e, transform=trans, synapse=None)
                self._ea_ensembles.append(e)

        self._outputs = []
        self.add_output('output', function=None)

    @with_self
    def add_output(self, name, function, synapse=None, **conn_kwargs):
        if hasattr(self, name):
            raise ValueError("The name '%s' is already taken" % name)

        if function is None:
            function_d = self.ens_dimensions
        else:
            func_output = function(np.zeros(self.ens_dimensions))
            function_d = np.asarray(func_output).size

        dim = self.n_ensembles * function_d
        output = nengo.Node(output=None, size_in=dim, label=name)
        setattr(self, name, output)

        if self.optimize:
            conn = nengo.Connection(
                self.ensemble, output, function=function, synapse=synapse)

            node = nengo.Node(size_in=function_d)
            conn_template = nengo.Connection(
                self.ens_template, node, function=function, synapse=synapse,
                **conn_kwargs)

            self._outputs.append((name, conn, conn_template, node))
        else:
            for i, e in enumerate(self._ea_ensembles):
                nengo.Connection(
                    e, output[i*function_d:(i+1)*function_d], function=function,
                    synapse=synapse, **conn_kwargs)

        return output

    @property
    def dimensions(self):
        return self.n_ensembles * self.ens_dimensions

    @property
    def ens_dimensions(self):
        return self._ens_dimensions

    @property
    def n_ensembles(self):
        return self._n_ensembles

    @property
    def optimize(self):
        return self._optimize


import nengo.builder as nb
import nengo.utils.numpy as npext

def build_ensemblearray(array, model):  # noqa: C901

    from nengo.builder import *

    if not array.optimize:
        # build the network normally
        return build_network(array, model)

    # Create random number generator
    seed = model.next_seed() if array.seed is None else array.seed
    rng = np.random.RandomState(seed)

    # Define parameters
    config = array.config
    ens = array.ensemble
    ens_template = array.ens_template
    neuron_type = ens_template.neuron_type
    assert not isinstance(neuron_type, nengo.neurons.Direct)

    # --- Build input and output nodes
    for node in array.nodes:
        Builder.build(node, model=model, config=config)

    # --- Build main ensemble

    # create input
    model.sig[ens]['in'] = Signal(
        np.zeros((array.n_ensembles, array.ens_dimensions)),
        name="%s.in" % ens.label)
    # model.add_op(Reset(model.sig[ens]['in']))

    model.add_op(Copy(
            src=model.sig[array.input]['out'].reshape(
                array.n_ensembles, array.ens_dimensions),
            dst=model.sig[ens]['in']))

    # create neurons
    model.sig[ens]['neuron_in'] = Signal(
        np.zeros(ens.n_neurons), name="%s.neuron_in" % ens.label)
    model.sig[ens]['neuron_out'] = Signal(
        np.zeros(ens.n_neurons), name="%s.neuron_out" % ens.label)

    Builder.build(neuron_type, ens, model=model, config=config)


    models = []
    bias = []
    encoders = []
    for i in range(array.n_ensembles):
        seedi = rng.randint(2**30)
        modeli = Model(dt=model.dt, seed=seedi)
        models.append(modeli)

        build_ensemble(ens_template, modeli, config)
        params = modeli.params[ens_template]
        bias.append(params.bias)
        encoders.append(params.scaled_encoders)

        if array.fast:
            # all encoders and biases will be the same
            break

    if array.fast:
        bias = np.tile(bias[0], array.n_ensembles).reshape((ens.n_neurons,))
    else:
        bias = np.array(bias).reshape((total_neurons,))
    model.add_op(Copy(
            src=Signal(bias, name="%s.bias" % array.label),
            dst=model.sig[ens]['neuron_in']))

    # create encoders
    encoders = np.array(encoders[0] if array.fast else encoders)
    model.sig[ens]['encoders'] = Signal(
        encoders, name="%s.scaled_encoders" % ens.label)

    if array.fast:
        model.add_op(DotInc(
            model.sig[ens]['encoders'],
            model.sig[ens]['in'].T,
            model.sig[ens]['neuron_in'].reshape(
                array.n_ensembles, ens_template.n_neurons).T,
            tag="%s encoding" % ens.label))
    else:
        raise NotImplementedError()

    # --- Build output connections
    for name, conn, conn_template, node_template in array._outputs:
        assert not conn_template.modulatory
        assert not conn_template.learning_rule

        output = getattr(array, name)
        function = conn.function
        synapse = conn.synapse

        f_dimensions = output.size_in / array.n_ensembles

        decoders = []
        transforms = []
        for i in range(array.n_ensembles):
            modeli = models[i]
            Builder.build(node_template, model=modeli, config=config)
            Builder.build(conn_template, model=modeli, config=config)

            decoders.append(modeli.params[conn_template].decoders)
            transforms.append(modeli.params[conn_template].transform)

            if array.fast:
                # same decoders and transform for all connections
                break

        decoders = np.array(decoders[0] if array.fast else decoders)
        transforms = np.array(transforms[0] if array.fast else transforms)

        model.sig[conn]['decoders'] = Signal(
            decoders, name="%s.decoders" % conn.label)
        model.sig[conn]['transforms'] = Signal(
            transforms, name="%s.transforms" % conn.label)
        signal = Signal(
            np.zeros((array.n_ensembles, f_dimensions)), name=conn.label)

        # create decoders
        if array.fast:
            model.add_op(ProdUpdate(
                model.sig[conn]['decoders'],
                model.sig[ens]['neuron_out'].reshape(
                    array.n_ensembles, ens_template.n_neurons).T,
                model.sig['common'][0],
                signal.T,
                tag="%s decoding" % conn.label))
        else:
            raise NotImplementedError()

        # create synapse
        if synapse is not None:
            signal = filtered_signal(conn, signal, synapse, model, config)

        # create transform to output node
        if array.fast:
            model.add_op(DotInc(
                model.sig[conn]['transforms'],
                signal,
                model.sig[output]['in'].reshape(
                    array.n_ensembles, f_dimensions),
                tag="%s.transform" % conn.label))


nengo.builder.Builder.register_builder(build_ensemblearray, EnsembleArray)
