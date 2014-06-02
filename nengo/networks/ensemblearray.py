import numpy as np

import nengo
import nengo.builder as nb
from nengo.utils.compat import range
from nengo.utils.network import with_self


class EnsembleArray(nengo.Network):

    def __init__(self, n_neurons, n_ensembles, ens_dimensions=1, label=None,
                 **ens_kwargs):
        if "dimensions" in ens_kwargs:
            raise TypeError(
                "'dimensions' is not a valid argument to EnsembleArray. "
                "To set the number of ensembles, use 'n_ensembles'. To set "
                "the number of dimensions per ensemble, use 'ens_dimensions'.")

        self.config[nengo.Ensemble].update(ens_kwargs)

        label_prefix = "" if label is None else label + "_"

        self.n_ensembles = n_ensembles
        self.dimensions_per_ensemble = ens_dimensions
        transform = np.eye(self.dimensions)

        self.input = nengo.Node(size_in=self.dimensions, label="input")

        self.ea_ensembles = []
        for i in range(n_ensembles):
            e = nengo.Ensemble(
                n_neurons, self.dimensions_per_ensemble,
                label=label_prefix + str(i))
            trans = transform[i * self.dimensions_per_ensemble:
                              (i + 1) * self.dimensions_per_ensemble, :]
            nengo.Connection(self.input, e, transform=trans, synapse=None)
            self.ea_ensembles.append(e)

        self.add_output('output', function=None)

    @with_self
    def add_output(self, name, function, synapse=None, **conn_kwargs):
        if function is None:
            function_d = self.dimensions_per_ensemble
        else:
            func_output = function(np.zeros(self.dimensions_per_ensemble))
            function_d = np.asarray(func_output).size

        dim = self.n_ensembles * function_d
        output = nengo.Node(output=None, size_in=dim, label=name)
        setattr(self, name, output)

        for i, e in enumerate(self.ea_ensembles):
            nengo.Connection(
                e, output[i*function_d:(i+1)*function_d], function=function,
                synapse=synapse, **conn_kwargs)
        return output

    @property
    def dimensions(self):
        return self.n_ensembles * self.dimensions_per_ensemble


def build_ensemblearray(array, model, config):

    n_ensembles = array.n_ensembles
    n_neurons = array.ea_ensembles

    # Create random number generator
    # seed = model.next_seed() if array.seed is None else array.seed
    # rng = np.random.RandomState(seed)

    # Generate eval points
    if ens.eval_points is None or is_integer(ens.eval_points):
        eval_points = pick_eval_points(
            ens=ens, n_points=ens.eval_points, rng=rng)
    else:
        eval_points = npext.array(
            ens.eval_points, dtype=np.float64, min_dims=2)

    # Set up signal
    model.sig[ens]['in'] = Signal(np.zeros(ens.dimensions),
                                  name="%s.signal" % ens.label)
    model.add_op(Reset(model.sig[ens]['in']))

    # Set up encoders
    if isinstance(ens.neuron_type, nengo.neurons.Direct):
        encoders = np.identity(ens.dimensions)
    elif ens.encoders is None:
        sphere = dists.UniformHypersphere(ens.dimensions, surface=True)
        encoders = sphere.sample(ens.n_neurons, rng=rng)
    else:
        encoders = np.array(ens.encoders, dtype=np.float64)
        enc_shape = (ens.n_neurons, ens.dimensions)
        if encoders.shape != enc_shape:
            raise ShapeMismatch(
                "Encoder shape is %s. Should be (n_neurons, dimensions); "
                "in this case %s." % (encoders.shape, enc_shape))
        encoders /= npext.norm(encoders, axis=1, keepdims=True)

    # Determine max_rates and intercepts
    if isinstance(ens.max_rates, dists.Distribution):
        max_rates = ens.max_rates.sample(ens.n_neurons, rng=rng)
    else:
        max_rates = np.array(ens.max_rates)
    if isinstance(ens.intercepts, dists.Distribution):
        intercepts = ens.intercepts.sample(ens.n_neurons, rng=rng)
    else:
        intercepts = np.array(ens.intercepts)

    # Build the neurons
    gain, bias = ens.neuron_type.gain_bias(max_rates, intercepts)
    if isinstance(ens.neuron_type, nengo.neurons.Direct):
        model.sig[ens]['neuron_in'] = Signal(
            np.zeros(ens.dimensions), name='%s.neuron_in' % ens.label)
        model.sig[ens]['neuron_out'] = model.sig[ens]['neuron_in']
        model.add_op(Reset(model.sig[ens]['neuron_in']))
    else:
        model.sig[ens]['neuron_in'] = Signal(
            np.zeros(ens.n_neurons), name="%s.neuron_in" % ens.label)
        model.sig[ens]['neuron_out'] = Signal(
            np.zeros(ens.n_neurons), name="%s.neuron_out" % ens.label)
        model.add_op(Copy(src=Signal(bias, name="%s.bias" % ens.label),
                          dst=model.sig[ens]['neuron_in']))
        # This adds the neuron's operator and sets other signals
        Builder.build(ens.neuron_type, ens, model=model, config=config)

    # Scale the encoders
    if isinstance(ens.neuron_type, nengo.neurons.Direct):
        scaled_encoders = encoders
    else:
        scaled_encoders = encoders * (gain / ens.radius)[:, np.newaxis]

    model.sig[ens]['encoders'] = Signal(
        scaled_encoders, name="%s.scaled_encoders" % ens.label)

    # Create output signal, using built Neurons
    model.add_op(DotInc(
        model.sig[ens]['encoders'],
        model.sig[ens]['in'],
        model.sig[ens]['neuron_in'],
        tag="%s encoding" % ens.label))

    # Output is neural output
    model.sig[ens]['out'] = model.sig[ens]['neuron_out']

    model.params[ens] = BuiltEnsemble(eval_points=eval_points,
                                      encoders=encoders,
                                      intercepts=intercepts,
                                      max_rates=max_rates,
                                      scaled_encoders=scaled_encoders,
                                      gain=gain,
                                      bias=bias)


nb.Builder.register_builder(build_ensemblearray, nengo.networks.EnsembleArray)
