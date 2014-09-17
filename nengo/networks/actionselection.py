import warnings

import numpy as np

import nengo
from nengo.solvers import NnlsL2nz
from nengo.networks.ensemblearray import EnsembleArray
from nengo.utils.distributions import Choice, Uniform


# Affects all ensembles / connections in the BG
bg_config = nengo.Config(nengo.Ensemble, nengo.Connection)
bg_config[nengo.Ensemble].radius = 1.5
bg_config[nengo.Ensemble].encoders = Choice([[1]])
try:
    # Best, if we have SciPy
    bg_config[nengo.Connection].solver = NnlsL2nz()
except ImportError:
    # Don't warn here; warn when the BG is made
    pass

# Affects connections to AMPA receptors
ampa_config = nengo.Config(nengo.Connection)
ampa_config[nengo.Connection].synapse = 0.002

# Affects connections to GABA receptors
gaba_config = nengo.Config(nengo.Connection)
gaba_config[nengo.Connection].synapse = 0.008


# connection weights from (Gurney, Prescott, & Redgrave, 2001)
class Weights(object):
    mm = 1
    mp = 1
    me = 1
    mg = 1
    ws = 1
    wt = 1
    wm = 1
    wg = 1
    wp = 0.9
    we = 0.3
    e = 0.2
    ep = -0.25
    ee = -0.2
    eg = -0.2
    le = 0.2
    lg = 0.2

    @classmethod
    def str_func(cls, x):
        if x < cls.e:
            return 0
        return cls.mm * (x - cls.e)

    @classmethod
    def stn_func(cls, x):
        if x < cls.ep:
            return 0
        return cls.mp * (x - cls.ep)

    @classmethod
    def gpe_func(cls, x):
        if x < cls.ee:
            return 0
        return cls.me * (x - cls.ee)

    @classmethod
    def gpi_func(cls, x):
        if x < cls.eg:
            return 0
        return cls.mg * (x - cls.eg)


def BasalGanglia(dimensions, n_neurons_per_ensemble=100,
                 output_weight=-3, input_bias=0.0, net=None):
    """Winner takes all; outputs 0 at max dimension, negative elsewhere."""

    if net is None:
        net = nengo.Network("Basal Ganglia")

    # Warn if we can't use the better decoder solver.
    try:
        NnlsL2nz()
    except ImportError:
        warnings.warn("SciPy is not installed, so BasalGanglia will "
                      "use default decoder solver. Installing SciPy "
                      "may improve BasalGanglia performance.")

    ea_params = {'n_neurons': n_neurons_per_ensemble,
                 'n_ensembles': dimensions}

    with net, bg_config:
        net.strD1 = EnsembleArray(label="Striatal D1 neurons",
                                  intercepts=Uniform(Weights.e, 1),
                                  **ea_params)
        net.strD2 = EnsembleArray(label="Striatal D2 neurons",
                                  intercepts=Uniform(Weights.e, 1),
                                  **ea_params)
        net.stn = EnsembleArray(label="Subthalamic nucleus",
                                intercepts=Uniform(Weights.ep, 1),
                                **ea_params)
        net.gpi = EnsembleArray(label="Globus pallidus internus",
                                intercepts=Uniform(Weights.eg, 1),
                                **ea_params)
        net.gpe = EnsembleArray(label="Globus pallidus externus",
                                intercepts=Uniform(Weights.ee, 1),
                                **ea_params)

        net.input = nengo.Node(label="input", size_in=dimensions)
        net.output = nengo.Node(label="output", size_in=dimensions)

        # add bias input (BG performs best in the range 0.5--1.5)
        if abs(input_bias) > 0.0:
            net.bias_input = nengo.Node(np.ones(dimensions) * input_bias)
            nengo.Connection(net.bias_input, net.input)

        # spread the input to StrD1, StrD2, and STN
        nengo.Connection(net.input, net.strD1.input, synapse=None,
                         transform=Weights.ws * (1 + Weights.lg))
        nengo.Connection(net.input, net.strD2.input, synapse=None,
                         transform=Weights.ws * (1 - Weights.le))
        nengo.Connection(net.input, net.stn.input, synapse=None,
                         transform=Weights.wt)

        # connect the striatum to the GPi and GPe (inhibitory)
        strD1_output = net.strD1.add_output('func_str', Weights.str_func)
        strD2_output = net.strD2.add_output('func_str', Weights.str_func)
        with gaba_config:
            nengo.Connection(strD1_output, net.gpi.input,
                             transform=-np.eye(dimensions) * Weights.wm)
            nengo.Connection(strD2_output, net.gpe.input,
                             transform=-np.eye(dimensions) * Weights.wm)

        # connect the STN to GPi and GPe (broad and excitatory)
        tr = Weights.wp * np.ones((dimensions, dimensions))
        stn_output = net.stn.add_output('func_stn', Weights.stn_func)
        with ampa_config:
            nengo.Connection(stn_output, net.gpi.input, transform=tr)
            nengo.Connection(stn_output, net.gpe.input, transform=tr)

        # connect the GPe to GPi and STN (inhibitory)
        gpe_output = net.gpe.add_output('func_gpe', Weights.gpe_func)
        with gaba_config:
            nengo.Connection(gpe_output, net.gpi.input, transform=-Weights.we)
            nengo.Connection(gpe_output, net.stn.input, transform=-Weights.wg)

        # connect GPi to output (inhibitory)
        gpi_output = net.gpi.add_output('func_gpi', Weights.gpi_func)
        nengo.Connection(gpi_output, net.output, synapse=None,
                         transform=output_weight)

    return net


def Thalamus(dimensions, n_neurons_per_ensemble=50,
             mutual_inhib=1, threshold=0, net=None):
    """Inhibits non-selected actions.

    Converts basal ganglia output into a signal with
    (approximately) 1 for the selected action and 0 elsewhere.
    """

    if net is None:
        net = nengo.Network("Thalamus")

    with net:
        net.actions = EnsembleArray(n_neurons_per_ensemble, dimensions,
                                    intercepts=Uniform(threshold, 1),
                                    encoders=Choice([[1.0]]),
                                    label="actions")
        nengo.Connection(net.actions.output, net.actions.input,
                         transform=(np.eye(dimensions) - 1) * mutual_inhib)
        net.bias = nengo.Node([1])
        nengo.Connection(net.bias, net.actions.input,
                         transform=np.ones((dimensions, 1)))

    net.input = net.actions.input
    net.output = net.actions.output
    return net
