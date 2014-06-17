
# coding: utf-8

# # Nengo Example: Squaring the Input
# This demo shows you how to construct a network that squares the value encoded in a first population in the output of a second population. 
# ## Step 1: Create the Model
# The model is comprised of an input ensemble ('A') and an output ensemble ('B'), from which the squared value of the input signal can be decoded.

# Create the model objectl
from functools import partial
import numpy as np
import nengo
import nengo.perturb_neurons as pb

N = 1000 ## the number of neurons

model = nengo.Network(label='Squaring')

with model:
    osc = nengo.perturb_neurons.Oscillator(N, p=0.3, scale=1, f=10, rate=50)
    # Create two ensembles of 100 leaky-integrate-and-fire neurons
    A = nengo.Ensemble(
        N, dimensions=1, 
        neuron_type=nengo.neurons.PerturbLIF(
            perturb=osc
            )
        )
    # A = nengo.Ensemble(1000, dimensions=1)
    B = nengo.Ensemble(N, dimensions=1)


# ##Step 2: Provide Input to the Model
# A single input signal (a sine wave) will be used to drive the neural activity in ensemble A.
import numpy as np
with model:
    # Create an input node that represents a sine wave
    sin = nengo.Node(np.sin)
    
    # Connect the input node to ensemble A
    nengo.Connection(sin, A)
    
    # Define the squaring function
    def square(x):
        return x[0] * x[0]
    
    # Connection ensemble A to ensemble B
    nengo.Connection(A, B, function=square)

# ##Step 3: Probe the Output
# Let's collect output data from each ensemble and output.
with model:
    sin_probe = nengo.Probe(sin)
    A_probe = nengo.Probe(A, synapse=0.01)
    B_probe = nengo.Probe(B, synapse=0.01)


# ## Step 4: Run the Model
# Create the simulator
sim = nengo.Simulator(model)
# Run the simulator for 5 seconds
sim.run(5)

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

with PdfPages('test.pdf') as pdf:
    # Plot the input signal and decoded ensemble values
    plt.plot(sim.trange(), sim.data[A_probe],  label='Decoded Ensemble A')
    plt.plot(sim.trange(), sim.data[B_probe], label='Decoded Ensemble B')  
    plt.plot(sim.trange(), sim.data[sin_probe], label='Input Sine Wave', color='k', linewidth=2.0)
    plt.plot(sim.trange(), sim.data[sin_probe]**2, label='Input Sine Wave Squared', color='k', linewidth=2.0)
    plt.legend(loc='best')
    plt.ylim(-1.2, 1.2)
    pdf.savefig()


# The plotted ouput of ensemble B should show the decoded squared value of the input sine wave.  
