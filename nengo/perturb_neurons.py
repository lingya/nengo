"""A set of functions suitable for passing to `PerturbLIF()` in 
`nengo.neurons`. 

Signatures
----------

f : f(x_i)
    A function that perturbs a scalar voltage, x_i. The function signature 
    at call is `f(x)`, i.e there is one positional argument. 


Note
----
Before use these functions will be 'vectorized' using
`np.frompyfunc`.

To change keyword parameters use partial() prior to passing to PerturbLIF().
For example, to set the SD to 1 on white:

>>> from functools import partial
>>> plif = PerturbLIF(perturb=partial(white, scale=1))
"""
import numpy as np
from itertools import cycle


def null(x):
    """Always returns 0"""
    return 0.0

def randomize(x, scale=0.3):
    """Always returns a random normal value, ignoring x"""
    return np.random.normal(0, scale, 1)[0]
    
def white(x, scale=0.3):
    """Bias x with white noise (mean = 0)"""
    return x + np.random.normal(0, scale, 1)[0]

def exp(x, scale=3):
    """Bias x with Exponential noise"""
    return x + np.random.exponential(scale)[0]
    
# do 0.5, 2.5 and 5
def gamma(x, shape=2.5, scale=0.3):
    """Bias x with Gamma noise"""
    return x + np.random.gamma(shape, scale, 1)[0]
    
def silent(x, p=0.1):
    """Reset x to 0 with probability p"""
    thresh = np.random.uniform(0,1,1)
    if thresh < p:
        return 0.0
    else:
        return x

class Oscillator(object):
    """Perturb by inducing an oscillation. 
    
    Parameters
    ----------
    N : int
        Number of neurons to perturb at each time-step
    scale : float (default 1)
        Amplitude of the oscillation
    f : float (default 10)
        The oscillation frequency
    rate : float (default 50)
        The sampling rate
    """
    
    def __init__(self, N, p=0.5, scale=0.3, f=1, rate=50):
        self.wave = cycle(scale * np.sin(f * np.linspace(-np.pi, np.pi, rate)))
        self.w = self.wave.next()

        self.N = int(N)
        if self.N < 0:
            raise ValueError("N must be greater than 1")
        
        self.count = 0
        
        # Neuron n from N will have a oscillation?
        self.has_oscillation = np.random.binomial(1, p, N).astype(np.bool)
        
    def __call__(self, x):
        if self.count == self.N:
            self.count = 0
            self.w = self.wave.next()
        
        if self.has_oscillation[self.count]:
            x = self.w + x

        self.count += 1

        return x
