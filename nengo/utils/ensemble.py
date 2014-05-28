from __future__ import absolute_import
import numpy as np
import nengo.utils.numpy as npext


def tuning_curves(ens, sim, inputs=None, apply_encoders=False):
    """Calculates the tuning curves of an ensemble.

    Parameters
    ----------
    ens : nengo.Ensemble
        Ensemble to calculate the tuning curves of.
    sim : nengo.Simulator
        Simulator providing information about the build ensemble. (An unbuild
        ensemble does not have tuning curves assigned to it.)
    inputs : ndarray, optional
        Input points at which the tuning curves will be evaluated. Will be
        automatically chosen if ``None``.
    apply_encoders : boolean, optional
        If ``True``, the (unscaled, i.e. the radius will be ignored) encoders
        will be applied to the inputs. Otherwise it is assumed that the input
        is already encoded.

    Returns
    -------
    inputs : ndarray
        The passed or auto-generated `inputs`.
    activities : ndarray
        The activities of the individual neurons given the `inputs`.
    """

    if inputs is None:
        inputs = np.array(sim.data[ens].eval_points) if apply_encoders \
            else np.linspace(-1.0, 1.0)
        # Sort inputs for easier plotting
        inputs = inputs[np.lexsort(np.atleast_2d(inputs).T[::-1]), ...]

    x = np.dot(inputs, sim.data[ens].encoders.T) if apply_encoders else inputs

    activities = ens.neuron_type.rates(
        x, sim.data[ens].gain, sim.data[ens].bias)
    return inputs, activities


def _similarity(encoders, index, rows, cols=1):
    """Helper function to compute similarity for one encoder.

    Parameters
    ----------

    encoders: ndarray
        The encoders.
    index: int
        The encoder to compute for.
    rows: int
        The width of the 2d grid.
    cols: int
        The height of the 2d grid.
    """
    i = index % cols   # find the 2d location of the indexth element
    j = index // cols

    sim = 0  # total of dot products
    count = 0  # number of neighbours
    if i > 0:  # if we're not at the left edge, do the WEST comparison
        sim += np.dot(encoders[j * cols + i], encoders[j * cols + i - 1])
        count += 1
    if i < cols - 1:  # if we're not at the right edge, do EAST
        sim += np.dot(encoders[j * cols + i], encoders[j * cols + i + 1])
        count += 1
    if j > 0:  # if we're not at the top edge, do NORTH
        sim += np.dot(encoders[j * cols + i], encoders[(j - 1) * cols + i])
        count += 1
    if j < rows - 1:  # if we're not at the bottom edge, do SOUTH
        sim += np.dot(encoders[j * cols + i], encoders[(j + 1) * cols + i])
        count += 1
    return sim / count


def sorted_neurons(ensemble, sim, iterations=100, seed=None):
    '''Sort neurons in an ensemble by encoder and intercept.

    Parameters
    ----------
    ensemble: nengo.Ensemble
        The population of neurons to be sorted.
        The ensemble must have its encoders specified.

    iterations: int
        The number of times to iterate during the sort.

    seed: float
        A random number seed.

    Returns
    -------
    indices: ndarray
        An array with sorted indices into the neurons in the ensemble

    Examples
    --------

    You can use this to generate an array of sorted indices for plotting. This
    can be done after collecting the data. E.g.

    >>>indices = sorted_neurons(simulator, 'My neurons')
    >>>plt.figure()
    >>>rasterplot(sim.data['My neurons.spikes'][:,indices])

    Algorithm
    ---------

    The algorithm is for each encoder in the initial set, randomly
    pick another encoder and check to see if swapping those two
    encoders would reduce the average difference between the
    encoders and their neighbours.  Difference is measured as the
    dot product.  Each encoder has four neighbours (N, S, E, W),
    except for the ones on the edges which have fewer (no wrapping).
    This algorithm is repeated `iterations` times, so a total of
    `iterations*N` swaps are considered.
    '''

    # Normalize all the encoders
    encoders = np.array(sim.data[ensemble].encoders)
    encoders /= npext.norm(encoders, axis=1, keepdims=True)

    # Make an array with the starting order of the neurons
    N = encoders.shape[0]
    indices = np.arange(N)
    rng = np.random.RandomState(seed)

    for k in range(iterations):
        target = rng.randint(0, N, N)  # pick random swap targets
        for i in range(N):
            j = target[i]
            if i != j:  # if not swapping with yourself
                # compute similarity score how we are (unswapped)
                sim1 = (_similarity(encoders, i, N)
                        + _similarity(encoders, j, N))
                # swap the encoder
                encoders[[i, j], :] = encoders[[j, i], :]
                indices[[i, j]] = indices[[j, i]]
                # compute similarity score how we are (swapped)
                sim2 = (_similarity(encoders, i, N)
                        + _similarity(encoders, j, N))

                # if we were better unswapped
                if sim1 > sim2:
                    # swap them back
                    encoders[[i, j], :] = encoders[[j, i], :]
                    indices[[i, j]] = indices[[j, i]]

    return indices
