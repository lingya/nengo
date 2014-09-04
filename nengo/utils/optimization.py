"""Utilities for optimizing neuron populations and networks."""

from __future__ import absolute_import

import numpy as np
from scipy.special import beta, betainc

from nengo.utils.distributions import SqrtBeta


def sp_subvector_optimal_radius(
        sp_dimensions, sp_subdimensions, ens_dimensions, eval_points):
    """Determines the optimal radius for ensembles when splitting up a semantic
    pointer (unit vector) into subvectors.

    Requires Scipy.

    Parameters
    ----------
    dimensions : int
        Dimensionality of the complete semantic pointer/unit vector.
    subdimensions : int
        Dimensionality of the subvectors represented by the ensembles.
    eval_points : int
        Number of evaluation points used for the representing ensembles.

    Returns
    -------
    float
        Optimal radius for the representing ensembles.
    """
    import scipy.optimize
    res = scipy.optimize.minimize(
        lambda x: sp_subvector_error(
            sp_dimensions, sp_subdimensions, ens_dimensions, eval_points, x),
        0.5)
    return np.asscalar(res.x)


def sp_subvector_error(
        sp_dimensions, sp_subdimensions, ens_dimensions, eval_points, radius):
    """Estimate of representational error of a subvector of a semantic pointer
    (unit vector).

    Requires Scipy.

    Paramaters
    ----------
    dimensions : int
        Dimensionality of the complete semantic pointer/unit vector.
    subdimensions : int
        Dimensionality of the subvector represented by some ensemble.
    eval_points : int
        Number of evaluations points used for the representing ensemble.
    radius : float or ndarray
        Radius of the representing ensemble.

    Returns
    -------
    Error estimates for representing a subvector with `subdimensions`
    dimensions of a `dimensions` dimensional unit vector with an ensemble
    initialized with `eval_points` evaluation points and a radius of `radius`.
    """
    dist = SqrtBeta(sp_dimensions - sp_subdimensions, sp_subdimensions)
    return (dist.cdf(radius) * _sp_subvector_error_in_range(
        ens_dimensions, eval_points, radius) +
        (1.0 - dist.cdf(radius)) * _sp_subvector_error_out_of_range(
            sp_dimensions, sp_subdimensions, eval_points, radius))


def _sp_subvector_error_in_range(ens_dimensions, eval_points, radius):
    return (radius / max(
        1.0, (eval_points) ** (1.0 / ens_dimensions) - 1)) ** 2 / 3.0


def _sp_subvector_error_out_of_range(
        dimensions, subdimensions, eval_points, radius):
    dist = SqrtBeta(dimensions - subdimensions, subdimensions)
    sq_r = radius * radius

    normalization = 1.0 - dist.cdf(radius)
    b = (dimensions - subdimensions) / 2.0
    aligned_integral = beta(subdimensions / 2.0 + 1.0, b) * (1.0 - betainc(
        subdimensions / 2.0 + 1.0, b, sq_r))
    cross_integral = beta((subdimensions + 1) / 2.0, b) * (1.0 - betainc(
        (subdimensions + 1) / 2.0, b, sq_r))

    numerator = (sq_r * normalization + (
        aligned_integral - 2.0 * radius * cross_integral) / beta(
        subdimensions / 2.0, b))
    return numerator / normalization
