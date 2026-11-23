"""Piecewise linear interpolation and its convergence."""

import numpy as np


def linear_interpolation(f, a, b, n, oversampling=100):
    """Interpolate f linearly between n + 1 equally spaced nodes on [a, b].

    Returns the nodes, the fine evaluation grid and the interpolated values on it.
    """
    nodes = np.linspace(a, b, n + 1)
    grid = np.linspace(a, b, oversampling * n + 1)
    return nodes, grid, np.interp(grid, nodes, f(nodes))


def max_interpolation_error(f, a, b, ns, oversampling=100):
    """Maximum absolute interpolation error for each number of sub-intervals in ns."""
    errors = []
    for n in ns:
        _, grid, values = linear_interpolation(f, a, b, n, oversampling)
        errors.append(np.max(np.abs(f(grid) - values)))
    return np.array(errors)
