"""Numerical integration: composite midpoint and trapezoidal rules.

Each rule is provided in two versions:
- a loop version, written step by step in pure Python;
- a vectorised version, where NumPy evaluates the whole array at once.
"""

import numpy as np


def midpoint_loop(f, a, b, n):
    """Composite midpoint rule on [a, b] with n sub-intervals (Python loop)."""
    h = (b - a) / n
    x = np.linspace(a, b, n + 1)
    integral = 0.0
    for i in range(n):
        integral += f((x[i] + x[i + 1]) / 2) * h
    return integral


def trapezoid_loop(f, a, b, n):
    """Composite trapezoidal rule on [a, b] with n sub-intervals (Python loop)."""
    h = (b - a) / n
    x = np.linspace(a, b, n + 1)
    integral = 0.0
    for i in range(n):
        integral += h * (f(x[i]) + f(x[i + 1])) / 2
    return integral


def midpoint_vectorized(f, a, b, n):
    """Composite midpoint rule, vectorised: all midpoints are evaluated at once."""
    h = (b - a) / n
    x = np.linspace(a, b, n + 1)
    midpoints = (x[:-1] + x[1:]) / 2
    return h * np.sum(f(midpoints))


def trapezoid_vectorized(f, a, b, n):
    """Composite trapezoidal rule, vectorised."""
    h = (b - a) / n
    y = f(np.linspace(a, b, n + 1))
    return h * ((y[0] + y[-1]) / 2 + np.sum(y[1:-1]))
