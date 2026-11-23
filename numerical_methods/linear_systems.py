"""Solving Ax = b with a QR decomposition.

Since A = QR with Q orthogonal and R upper triangular, Ax = b becomes
Rx = Q^T b, which is solved by back substitution.
"""

import numpy as np
from scipy.linalg import qr, solve_triangular


def back_substitution(R, c):
    """Solve the upper triangular system Rx = c, written by hand."""
    n = R.shape[0]
    x = np.zeros(n)
    for i in reversed(range(n)):
        x[i] = (c[i] - np.dot(R[i, i + 1:], x[i + 1:])) / R[i, i]
    return x


def solve_qr(A, b, method="manual"):
    """Solve Ax = b via QR, finishing with one of three triangular solvers.

    method:
        "manual"     -- hand-written back substitution
        "general"    -- np.linalg.solve, which does not exploit the triangular structure
        "triangular" -- scipy.linalg.solve_triangular, specialised for triangular systems
    """
    Q, R = qr(A)
    c = Q.T @ b
    if method == "manual":
        return back_substitution(R, c)
    if method == "general":
        return np.linalg.solve(R, c)
    if method == "triangular":
        return solve_triangular(R, c)
    raise ValueError(f"Unknown method: {method}")


def relative_error(x, x_exact):
    """Relative error ||x - x_exact|| / ||x_exact||."""
    return np.linalg.norm(x - x_exact) / np.linalg.norm(x_exact)
