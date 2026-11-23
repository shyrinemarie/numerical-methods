"""Eigenvalues with the QR algorithm, using a hand-written Gram-Schmidt QR decomposition."""

import numpy as np


def gram_schmidt_qr(A, tol=1e-10):
    """QR decomposition by classical Gram-Schmidt orthogonalisation of the columns of A."""
    A = A.astype(float)
    n, m = A.shape
    Q = np.zeros((n, m))
    R = np.zeros((m, m))
    for k in range(m):
        v = A[:, k].copy()
        for j in range(k):
            R[j, k] = Q[:, j] @ A[:, k]
            v -= R[j, k] * Q[:, j]
        R[k, k] = np.linalg.norm(v)
        # Avoid dividing by zero when a column is (numerically) dependent on the previous ones
        Q[:, k] = v / R[k, k] if R[k, k] > tol else 0.0
    return Q, R


def qr_algorithm(A, max_iter=100):
    """Unshifted QR algorithm: A_{k+1} = R_k Q_k, whose diagonal tends to the eigenvalues.

    Returns the eigenvalue estimates and the history of the diagonal at each iteration.
    """
    A_k = A.astype(float).copy()
    history = []
    for _ in range(max_iter):
        Q, R = gram_schmidt_qr(A_k)
        A_k = R @ Q
        history.append(np.diag(A_k).copy())
    return np.diag(A_k), np.array(history)


def iterations_to_tolerance(A, tol=1e-5, max_iter=5000):
    """Iterations needed for the dominant eigenvalue estimate to reach an absolute error below tol.

    Returns the number of iterations and the error history.
    """
    target = np.max(np.abs(np.linalg.eigvals(A)))
    A_k = A.astype(float).copy()
    errors = []
    while not errors or (errors[-1] > tol and len(errors) < max_iter):
        Q, R = gram_schmidt_qr(A_k)
        A_k = R @ Q
        errors.append(abs(np.max(np.abs(np.diag(A_k))) - target))
    return len(errors), np.array(errors)
