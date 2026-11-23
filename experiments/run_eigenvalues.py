"""QR algorithm for eigenvalues: effect of the spectral gap on convergence speed."""

import numpy as np
from _common import plt, save

from numerical_methods.eigenvalues import iterations_to_tolerance, qr_algorithm

rng = np.random.default_rng(42)

# --- Sanity check against NumPy on a symmetric matrix -------------------------
M = rng.random((6, 6))
A = M @ M.T
estimates, _ = qr_algorithm(A, max_iter=300)
print("QR algorithm:", np.sort(estimates))
print("NumPy       :", np.sort(np.linalg.eigvalsh(A)))

# --- Spectral gap ------------------------------------------------------------
# Same eigenvectors, different eigenvalues: A = P D P^{-1}
P = rng.random((5, 5))
P_inv = np.linalg.inv(P)
cases = {
    "Well separated ($\\lambda_2/\\lambda_1 = 0.2$)": np.diag([10, 2, 0.5, 0.1, 0.01]),
    "Close eigenvalues ($\\lambda_2/\\lambda_1 = 0.99$)": np.diag([10, 9.9, 5, 2, 1]),
}
tol = 1e-5
fig, ax = plt.subplots(figsize=(8, 5))
for (name, D), style in zip(cases.items(), ["g-", "r-"]):
    n_iter, errors = iterations_to_tolerance(P @ D @ P_inv, tol=tol)
    print(f"{name}: {n_iter} iterations to reach {tol:g}")
    ax.plot(errors, style, label=f"{name}: {n_iter} iterations")
ax.axhline(tol, color="k", ls="--", label=f"Tolerance {tol:g}")
ax.set(yscale="log", xlabel="Iteration", ylabel="Error on the dominant eigenvalue",
       title="Impact of the spectral gap on the QR algorithm")
ax.grid(True, which="both", alpha=0.3)
ax.legend()
save(fig, "eigenvalues_spectral_gap.png")
