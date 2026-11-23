"""Solving Ax = b by QR: speed of three triangular solvers, and accuracy on ill-conditioned matrices."""

import numpy as np
from _common import plt, save, timed
from scipy.linalg import hilbert, solve_triangular

from numerical_methods.linear_systems import back_substitution, relative_error, solve_qr

rng = np.random.default_rng(0)

# --- Speed of the triangular solve -------------------------------------------
sizes = [100, 500, 1000, 2000, 4000]
solvers = {
    "Manual back substitution": back_substitution,
    "np.linalg.solve (general)": np.linalg.solve,
    "solve_triangular (specialised)": solve_triangular,
}
times = {name: [] for name in solvers}
for n in sizes:
    R = np.triu(rng.random((n, n))) + n * np.eye(n)  # well-conditioned upper triangular matrix
    c = rng.random(n)
    for name, solver in solvers.items():
        times[name].append(timed(solver, R, c))

fig, ax = plt.subplots(figsize=(8, 5))
for (name, values), style in zip(times.items(), ["r-o", "g-s", "b-d"]):
    ax.plot(sizes, values, style, label=name)
ax.set(yscale="log", xlabel="Matrix size n", ylabel="Execution time (s)",
       title="Solving an upper triangular system $Rx = c$")
ax.grid(True, which="both", alpha=0.3)
ax.legend()
save(fig, "linear_systems_timing.png")
for name, values in times.items():
    print(f"{name}: {values[-1] * 1e3:.1f} ms for n = {sizes[-1]}")

# --- Accuracy on the ill-conditioned Hilbert matrix ---------------------------
sizes = range(2, 30)
errors = {m: [] for m in ["manual", "general", "triangular"]}
conditioning = []
for n in sizes:
    A = hilbert(n)
    x_exact = np.ones(n)
    b = A @ x_exact
    conditioning.append(np.linalg.cond(A))
    for m in errors:
        errors[m].append(relative_error(solve_qr(A, b, m), x_exact))

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(sizes, errors["manual"], "r-o", lw=3, alpha=0.3, label="Manual")
ax.plot(sizes, errors["triangular"], "b-s", alpha=0.6, label="solve_triangular")
ax.plot(sizes, errors["general"], "g--", label="np.linalg.solve")
ax.plot(sizes, np.array(conditioning) * np.finfo(float).eps, "k:", label=r"cond$(A) \times \varepsilon_{machine}$")
ax.set(yscale="log", xlabel="Matrix size n", ylabel="Relative error",
       title="Numerical stability on the Hilbert matrix")
ax.grid(True, which="both", alpha=0.3)
ax.legend()
save(fig, "linear_systems_hilbert.png")
print(f"Hilbert n = {sizes[-1]}: cond = {conditioning[-1]:.1e}, relative error = {errors['manual'][-1]:.1e}")
