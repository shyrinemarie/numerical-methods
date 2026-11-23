"""Midpoint vs trapezoidal rule: effect of vectorisation on speed, and convergence."""

import numpy as np
from _common import plt, save, timed

from numerical_methods.quadrature import (
    midpoint_loop,
    midpoint_vectorized,
    trapezoid_loop,
    trapezoid_vectorized,
)


def square(x):
    return x**2


# --- Execution time: loop vs vectorised -------------------------------------
ns = [100, 1_000, 10_000, 100_000, 1_000_000]
times = {name: [] for name in ["midpoint loop", "trapezoid loop", "midpoint vectorised", "trapezoid vectorised"]}
for n in ns:
    times["midpoint loop"].append(timed(midpoint_loop, square, 0, 1, n, repeats=1))
    times["trapezoid loop"].append(timed(trapezoid_loop, square, 0, 1, n, repeats=1))
    times["midpoint vectorised"].append(timed(midpoint_vectorized, square, 0, 1, n))
    times["trapezoid vectorised"].append(timed(trapezoid_vectorized, square, 0, 1, n))

for rule in ["midpoint", "trapezoid"]:
    speedup = times[f"{rule} loop"][-1] / times[f"{rule} vectorised"][-1]
    print(f"{rule}: vectorised version is {speedup:.0f}x faster for n = {ns[-1]:,}")

fig, ax = plt.subplots(figsize=(8, 5))
styles = {"midpoint loop": "r-o", "trapezoid loop": "g-o", "midpoint vectorised": "r--s", "trapezoid vectorised": "g--s"}
for name, values in times.items():
    ax.plot(ns, values, styles[name], label=name)
ax.set(xscale="log", yscale="log", xlabel="Number of sub-intervals n", ylabel="Execution time (s)",
       title="Impact of vectorisation on execution time")
ax.grid(True, which="both", alpha=0.3)
ax.legend()
save(fig, "quadrature_timing.png")

# --- Convergence -------------------------------------------------------------
tests = [
    ("$x^2$ on [-1, 1]", square, -1, 1, 2 / 3),
    ("Runge $1/(1+25x^2)$ on [-1, 1]", lambda x: 1 / (1 + 25 * x**2), -1, 1, 0.4 * np.arctan(5)),
]
ns = np.array([4, 10, 50, 100, 500, 1_000, 5_000])
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
for ax, (name, f, a, b, exact) in zip(axes, tests):
    err_mid = [abs(midpoint_vectorized(f, a, b, n) - exact) for n in ns]
    err_trap = [abs(trapezoid_vectorized(f, a, b, n) - exact) for n in ns]
    ax.plot(ns, err_mid, "r-o", label="Midpoint")
    ax.plot(ns, err_trap, "g-s", label="Trapezoid")
    ax.plot(ns, err_trap[0] * (ns[0] / ns) ** 2, "k--", alpha=0.5, label="Slope $O(1/n^2)$")
    ax.set(xscale="log", yscale="log", xlabel="Number of sub-intervals n", ylabel="Absolute error", title=name)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
save(fig, "quadrature_convergence.png")
