"""Piecewise linear interpolation: examples and O(1/N^2) convergence."""

import numpy as np
from _common import plt, save

from numerical_methods.interpolation import linear_interpolation, max_interpolation_error

functions = [
    ("Runge $1/(1+25x^2)$", lambda x: 1 / (1 + 25 * x**2), -1, 1),
    ("$\\sin(10x)$", lambda x: np.sin(10 * x), 0, np.pi),
    ("$x^2$", lambda x: x**2, 0, 3),
]

# --- Examples with few nodes -------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
for ax, (name, f, a, b) in zip(axes, functions):
    nodes, grid, values = linear_interpolation(f, a, b, n=8)
    ax.plot(grid, f(grid), "--", color="gray", label="Exact function")
    ax.plot(grid, values, color="purple", label="Linear interpolation")
    ax.scatter(nodes, f(nodes), color="indigo", zorder=5, label="Nodes (n = 8)")
    ax.set_title(name)
    ax.legend(fontsize=8)
save(fig, "interpolation_examples.png")

# --- Convergence -------------------------------------------------------------
ns = np.array([5, 10, 20, 50, 100, 200, 500, 1000, 2000])
fig, ax = plt.subplots(figsize=(8, 5))
for (name, f, a, b), style in zip(functions, ["r-s", "b-o", "g-^"]):
    errors = max_interpolation_error(f, a, b, ns)
    slope = np.polyfit(np.log(ns[3:]), np.log(errors[3:]), 1)[0]
    print(f"{name}: fitted convergence slope = {slope:.2f}")
    ax.plot(ns, errors, style, label=name)
ax.plot(ns, (ns[0] / ns) ** 2, "k--", alpha=0.5, label="Slope $O(1/N^2)$")
ax.set(xscale="log", yscale="log", xlabel="Number of sub-intervals N", ylabel="Maximum error",
       title="Convergence of piecewise linear interpolation")
ax.grid(True, which="both", alpha=0.3)
ax.legend()
save(fig, "interpolation_convergence.png")
