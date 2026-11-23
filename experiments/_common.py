"""Shared helpers for the experiment scripts."""

import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # save figures to files without opening windows
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "figures"
FIGURES.mkdir(exist_ok=True)
sys.path.insert(0, str(ROOT))


def timed(func, *args, repeats=3):
    """Best wall-clock time of func(*args) over several runs, in seconds."""
    best = float("inf")
    for _ in range(repeats):
        start = time.perf_counter()
        func(*args)
        best = min(best, time.perf_counter() - start)
    return best


def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIGURES / name, dpi=150)
    plt.close(fig)
    print(f"Saved figures/{name}")
