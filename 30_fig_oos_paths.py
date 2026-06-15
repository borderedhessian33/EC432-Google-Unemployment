"""
EC 432 Project — Out-of-sample nowcast paths (results figure)
==============================================================
Plots the recursive pseudo-out-of-sample nowcasts of the three most accurate
candidates against the AR(1) and No-Trends-SS+COVID benchmarks and the realized
unemployment rate, over the evaluation window 2018–2025.

Top-3 candidates (by full-window RMSE, from 11_candidate_grid.py):
  1. VPC3+VPC4 (no COVID)     0.533
  2. VPC3 + COVID             0.536
  3. VPC1+VPC3 + COVID        0.544

Reads only the locked outputs (candidate_forecasts.csv, ar_benchmarks.csv).
Displays the figure (no title; caption belongs in the LaTeX float).
"""

import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
HERE = Path(__file__).parent
COVID_START, COVID_END = pd.Timestamp("2020-04-01"), pd.Timestamp("2021-12-01")

cand  = pd.read_csv(HERE / "candidate_forecasts.csv", index_col=0, parse_dates=True)
bench = pd.read_csv(HERE / "ar_benchmarks.csv", index_col=0, parse_dates=True)
actual = bench["actual"]


def rmse(f):
    f = f.reindex(actual.index); m = actual.notna() & f.notna()
    return float(np.sqrt(np.mean((actual[m] - f[m]) ** 2)))


# (label, series, colour, linestyle, linewidth, alpha)
candidates = [
    ("VPC3+VPC4",        cand["VPC3+VPC4_nocov"], "#00467F", "-", 1.8, 0.95),
    ("VPC3 $+$COVID",    cand["VPC3_cov"],        "#2CA02C", "-", 1.5, 0.90),
    ("VPC1+VPC3 $+$COVID", cand["VPC1+VPC3_cov"], "#9467BD", "-", 1.5, 0.90),
]
benchmarks = [
    ("AR(1)",            bench["AR1"],          "#8C8C8C", "--", 1.3, 0.85),
    ("No-Trends $+$COVID", bench["NoTrends_cov"], "#404040", ":",  1.4, 0.85),
]

fig, ax = plt.subplots(figsize=(10, 5))
ax.axvspan(COVID_START, COVID_END, color="gold", alpha=0.12, zorder=0, label="COVID window")
ax.plot(actual.index, actual.values, color="black", lw=2.4, zorder=5,
        label="Unemployment (actual)")
for lbl, s, c, ls, lw, a in candidates + benchmarks:
    ax.plot(actual.index, s.reindex(actual.index).values, color=c, ls=ls, lw=lw,
            alpha=a, zorder=4, label=f"{lbl}  (RMSE {rmse(s):.3f})")

ax.set_ylabel("Unemployment rate (\\%)" if plt.rcParams["text.usetex"] else "Unemployment rate (%)")
ax.legend(frameon=False, ncol=2, fontsize=9, loc="upper right")
ax.spines[["top", "right"]].set_visible(False)
ax.grid(axis="y", ls="--", alpha=0.3)
ax.margins(x=0.01)
plt.tight_layout()
plt.show()
