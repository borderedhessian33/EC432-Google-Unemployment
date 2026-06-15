"""
EC 432 Project — Real-time (rolling) selection nowcast path (results figure)
=============================================================================
Plots the look-ahead-free rolling-selected nowcast against the realized
unemployment rate and the AR(1) benchmark over the test window 2019–2025.
Vertical dotted lines mark the annual re-selection points; the selector adopts
VPC3+VPC4 (no COVID) in every year, so the path is that model's real-time nowcast.

Reads the locked output rolling_forecasts.csv (from 14_realtime_selection.py).
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
TEST_START = pd.Timestamp("2019-01-01")

df = pd.read_csv(HERE / "rolling_forecasts.csv", index_col=0, parse_dates=True)
df = df[df.index >= TEST_START]
actual = df["actual"]


def rmse(f):
    m = actual.notna() & f.notna()
    return float(np.sqrt(np.mean((actual[m] - f[m]) ** 2)))


fig, ax = plt.subplots(figsize=(10, 5))
ax.axvspan(COVID_START, COVID_END, color="gold", alpha=0.12, zorder=0, label="COVID window")
for yr in range(2020, 2026):                                   # annual re-selection points
    ax.axvline(pd.Timestamp(f"{yr}-01-01"), color="gray", lw=0.6, ls=":", alpha=0.5, zorder=1)

ax.plot(actual.index, actual.values, color="black", lw=2.4, zorder=5,
        label="Unemployment (actual)")
ax.plot(df.index, df["rolling"].values, color="#00467F", lw=1.9, zorder=4,
        label=f"Rolling-selected  (RMSE {rmse(df['rolling']):.3f})")
ax.plot(df.index, df["AR1"].values, color="#8C8C8C", ls="--", lw=1.3, zorder=3,
        label=f"AR(1)  (RMSE {rmse(df['AR1']):.3f})")

ax.set_ylabel("Unemployment rate (%)")
ax.legend(frameon=False, ncol=2, fontsize=9, loc="upper right")
ax.spines[["top", "right"]].set_visible(False)
ax.grid(axis="y", ls="--", alpha=0.3)
ax.margins(x=0.01)
plt.tight_layout()
plt.show()
