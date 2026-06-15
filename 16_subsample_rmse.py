"""
EC 432 Project — Regime analysis (subsample RMSE + rolling RMSE)
================================================================
Two results that locate *when* the search factors help:

  (1) Subsample RMSE over pre-COVID / COVID / post-COVID / full windows, for the
      headline model, a labour single, and the AR / no-Trends benchmarks.
      -> printed table + subsample_rmse.csv
  (2) A 12-month rolling RMSE of the headline vs AR(1), displayed as a figure.

Reads only the locked outputs (candidate_forecasts.csv, ar_benchmarks.csv).
"""

import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
HERE = Path(__file__).parent
COVID_START, COVID_END = pd.Timestamp("2020-04-01"), pd.Timestamp("2021-12-01")
ROLL = 12

cand  = pd.read_csv(HERE / "candidate_forecasts.csv", index_col=0, parse_dates=True)
bench = pd.read_csv(HERE / "ar_benchmarks.csv", index_col=0, parse_dates=True)
actual = bench["actual"]

models = {
    "VPC3+VPC4 (headline)": cand["VPC3+VPC4_nocov"],
    "VPC3 + COVID":         cand["VPC3_cov"],
    "AR(1)":                bench["AR1"],
    "AR(2)":                bench["AR2"],
    "No-Trends + COVID":    bench["NoTrends_cov"],
}
periods = {
    "Pre-COVID":  ("2018-01-01", "2020-03-01"),
    "COVID":      ("2020-04-01", "2021-12-01"),
    "Post-COVID": ("2022-01-01", "2025-12-01"),
    "Full":       ("2018-01-01", "2025-12-01"),
}


def rmse(f, lo, hi):
    f = f.reindex(actual.index)
    m = (actual.index >= lo) & (actual.index <= hi) & actual.notna() & f.notna()
    return float(np.sqrt(np.mean((actual[m] - f[m]) ** 2)))


# ── (1) Subsample table ──────────────────────────────────────────────────────────
tbl = pd.DataFrame({p: {m: rmse(f, lo, hi) for m, f in models.items()}
                    for p, (lo, hi) in periods.items()})
ns = {p: int(((actual.index >= lo) & (actual.index <= hi) & actual.notna()).sum())
      for p, (lo, hi) in periods.items()}
print("Subsample RMSE  (n: " + ", ".join(f"{p}={n}" for p, n in ns.items()) + ")\n")
print(tbl.round(3).to_string())
tbl.round(4).to_csv(HERE / "subsample_rmse.csv")
print("\nSaved: subsample_rmse.csv")


# ── (2) Rolling 12-month RMSE: headline vs AR(1) ────────────────────────────────
head = cand["VPC3+VPC4_nocov"].reindex(actual.index)
ar1  = bench["AR1"].reindex(actual.index)
roll_h = np.sqrt(((actual - head) ** 2).rolling(ROLL).mean())
roll_a = np.sqrt(((actual - ar1) ** 2).rolling(ROLL).mean())

fig, ax = plt.subplots(figsize=(10, 4.6))
ax.axvspan(COVID_START, COVID_END, color="gold", alpha=0.12, zorder=0, label="COVID window")
ax.plot(roll_a.index, roll_a.values, color="#8C8C8C", ls="--", lw=1.6, zorder=3, label="AR(1)")
ax.plot(roll_h.index, roll_h.values, color="#00467F", lw=2.0, zorder=4, label="VPC3+VPC4 (headline)")
ax.fill_between(roll_h.index, roll_h.values, roll_a.values,
                where=(roll_a.values >= roll_h.values), color="#00467F", alpha=0.10, zorder=2)
ax.set_ylabel(f"Rolling {ROLL}-month RMSE")
ax.legend(frameon=False, ncol=3, fontsize=9, loc="upper left")
ax.spines[["top", "right"]].set_visible(False)
ax.grid(axis="y", ls="--", alpha=0.3)
ax.margins(x=0.01)
plt.tight_layout()
plt.show()
