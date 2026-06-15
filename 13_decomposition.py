"""
EC 432 Project — Decomposition: where the candidate's gain comes from
======================================================================
A candidate's edge over AR(1) could come from two things: (i) the state-space
FILTERING structure (latent AR factor + measurement error optimally smooths u and
fills the publication gap), and (ii) the GOOGLE signal.  This isolates them with
a clean ladder, reading only the locked, LASSO-free pipeline outputs:

    AR(1)                              — pure autoregressive floor   (10_benchmarks.py)
    → no-Trends SS (+COVID)            — gain from the filtering structure alone
    → headline candidate (with Google) — remaining gain = pure Google contribution

The no-Trends nowcasts are produced by 10_benchmarks.py (NoTrends / NoTrends_cov);
the candidate nowcasts by 11_candidate_grid.py.  Nothing here uses LASSO.

Output: printed decomposition table + decomposition.csv
"""

import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
HERE = Path(__file__).parent
COVID_START = pd.Timestamp("2020-04-01")
COVID_END   = pd.Timestamp("2021-12-01")
HAC_LAGS    = 1

# headline Google model for the decomposition (best full-window candidate)
HEADLINE = "VPC3+VPC4_nocov"


# ── Load clean benchmark and candidate nowcasts ──────────────────────────────────
bench = pd.read_csv(HERE / "ar_benchmarks.csv", index_col=0, parse_dates=True)
cand  = pd.read_csv(HERE / "candidate_forecasts.csv", index_col=0, parse_dates=True)

actual = bench["actual"]
covid  = (actual.index >= COVID_START) & (actual.index <= COVID_END)

series = {
    "AR(1) benchmark":                         bench["AR1"],
    "no-Trends SS  (filtering, no Google)":     bench["NoTrends"],
    "no-Trends SS  + COVID dummy":              bench["NoTrends_cov"],
    f"{HEADLINE}  (filtering + Google)":        cand[HEADLINE],
}


def rmse(f, excl=False):
    f = f.reindex(actual.index); m = actual.notna() & f.notna()
    if excl: m &= ~covid
    return float(np.sqrt(np.mean((actual[m] - f[m]) ** 2)))

def dm(f1, f2):
    f1 = f1.reindex(actual.index); f2 = f2.reindex(actual.index)
    m = actual.notna() & f1.notna() & f2.notna()
    d = (actual[m] - f1[m]) ** 2 - (actual[m] - f2[m]) ** 2
    r = sm.OLS(d.values, np.ones((len(d), 1))).fit(cov_type="HAC", cov_kwds={"maxlags": HAC_LAGS})
    return float(r.tvalues[0]), float(r.pvalues[0])


# ── Report ───────────────────────────────────────────────────────────────────────
print("=" * 76)
print("DECOMPOSITION — where the Google gain comes from   (eval 2018–2025)")
print("=" * 76)
print(f"{'Model':<42}{'RMSE full':>11}{'RMSE exC':>11}")
print("-" * 76)
for name, f in series.items():
    print(f"{name:<42}{rmse(f):>11.4f}{rmse(f, True):>11.4f}")
print("-" * 76)

ar1, nt_cov, head = bench["AR1"], bench["NoTrends_cov"], cand[HEADLINE]
g_struct = rmse(ar1) - rmse(nt_cov)
g_google = rmse(nt_cov) - rmse(head)
g_total  = rmse(ar1) - rmse(head)
print(f"Full-window ΔRMSE decomposition (vs AR(1) = {rmse(ar1):.4f}):")
print(f"   filtering structure (+COVID):  {g_struct:+.4f}")
print(f"   Google signal (no-Trends→cand): {g_google:+.4f}")
print(f"   total gain candidate vs AR(1): {g_total:+.4f}")
d = dm(head, nt_cov)
print(f"\n   DM, candidate vs no-Trends SS+COVID (isolates Google): "
      f"t = {d[0]:+.2f} (p = {d[1]:.3f})")
print("=" * 76)

pd.DataFrame({name: f.reindex(actual.index) for name, f in series.items()},
             ).assign(actual=actual).rename_axis("Tarih").to_csv(HERE / "decomposition.csv")
print("\nSaved: decomposition.csv")


# ── Decomposition figure: RMSE across the ladder (full vs ex-COVID) ──────────────
MAINBLUE, ORANGE = "#00467F", "#E05C1A"
ladder = [("AR(1)",          bench["AR1"]),
          ("No-Trends",      bench["NoTrends"]),
          ("No-Trends$+$C",  bench["NoTrends_cov"]),
          ("VPC3+VPC4",      cand[HEADLINE])]
labels = [m[0] for m in ladder]
full   = [rmse(m[1]) for m in ladder]
exc    = [rmse(m[1], True) for m in ladder]

x = np.arange(len(labels)); w = 0.38
fig, ax = plt.subplots(figsize=(7, 4.2))
b1 = ax.bar(x - w/2, full, w, label="Full window", color=MAINBLUE, alpha=0.90, zorder=3)
b2 = ax.bar(x + w/2, exc,  w, label="Ex-COVID",    color=ORANGE,   alpha=0.90, zorder=3)
ax.axhline(rmse(bench["AR1"]), color="grey", ls="--", lw=1, zorder=1)   # AR(1) full reference
ax.bar_label(b1, fmt="%.3f", fontsize=8, padding=2)
ax.bar_label(b2, fmt="%.3f", fontsize=8, padding=2)
ax.set_xticks(x); ax.set_xticklabels(labels)
ax.set_ylabel("RMSE")
ax.set_ylim(0.40, 0.63)
ax.legend(frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.08))
ax.spines[["top", "right"]].set_visible(False)
ax.grid(axis="y", ls="--", alpha=0.3, zorder=0)
plt.tight_layout()
plt.show()
