"""
EC 432 Project — Out-of-sample accuracy table  (PAPER TABLE 2)
==============================================================
Builds the full Table 2 of the paper: the recursive pseudo-out-of-sample accuracy
of the 20 locked candidates plus the four Google-free benchmarks, 2018-2025
(n = 96), with — for every candidate against AR(1):

    RMSE (full)   RMSE (ex-COVID)   RMSE/AR(1)   p_DM   p_CW

  • p_DM : two-sided Diebold-Mariano p-value, Harvey-Leybourne-Newbold small-sample
           corrected, HAC (h-1 = 1 lag), Student-t(n-1).   (h = 2 publication lag)
  • p_CW : one-sided Clark-West p-value for the nested comparison (AR(1) nested in
           the candidate), HAC (h-1 = 1 lag), standard-normal.

This is a thin reader: it consumes the locked nowcasts produced by
    11_candidate_grid.py  -> candidate_forecasts.csv   (20 candidates)
    10_benchmarks.py      -> ar_benchmarks.csv         (AR1, AR2, NoTrends, NoTrends_cov)
and computes only the accuracy statistics — it estimates no models itself.

Output: printed Table 2 + oos_accuracy_table.csv
"""

import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import norm, t as tdist

warnings.filterwarnings("ignore")
HERE = Path(__file__).parent
EVAL_START  = pd.Timestamp("2018-01-01")
EVAL_END    = pd.Timestamp("2025-12-01")
COVID_START = pd.Timestamp("2020-04-01")
COVID_END   = pd.Timestamp("2021-12-01")
PUB_LAG     = 2
HAC_LAGS    = PUB_LAG - 1            # MA(h-1) structure of direct h-step errors


# ── Load the locked nowcasts ─────────────────────────────────────────────────────
cand  = pd.read_csv(HERE / "candidate_forecasts.csv", index_col=0, parse_dates=True)
bench = pd.read_csv(HERE / "ar_benchmarks.csv", index_col=0, parse_dates=True)
actual = bench["actual"]
AR1 = bench["AR1"]
covid = (actual.index >= COVID_START) & (actual.index <= COVID_END)


def rmse(f, excl=False):
    f = f.reindex(actual.index); m = actual.notna() & f.notna()
    if excl: m &= ~covid
    return float(np.sqrt(np.mean((actual[m] - f[m]) ** 2)))


def dm_harvey(f):
    """Two-sided Diebold-Mariano vs AR(1), Harvey small-sample correction (h=2)."""
    f = f.reindex(actual.index); m = actual.notna() & f.notna() & AR1.notna()
    d = (actual[m] - AR1[m]) ** 2 - (actual[m] - f[m]) ** 2     # >0 favours candidate
    n = len(d)
    r = sm.OLS(d.values, np.ones((n, 1))).fit(cov_type="HAC", cov_kwds={"maxlags": HAC_LAGS})
    h = PUB_LAG
    corr = np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)       # Harvey-Leybourne-Newbold
    dm = float(r.tvalues[0]) * corr
    return dm, float(2 * (1 - tdist.cdf(abs(dm), df=n - 1)))


def cw(f):
    """One-sided Clark-West vs the nested AR(1)."""
    f = f.reindex(actual.index); m = actual.notna() & f.notna() & AR1.notna()
    y, fr, fu = actual[m].values, AR1[m].values, f[m].values
    ft = (y - fr) ** 2 - ((y - fu) ** 2 - (fr - fu) ** 2)
    r = sm.OLS(ft, np.ones((len(ft), 1))).fit(cov_type="HAC", cov_kwds={"maxlags": HAC_LAGS})
    return float(r.tvalues[0]), float(1 - norm.cdf(r.tvalues[0]))


def stars(p):
    return "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else ""


# ── Build the table ──────────────────────────────────────────────────────────────
ar1_rmse = rmse(AR1)
rows = []
for c in cand.columns:
    _, pdm = dm_harvey(cand[c]); tcw, pcw = cw(cand[c])
    rows.append({"model": c, "RMSE": rmse(cand[c]), "RMSE_exC": rmse(cand[c], True),
                 "RMSE_ratio": rmse(cand[c]) / ar1_rmse, "p_DM": pdm, "p_CW": pcw})
for b in ["AR1", "AR2", "NoTrends", "NoTrends_cov"]:
    rows.append({"model": b, "RMSE": rmse(bench[b]), "RMSE_exC": rmse(bench[b], True),
                 "RMSE_ratio": rmse(bench[b]) / ar1_rmse, "p_DM": np.nan, "p_CW": np.nan})

tbl = pd.DataFrame(rows).set_index("model")
tbl = tbl.sort_values("RMSE")

print("=" * 78)
print("TABLE 2 — Out-of-sample nowcast accuracy, 2018-2025 (n = 96)")
print("=" * 78)
print(f"{'Model':<20}{'RMSE':>8}{'RMSE_exC':>10}{'/AR(1)':>8}{'p_DM':>8}{'p_CW':>10}")
print("-" * 78)
for m, r in tbl.iterrows():
    pdm = "  —  " if np.isnan(r["p_DM"]) else f"{r['p_DM']:.3f}"
    pcw = "  —  " if np.isnan(r["p_CW"]) else f"{r['p_CW']:.3f}{stars(r['p_CW'])}"
    print(f"{m:<20}{r['RMSE']:>8.3f}{r['RMSE_exC']:>10.3f}{r['RMSE_ratio']:>8.3f}{pdm:>8}{pcw:>10}")
print("=" * 78)
print("p_DM: two-sided Diebold-Mariano (Harvey-corrected) vs AR(1).")
print("p_CW: one-sided Clark-West vs nested AR(1).  CW stars: * .10  ** .05  *** .01")

tbl.round(4).to_csv(HERE / "oos_accuracy_table.csv")
print("\nSaved: oos_accuracy_table.csv")
