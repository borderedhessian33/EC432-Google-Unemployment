"""
EC 432 Project — Real-time model selection (rolling/expanding annual selection)
================================================================================
The look-ahead-free real-time headline.  Selects, each year, from the locked
candidate pool on the basis of past out-of-sample performance only:

  pool       = 20 locked candidates (candidate_forecasts.csv) + AR(1)/AR(2)
  procedure  = for each test year y, rank the pool by RMSE over ALL evaluation
               months before January y, adopt the leader for the 12 months of y.
  evaluation = 2019–2025 (2018 is the initial track record), Clark–West vs AR(1).

Consumes only the clean, LASSO-free locked outputs (11_candidate_grid.py,
10_benchmarks.py).  Generates nothing itself but the selection results.

Output: rolling_forecasts.csv, rolling_menu.csv, rolling_picks.csv
"""

import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import norm

warnings.filterwarnings("ignore")
HERE = Path(__file__).parent
EVAL_START  = pd.Timestamp("2018-01-01")
EVAL_END    = pd.Timestamp("2025-12-01")
COVID_START = pd.Timestamp("2020-04-01")
COVID_END   = pd.Timestamp("2021-12-01")
TEST_START  = pd.Timestamp("2019-01-01")     # first year selected from a prior record
HAC_LAGS    = 1                              # PUB_LAG - 1


# ── Locked candidate pool + benchmarks ───────────────────────────────────────────
cand  = pd.read_csv(HERE / "candidate_forecasts.csv", index_col=0, parse_dates=True)
bench = pd.read_csv(HERE / "ar_benchmarks.csv", index_col=0, parse_dates=True)
actual = bench["actual"]
AR1, AR2 = bench["AR1"], bench["AR2"]

pool = {c: cand[c] for c in cand.columns}    # 20 candidates
pool["AR1"] = AR1                            # + naive fallbacks
pool["AR2"] = AR2

covid = (actual.index >= COVID_START) & (actual.index <= COVID_END)


def rmse(f, lo, hi, excl=False):
    f = f.reindex(actual.index)
    m = (actual.index >= lo) & (actual.index <= hi) & actual.notna() & f.notna()
    if excl: m &= ~covid
    return float(np.sqrt(np.mean((actual[m] - f[m]) ** 2)))


# ── Rolling / expanding annual selection ─────────────────────────────────────────
fc = pd.Series(index=actual.index, dtype=float)
picks = {}
for yr in range(TEST_START.year, EVAL_END.year + 1):
    sel_hi = pd.Timestamp(f"{yr-1}-12-01")                       # rank on data before year yr
    val = pd.Series({n: rmse(f, EVAL_START, sel_hi) for n, f in pool.items()}).sort_values()
    pick = val.index[0]; picks[yr] = pick
    ym = (actual.index >= pd.Timestamp(f"{yr}-01-01")) & (actual.index <= pd.Timestamp(f"{yr}-12-01"))
    fc[ym] = pool[pick].reindex(actual.index)[ym]

lo, hi = TEST_START, EVAL_END


def cw(f_restricted, f_unrestricted):
    """Clark–West: is the (nesting) selected forecast better than the AR(1) null?"""
    fr = f_restricted.reindex(actual.index); fu = f_unrestricted.reindex(actual.index)
    m = (actual.index >= lo) & (actual.index <= hi) & actual.notna() & fr.notna() & fu.notna()
    y, a, b = actual[m].values, fr[m].values, fu[m].values
    ft = (y - a) ** 2 - ((y - b) ** 2 - (a - b) ** 2)
    r = sm.OLS(ft, np.ones((len(ft), 1))).fit(cov_type="HAC", cov_kwds={"maxlags": HAC_LAGS})
    return float(r.tvalues[0]), float(1 - norm.cdf(r.tvalues[0]))


# ── Report ───────────────────────────────────────────────────────────────────────
print("=" * 60)
print("LOCKED GRID — full-window RMSE (eval 2018–2025)")
print("=" * 60)
menu = pd.Series({n: rmse(f, EVAL_START, EVAL_END) for n, f in pool.items()}).sort_values()
for n, r in menu.items():
    print(f"  {n:<18}{r:>9.4f}  (exC {rmse(pool[n], EVAL_START, EVAL_END, True):.4f})")

print("\nRolling yearly picks:")
for yr, p in picks.items():
    print(f"  {yr}: {p}")

print("\n" + "=" * 60)
print("ROLLING-SELECTED (real-time) — test 2019–2025")
print("=" * 60)
print(f"{'Model':<24}{'RMSE full':>11}{'RMSE exC':>11}")
print("-" * 60)
print(f"{'Rolling-selected':<24}{rmse(fc,lo,hi):>11.4f}{rmse(fc,lo,hi,True):>11.4f}")
print(f"{'AR(1)':<24}{rmse(AR1,lo,hi):>11.4f}{rmse(AR1,lo,hi,True):>11.4f}")
print(f"{'AR(2)':<24}{rmse(AR2,lo,hi):>11.4f}{rmse(AR2,lo,hi,True):>11.4f}")
print("-" * 60)
c = cw(AR1, fc)          # restricted = AR(1) (nested), unrestricted = selected model
print(f"Rolling vs AR(1): Clark–West t = {c[0]:+.2f} (p = {c[1]:.3f})")
print("=" * 60)

pd.DataFrame({"actual": actual, "rolling": fc, "AR1": AR1, "AR2": AR2}
             ).rename_axis("Tarih").to_csv(HERE / "rolling_forecasts.csv")
menu.to_frame("rmse_full").to_csv(HERE / "rolling_menu.csv")
pd.Series(picks, name="pick").rename_axis("year").to_frame().to_csv(HERE / "rolling_picks.csv")
print("\nSaved: rolling_forecasts.csv, rolling_menu.csv, rolling_picks.csv")
