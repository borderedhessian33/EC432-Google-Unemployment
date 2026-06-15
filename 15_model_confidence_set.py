"""
EC 432 Project — Model Confidence Set (Hansen, Lunde & Nason 2011)
==================================================================
Determines the set of models statistically indistinguishable from the best,
controlling for multiple comparisons — the rigorous "did we cover the ground"
statement.  Loss = squared nowcast error; T_max statistic; moving-block bootstrap.

Pool: the pre-registered no-LASSO factor grid (cached) + AR(1), AR(2) benchmarks.
LASSO models are NOT part of the candidate set and are excluded entirely; the
L=0 / L=1 bridge-depth variants are robustness checks and are excluded too.
Evaluated on 2018–2025, full and ex-COVID.

A model is in the α-MCS if its MCS p-value ≥ α.  We report the 75% and 90% sets.
Output: mcs_results.csv
"""

import warnings
from pathlib import Path
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
HERE = Path(__file__).parent
EVAL_START  = pd.Timestamp("2018-01-01")
EVAL_END    = pd.Timestamp("2025-12-01")
COVID_START = pd.Timestamp("2020-04-01")
COVID_END   = pd.Timestamp("2021-12-01")
B_BOOT = 5000
BLOCK  = 3
rng = np.random.default_rng(42)


def load(f):
    d = pd.read_csv(HERE / f); d[d.columns[0]] = pd.to_datetime(d[d.columns[0]])
    return d.set_index(d.columns[0])

# ── Candidate pool: the locked 20-model L=2 grid + AR benchmarks ─────────────────
cache = load("candidate_forecasts.csv")                  # 20 locked no-LASSO candidates
bench = load("ar_benchmarks.csv")                        # AR1, AR2 (LASSO-free)
actual = bench["actual"]

models = {c: cache[c] for c in cache.columns}
models["AR1"] = bench["AR1"]
models["AR2"] = bench["AR2"]
models["NoTrends"] = bench["NoTrends"]            # filtering structure, no Google
models["NoTrends_cov"] = bench["NoTrends_cov"]    # + COVID dummy


def mcs(loss_df, alpha_report=(0.25, 0.10)):
    """Hansen-Lunde-Nason MCS with T_max statistic + moving-block bootstrap.
    loss_df: T×m DataFrame of per-period losses. Returns MCS p-values per model."""
    L = loss_df.values.astype(float)
    T, m0 = L.shape
    # precompute moving-block bootstrap row-index sets (shared across models)
    n_blocks = int(np.ceil(T / BLOCK))
    boot_rows = np.empty((B_BOOT, T), dtype=int)
    for b in range(B_BOOT):
        starts = rng.integers(0, T, size=n_blocks)
        idx = np.concatenate([np.arange(s, s + BLOCK) % T for s in starts])[:T]
        boot_rows[b] = idx

    cols = list(loss_df.columns)
    alive = list(range(m0))
    mcs_p = {}
    running = 0.0
    while len(alive) > 1:
        Lm = L[:, alive]                       # T × k
        k = Lm.shape[1]
        Lbar_t = Lm.mean(axis=1, keepdims=True)
        z = Lm - Lbar_t                        # deviations from cross-model mean
        dbar = z.mean(axis=0)                  # k
        # bootstrap means
        dstar = np.empty((B_BOOT, k))
        for b in range(B_BOOT):
            dstar[b] = z[boot_rows[b]].mean(axis=0)
        var = dstar.var(axis=0, ddof=1)
        se = np.sqrt(var); se[se < 1e-12] = np.nan
        t = dbar / se
        Tmax = np.nanmax(t)
        tstar = (dstar - dbar[None, :]) / se[None, :]
        Tmax_star = np.nanmax(tstar, axis=1)
        p = float(np.mean(Tmax_star >= Tmax))
        running = max(running, p)
        worst = alive[int(np.nanargmax(t))]
        mcs_p[cols[worst]] = running
        alive.remove(worst)
    mcs_p[cols[alive[0]]] = 1.0                # last survivor
    return pd.Series(mcs_p)


def run(excl):
    m = (actual.index >= EVAL_START) & (actual.index <= EVAL_END) & actual.notna()
    if excl:
        m &= ~((actual.index >= COVID_START) & (actual.index <= COVID_END))
    idx = actual.index[m]
    M = pd.DataFrame({k: v.reindex(idx) for k, v in models.items()})
    M = M.dropna(axis=0)                       # common non-NaN dates
    a = actual.reindex(M.index)
    loss = (M.sub(a, axis=0)) ** 2
    rmse = np.sqrt(loss.mean()).sort_values()
    p = mcs(loss).reindex(rmse.index)
    out = pd.DataFrame({"RMSE": rmse.round(4), "MCS_p": p.round(3)})
    return out, M.shape[0]


for excl, lbl in [(False, "FULL window (2018–2025)"), (True, "ex-COVID")]:
    res, n = run(excl)
    print("\n" + "=" * 60)
    print(f"MODEL CONFIDENCE SET — {lbl}   (n={n}, B={B_BOOT}, block={BLOCK})")
    print("=" * 60)
    print(f"{'Model':<20}{'RMSE':>9}{'MCS p':>9}  in 75%? in 90%?")
    print("-" * 60)
    for name, row in res.iterrows():
        in75 = "yes" if row["MCS_p"] >= 0.25 else " - "
        in90 = "yes" if row["MCS_p"] >= 0.10 else " - "
        print(f"{name:<20}{row['RMSE']:>9.4f}{row['MCS_p']:>9.3f}   {in75:>5}  {in90:>5}")
    set90 = list(res.index[res["MCS_p"] >= 0.10])
    print(f"\n90% MCS ({len(set90)} models): {set90}")
    print(f"AR(1) in 90% MCS? {'YES' if res.loc['AR1','MCS_p'] >= 0.10 else 'NO'}"
          f"  (AR1 MCS p = {res.loc['AR1','MCS_p']:.3f})")
    res.to_csv(HERE / f"mcs_results_{'exc' if excl else 'full'}.csv")

print("\nSaved: mcs_results_full.csv, mcs_results_exc.csv")
