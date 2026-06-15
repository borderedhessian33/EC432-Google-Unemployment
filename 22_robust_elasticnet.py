"""
EC 432 Project — Regularized ARDL on the keywords directly  (no PCA)
====================================================================
With only 21 keywords (6 LASSO-selected), PCA is arguably unnecessary.  Here we
skip the PCA/varimax step and put the (deseasonalised, YoY-differenced) keyword
series straight into a direct-horizon ARDL, letting a penalty do the selection:

    u_t = α + Σ_{ℓ=2}^{p+1} φ_ℓ u_{t-ℓ}                  (unemployment lags from t-2)
              + Σ_{i=1}^{N} Σ_{k=0}^{K} β_{ik} x_{i,t-k}   (keyword lags, contemporaneous OK)
              + ε_t

estimated with Ridge, Elastic-Net and Lasso (penalty strength chosen by
time-series CV at every origin).  Everything else matches the project:
  • STL seasonal pattern frozen at Dec-2017; YoY (12-month) difference.
  • Publication lag h = 2 (u_{t-1} unobserved): u-lags start at t-2.
  • Recursive pseudo-OOS 2018–2025, expanding window, specification frozen.
  • Predictors standardised inside the CV pipeline on the training window only
    (no look-ahead); intercept unpenalised.

This is the regularized counterpart of 20_robust_ardl_direct.py.
Output: regularized_ardl_forecasts.csv + printed tables (+ plot via plt.show()).
"""

import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy.stats import norm

from statsmodels.tsa.seasonal import STL
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV, ElasticNetCV, LassoCV
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import TimeSeriesSplit

warnings.filterwarnings("ignore")
HERE = Path(__file__).parent

# ── Configuration ──────────────────────────────────────────────────────────────
FREEZE_DATE = pd.Timestamp("2017-12-31")
EVAL_START  = pd.Timestamp("2018-01-01")
EVAL_END    = pd.Timestamp("2025-12-01")
COVID_START = pd.Timestamp("2020-04-01")
COVID_END   = pd.Timestamp("2021-12-31")
PUB_LAG     = 2
HAC_LAGS    = PUB_LAG - 1
P_LAGS      = 3        # unemployment lags: t-2, t-3, t-4
K_LAGS      = 2        # keyword lags:      k = 0,1,2
USE_ALL_KEYWORDS = True   # True → all 21 ; False → the 6 LASSO-selected terms
MIN_TRAIN   = 30


# ── Frozen deseasonalisation (no PCA) ───────────────────────────────────────────
def frozen_deseas(series, freeze):
    train = series[series.index <= freeze].dropna()
    if len(train) < 24:
        return series
    stl = STL(train, period=12, robust=False).fit()
    pattern = stl.seasonal.groupby(stl.seasonal.index.month).mean()
    seas = pd.Series(series.index.month, index=series.index).map(pattern).values
    return series - seas


print("Loading data …")
trends = pd.read_csv(HERE / "calibrated_trends_custom.csv",
                     index_col=0, parse_dates=True).resample("MS").first()
unemp = pd.read_csv(HERE / "unemp_csv.csv", sep=";", decimal=",",
                    encoding="utf-8-sig", index_col=0, parse_dates=False)
unemp.columns = ["u"]
unemp = unemp[unemp.index.notna() & (unemp.index.astype(str) != "nan")]
unemp.index = pd.to_datetime(unemp.index, format="%Y-%m")
unemp = unemp.resample("MS").first()
u_full = unemp["u"].astype(float).dropna()

deseas = pd.DataFrame(index=trends.index, columns=trends.columns, dtype=float)
for c in trends.columns:
    deseas[c] = frozen_deseas(trends[c].astype(float), FREEZE_DATE)
kw = deseas.diff(12).dropna()              # stationary keyword signals (all 21)
train_mask = kw.index <= FREEZE_DATE

# optional 6-keyword subset by LASSO (same as the main pipeline) for the variant
if not USE_ALL_KEYWORDS:
    from sklearn.linear_model import LassoCV as _L
    Xp = StandardScaler().fit_transform(kw[train_mask])
    ul = u_full.reindex(kw[train_mask].index)
    ok = ul.notna().values
    lc = _L(cv=5, max_iter=10000, random_state=42).fit(Xp[ok], ul.dropna().values)
    keep = kw.columns[np.abs(lc.coef_) > 0].tolist()
    kw = kw[keep]
print(f"  using {kw.shape[1]} keyword series  (USE_ALL_KEYWORDS={USE_ALL_KEYWORDS})")


# ── ARDL design with raw keyword lags ───────────────────────────────────────────
def build_design(p, K):
    idx = u_full.index.union(kw.index)
    u = u_full.reindex(idx)
    d = pd.DataFrame({"y": u}, index=idx)
    ar = []
    for l in range(2, p + 2):
        d[f"u_l{l}"] = u.shift(l); ar.append(f"u_l{l}")
    kf = []
    for c in kw.columns:
        s = kw[c].reindex(idx)
        for k in range(0, K + 1):
            nm = f"{c}__l{k}"; d[nm] = s.shift(k); kf.append(nm)
    return d, ar, kf


D, AR_COLS, KW_COLS = build_design(P_LAGS, K_LAGS)
ALL_FEAT = AR_COLS + KW_COLS
print(f"  ARDL design: {len(AR_COLS)} u-lags + {len(KW_COLS)} keyword-lag regressors "
      f"= {len(ALL_FEAT)} predictors")


# ── Estimator factories (fresh pipeline each fit; CV picks the penalty) ──────────
TSCV = TimeSeriesSplit(n_splits=5)
def make_ridge():
    return make_pipeline(StandardScaler(),
                         RidgeCV(alphas=np.logspace(-3, 3, 25)))
def make_enet():
    return make_pipeline(StandardScaler(),
                         ElasticNetCV(l1_ratio=[.2, .5, .8, 1.0], n_alphas=50,
                                      cv=TSCV, max_iter=20000, random_state=42))
def make_lasso():
    return make_pipeline(StandardScaler(),
                         LassoCV(n_alphas=50, cv=TSCV, max_iter=20000, random_state=42))


def recursive(make_model, feat):
    eval_dates = u_full[(u_full.index >= EVAL_START) & (u_full.index <= EVAL_END)].index
    out = pd.Series(index=eval_dates, dtype=float)
    for t in eval_dates:
        cutoff = t - pd.DateOffset(months=PUB_LAG)
        tr = D[D.index <= cutoff].dropna(subset=feat + ["y"])
        xrow = D.loc[[t], feat]
        if len(tr) < MIN_TRAIN or xrow.isna().any(axis=1).iloc[0]:
            continue
        m = make_model()
        m.fit(tr[feat].values, tr["y"].values)
        out[t] = float(m.predict(xrow.values)[0])
    return out


# AR-only direct benchmark (unpenalised OLS, u-lags only)
def recursive_ar(feat):
    eval_dates = u_full[(u_full.index >= EVAL_START) & (u_full.index <= EVAL_END)].index
    out = pd.Series(index=eval_dates, dtype=float)
    for t in eval_dates:
        cutoff = t - pd.DateOffset(months=PUB_LAG)
        tr = D[D.index <= cutoff].dropna(subset=feat + ["y"])
        xrow = D.loc[[t], feat]
        if len(tr) < MIN_TRAIN or xrow.isna().any(axis=1).iloc[0]:
            continue
        X = np.column_stack([np.ones(len(tr)), tr[feat].values])
        b, *_ = np.linalg.lstsq(X, tr["y"].values, rcond=None)
        out[t] = float(np.r_[1.0, xrow.values.ravel()] @ b)
    return out


print("\nRunning recursive pseudo-OOS (this re-runs CV each month) …")
fc_ar    = recursive_ar(AR_COLS)
fc_ridge = recursive(make_ridge, ALL_FEAT);  print("  ridge done")
fc_enet  = recursive(make_enet,  ALL_FEAT);  print("  elastic-net done")
fc_lasso = recursive(make_lasso, ALL_FEAT);  print("  lasso done")

actual = u_full.reindex(fc_ridge.index)
covid  = (actual.index >= COVID_START) & (actual.index <= COVID_END)


# ── Evaluation helpers ───────────────────────────────────────────────────────────
def rmse(f, excl=False):
    m = actual.notna() & f.notna()
    if excl:
        m &= ~covid
    return float(np.sqrt(np.mean((actual[m] - f[m]) ** 2)))

def dm_test(f1, f2):                       # f1=AR vs f2=model ; t>0 ⇒ model better
    m = actual.notna() & f1.notna() & f2.notna()
    d = (actual[m] - f1[m]) ** 2 - (actual[m] - f2[m]) ** 2
    r = sm.OLS(d.values, np.ones((len(d), 1))).fit(cov_type="HAC", cov_kwds={"maxlags": HAC_LAGS})
    return float(r.tvalues[0]), float(r.pvalues[0])

def cw_test(f_restr, f_unr):               # nested AR-only vs regularized ARDL
    m = actual.notna() & f_restr.notna() & f_unr.notna()
    y, fr, fu = actual[m].values, f_restr[m].values, f_unr[m].values
    fterm = (y - fr) ** 2 - ((y - fu) ** 2 - (fr - fu) ** 2)
    r = sm.OLS(fterm, np.ones((len(fterm), 1))).fit(cov_type="HAC", cov_kwds={"maxlags": HAC_LAGS})
    return float(r.tvalues[0]), float(1 - norm.cdf(r.tvalues[0]))

# reference RMSEs from existing engines
ref = {}
for fn, cols in [("option_b_lag2_forecasts.csv", {"AR(1) recursive": "AR1", "B5 state space": "B5_VPC1_AR1"}),
                 ("ardl_forecasts.csv",          {"ARDL (PCA, sel)": "ARDL_sel"})]:
    fp = HERE / fn
    if fp.exists():
        e = pd.read_csv(fp); e[e.columns[0]] = pd.to_datetime(e[e.columns[0]]); e = e.set_index(e.columns[0])
        for lbl, col in cols.items():
            if col in e.columns:
                s = e[col].reindex(actual.index)
                ref[lbl] = (rmse(s), rmse(s, True))

print("\n" + "=" * 80)
print(f"RESULTS — regularized ARDL on keywords (p={P_LAGS}, K={K_LAGS}, "
      f"N={kw.shape[1]}), n_full={int((actual.notna()&fc_ridge.notna()).sum())}")
print("=" * 80)
print(f"{'Model':<30}{'RMSE full':>11}{'RMSE exC':>11}{'DM vs AR':>11}{'CW vs AR':>11}")
print("-" * 80)
print(f"{'AR-only direct (benchmark)':<30}{rmse(fc_ar):>11.4f}{rmse(fc_ar,True):>11.4f}{'—':>11}{'—':>11}")
for lbl, f in [("Ridge ARDL", fc_ridge), ("Elastic-Net ARDL", fc_enet), ("Lasso ARDL", fc_lasso)]:
    dm = dm_test(fc_ar, f); cw = cw_test(fc_ar, f)
    dm_s = f"{dm[0]:+.2f}{'*' if dm[1]<0.05 else ''}"
    cw_s = f"{cw[0]:+.2f}{'**' if cw[1]<0.05 else ('*' if cw[1]<0.10 else '')}"
    print(f"{lbl:<30}{rmse(f):>11.4f}{rmse(f,True):>11.4f}{dm_s:>11}{cw_s:>11}")
print("-" * 80)
for lbl, (rf, re) in ref.items():
    print(f"{'[ref] '+lbl:<30}{rf:>11.4f}{re:>11.4f}{'—':>11}{'—':>11}")
print("=" * 80)
print("DM>0 ⇒ regularized ARDL beats AR-only (HAC, h-1 lags); * p<.05.")
print("CW one-sided nested test; * p<.10, ** p<.05.")


# ── Which keywords does the penalty actually use? (final fit on full training) ───
m = make_enet(); tr0 = D[D.index <= FREEZE_DATE].dropna(subset=ALL_FEAT + ["y"])
m.fit(tr0[ALL_FEAT].values, tr0["y"].values)
coefs = pd.Series(m[-1].coef_, index=ALL_FEAT)
nz = coefs[coefs.abs() > 1e-8]
print(f"\nElastic-Net on training window keeps {len(nz)}/{len(ALL_FEAT)} regressors.")
print("Top retained keyword lags by |coef|:")
print(nz[[c for c in nz.index if '__l' in c]].abs().sort_values(ascending=False).head(8).to_string())


# ── Save + plot ──────────────────────────────────────────────────────────────────
out = pd.DataFrame({"actual": actual, "AR_direct": fc_ar,
                    "Ridge": fc_ridge, "ElasticNet": fc_enet, "Lasso": fc_lasso})
out.index.name = "Tarih"
out.to_csv(HERE / "regularized_ardl_forecasts.csv")
print("\nSaved: regularized_ardl_forecasts.csv")

fig, ax = plt.subplots(figsize=(11, 4.4))
ax.axvspan(COVID_START, COVID_END, color="gold", alpha=0.18)
ax.plot(actual.index, actual.values, color="black", lw=2.0, label="Actual $u_t$")
ax.plot(fc_ar.index,   fc_ar.values,   color="#7F8C8D", lw=1.3, ls="--", label="AR-only direct")
ax.plot(fc_ridge.index, fc_ridge.values, color="#00467F", lw=1.5, label="Ridge ARDL")
ax.plot(fc_enet.index,  fc_enet.values,  color="#C0392B", lw=1.5, ls="-.", label="Elastic-Net ARDL")
ax.plot(fc_lasso.index, fc_lasso.values, color="#27AE60", lw=1.5, ls=":", label="Lasso ARDL")
ax.set_ylabel("Non-agricultural unemployment (%)")
ax.set_title(f"Regularized ARDL on keywords directly (p={P_LAGS}, K={K_LAGS}, N={kw.shape[1]})")
ax.legend(loc="upper right", ncol=2); ax.grid(axis="y", ls="--", alpha=0.35)
plt.tight_layout()
plt.show()
