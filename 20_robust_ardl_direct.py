"""
EC 432 Project — Direct-Horizon Factor-Augmented ARDL  (alternative to the state space)
=======================================================================================
A single-equation alternative to the Kalman / state-space nowcast.  Because the
4 varimax PCs can inject extra idiosyncratic noise through the state-space
*measurement* block, here the factors enter directly in one OLS equation and we
let standard model-selection decide how many lags and which PCs survive.

Direct two-month nowcast (publication lag h = 2, so u_{t-1} is NOT yet observed):

    u_t = α + Σ_{ℓ=2}^{p+1} φ_ℓ u_{t-ℓ}                 (unemployment lags start at t-2)
              + Σ_{j∈S} Σ_{k=0}^{K} β_{jk} F_{j,t-k}      (Google factors, contemporaneous OK)
              + ε_t

This is a direct-horizon factor-augmented ARDL.  Forecasting u_t directly (rather
than iterating u_{t-1} then u_t) estimates the exact relationship needed for the
two-month information gap.

Procedure (mirrors the rest of the project)
  1. Frozen preprocessing at Dec-2017 → VPC1..VPC4 (STL deseas, YoY diff, z-score,
     PCA + varimax on ALL 21 keywords — no LASSO).  Same factors as 02_pca_varimax_factors.py.
  2. Lag selection: grid over (p, K) with ALL PCs, chosen by BIC on the training
     window (≤ 2017-12) — no look-ahead.
  3. PC selection: start with all PCs and drop them one by one (general-to-specific
     backward elimination by BIC) at the chosen (p, K).
  4. Recursive pseudo-OOS (2018–2025): at each origin t, re-estimate by OLS on all
     target months observed by t (index ≤ t-2) and predict u_t.  Specification is
     frozen; only coefficients are re-estimated.
  5. RMSE (full & ex-COVID), Diebold–Mariano and Clark–West vs the AR-only direct
     benchmark; reference RMSEs for AR(1) and the B5 state-space model if available.

Output: ardl_selection.csv, ardl_forecasts.csv  + printed tables (+ plots via plt.show)
"""

import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy.stats import norm

from statsmodels.tsa.seasonal import STL
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoCV

warnings.filterwarnings("ignore")
HERE = Path(__file__).parent

# ── Configuration ──────────────────────────────────────────────────────────────
FREEZE_DATE = pd.Timestamp("2017-12-31")
EVAL_START  = pd.Timestamp("2018-01-01")
EVAL_END    = pd.Timestamp("2025-12-01")
COVID_START = pd.Timestamp("2020-04-01")
COVID_END   = pd.Timestamp("2021-12-31")
PUB_LAG     = 2                 # u_{t-1}, u_t unobserved at origin t
N_TOP_PCS   = 4
MIN_KW      = 3
P_GRID      = [1, 2, 3, 4]      # number of unemployment lags (lags t-2 .. t-(p+1))
K_GRID      = [0, 1, 2, 3]      # max Google-factor lag (k = 0 .. K)
HAC_LAGS    = PUB_LAG - 1       # MA(h-1) structure of direct h-step errors


# ── Frozen preprocessing → VPCs (no-LASSO, all 21 keywords; see 02_pca_varimax_factors.py) ──
def varimax(loadings, max_iter=1000, tol=1e-6):
    Phi = loadings.T; p, k = Phi.shape; R = np.eye(k)
    for _ in range(max_iter):
        Lam = Phi @ R
        grad = Phi.T @ (Lam**3 - Lam * (Lam**2).sum(axis=0) / p)
        U, _, Vt = np.linalg.svd(grad)
        Rn = U @ Vt
        if np.max(np.abs(Rn - R)) < tol:
            R = Rn; break
        R = Rn
    return (Phi @ R).T, R


def frozen_deseas(series, freeze):
    train = series[series.index <= freeze].dropna()
    if len(train) < 24:
        return series
    stl = STL(train, period=12, robust=False).fit()
    pattern = stl.seasonal.groupby(stl.seasonal.index.month).mean()
    seas = pd.Series(series.index.month, index=series.index).map(pattern).values
    return series - seas


print("Loading data & building frozen VPCs …")
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
diff = deseas.diff(12).dropna()
train_mask = diff.index <= FREEZE_DATE

sc_pre = StandardScaler().fit(diff[train_mask])
X_pre = sc_pre.transform(diff[train_mask])
u_lasso = u_full.reindex(diff[train_mask].index)
ok = u_lasso.notna().values
lasso = LassoCV(cv=5, max_iter=10_000, random_state=42, n_jobs=-1).fit(X_pre[ok], u_lasso.dropna().values)
sel = diff.columns[np.abs(lasso.coef_) > 0].tolist()
if len(sel) < MIN_KW:
    corr = [abs(np.corrcoef(X_pre[ok][:, i], u_lasso.dropna().values)[0, 1]) for i in range(X_pre.shape[1])]
    sel = diff.columns[np.argsort(corr)[::-1][:MIN_KW]].tolist()
sel = list(diff.columns)   # NO-LASSO robustness: keep all 21 keywords
print(f"  Using {len(sel)} keywords (no-LASSO)")

sc_sel = StandardScaler().fit(diff[sel][train_mask])
pca = PCA(n_components=min(N_TOP_PCS, len(sel) - 1)).fit(sc_sel.transform(diff[sel][train_mask]))
rot, R_var = varimax(pca.components_)
scores = pca.transform(sc_sel.transform(diff[sel])) @ R_var
vpc = pd.DataFrame(scores, index=diff.index,
                   columns=[f"VPC{i+1}" for i in range(pca.n_components_)])
# sign-anchor on training correlation with u
u_tr = u_full.reindex(vpc[train_mask].index).dropna()
for c in vpc.columns:
    if np.corrcoef(vpc.loc[train_mask, c].reindex(u_tr.index).values, u_tr.values)[0, 1] < 0:
        vpc[c] = -vpc[c]
ALL_PCS = list(vpc.columns)
print(f"  factors available: {ALL_PCS}")


# ── ARDL design matrix ──────────────────────────────────────────────────────────
def build_design(pcs, p, K):
    """y=u_t and regressors u_{t-2..t-(p+1)}, F_{j,t-k} (k=0..K). Monthly index."""
    idx = u_full.index.union(vpc.index)
    u = u_full.reindex(idx)
    d = pd.DataFrame({"y": u}, index=idx)
    for l in range(2, p + 2):                 # ℓ = 2 .. p+1  (publication lag)
        d[f"u_l{l}"] = u.shift(l)
    for j in pcs:
        fj = vpc[j].reindex(idx)
        for k in range(0, K + 1):             # k = 0 .. K  (contemporaneous allowed)
            d[f"{j}_l{k}"] = fj.shift(k)
    feat = [c for c in d.columns if c != "y"]
    return d, feat


def ols_bic_train(pcs, p, K):
    """BIC of the ARDL on the frozen training window (≤ FREEZE)."""
    d, feat = build_design(pcs, p, K)
    tr = d[d.index <= FREEZE_DATE].dropna()
    if len(tr) <= len(feat) + 2:
        return np.inf
    res = sm.OLS(tr["y"].values, sm.add_constant(tr[feat].values)).fit()
    return res.bic


# ── Step 1: lag selection (all PCs) ──────────────────────────────────────────────
print("\n" + "=" * 64)
print("STEP 1  Lag selection by BIC on training window (all PCs)")
print("=" * 64)
print(f"{'p (u-lags)':>11} {'K (F-lags)':>11} {'#params':>8} {'BIC':>10}")
grid = []
for p in P_GRID:
    for K in K_GRID:
        d, feat = build_design(ALL_PCS, p, K)
        b = ols_bic_train(ALL_PCS, p, K)
        grid.append((p, K, len(feat) + 1, b))
        print(f"{p:>11} {K:>11} {len(feat)+1:>8} {b:>10.2f}")
p_star, K_star, _, _ = min(grid, key=lambda r: r[3])
print(f"\n  → selected (p, K) = ({p_star}, {K_star})")


# ── Step 2: PC selection — drop them one by one (general-to-specific) ────────────
print("\n" + "=" * 64)
print(f"STEP 2  General-to-specific PC elimination at (p,K)=({p_star},{K_star})")
print("=" * 64)
current = list(ALL_PCS)
path = [("ALL: " + "+".join(current), ols_bic_train(current, p_star, K_star))]
print(f"  start  {path[0][0]:<28}  BIC={path[0][1]:.2f}")
while len(current) > 0:
    base = ols_bic_train(current, p_star, K_star)
    trials = [( [x for x in current if x != pc], pc) for pc in current]
    cand = [(ols_bic_train(t, p_star, K_star), t, pc) for t, pc in trials]
    best_bic, best_set, dropped = min(cand, key=lambda r: r[0])
    if best_bic < base - 1e-6:
        current = best_set
        lbl = "+".join(current) if current else "(AR-only)"
        path.append((f"drop {dropped} → {lbl}", best_bic))
        print(f"  drop {dropped:<6} → {lbl:<22}  BIC={best_bic:.2f}")
    else:
        print(f"  no further drop improves BIC (stop). retained: {'+'.join(current)}")
        break
SEL_PCS = current
print(f"\n  → retained PCs: {SEL_PCS if SEL_PCS else '(none → pure AR)'}")


# ── Recursive pseudo-OOS ────────────────────────────────────────────────────────
def recursive_ardl(pcs, p, K):
    d, feat = build_design(pcs, p, K)
    eval_dates = u_full[(u_full.index >= EVAL_START) & (u_full.index <= EVAL_END)].index
    out = pd.Series(index=eval_dates, dtype=float)
    for t in eval_dates:
        cutoff = t - pd.DateOffset(months=PUB_LAG)        # last target observed at origin t
        tr = d[d.index <= cutoff].dropna()
        xrow = d.loc[[t], feat]
        if len(tr) < len(feat) + 5 or xrow.isna().any(axis=1).iloc[0]:
            continue
        Xtr = np.column_stack([np.ones(len(tr)), tr[feat].values])
        beta, *_ = np.linalg.lstsq(Xtr, tr["y"].values, rcond=None)
        out[t] = float(np.r_[1.0, xrow.values.ravel()] @ beta)
    return out

print("\nRunning recursive pseudo-OOS (2018–2025, h=2) …")
fc_ar   = recursive_ardl([], p_star, K_star)                       # AR-only direct benchmark
fc_all  = recursive_ardl(ALL_PCS, p_star, K_star)                  # full ARDL (all PCs)
fc_sel  = recursive_ardl(SEL_PCS, p_star, K_star) if SEL_PCS else fc_ar.copy()

actual = u_full.reindex(fc_all.index)
covid  = (actual.index >= COVID_START) & (actual.index <= COVID_END)


# ── Evaluation ───────────────────────────────────────────────────────────────────
def rmse(f, excl=False):
    m = actual.notna() & f.notna()
    if excl:
        m &= ~covid
    return float(np.sqrt(np.mean((actual[m] - f[m]) ** 2)))

def dm_test(f1, f2):                       # f1 vs f2 ; >0 means f1 worse
    m = actual.notna() & f1.notna() & f2.notna()
    d = (actual[m] - f1[m]) ** 2 - (actual[m] - f2[m]) ** 2
    r = sm.OLS(d.values, np.ones((len(d), 1))).fit(cov_type="HAC",
                                                   cov_kwds={"maxlags": HAC_LAGS})
    return float(r.tvalues[0]), float(r.pvalues[0])

def cw_test(f_restr, f_unr):               # nested: ARDL vs AR-only ; one-sided
    m = actual.notna() & f_restr.notna() & f_unr.notna()
    y, fr, fu = actual[m].values, f_restr[m].values, f_unr[m].values
    fterm = (y - fr) ** 2 - ((y - fu) ** 2 - (fr - fu) ** 2)
    r = sm.OLS(fterm, np.ones((len(fterm), 1))).fit(cov_type="HAC",
                                                    cov_kwds={"maxlags": HAC_LAGS})
    stat = float(r.tvalues[0])
    return stat, float(1 - norm.cdf(stat))

# reference RMSEs from existing engines (if their forecast files are present)
ref = {}
for fn, cols in [("option_b_lag2_forecasts.csv", {"AR(1)": "AR1", "B5 (state space)": "B5_VPC1_AR1"}),
                 ("tv_covid_forecasts.csv",      {"TV3 (state space)": "TV3_VPC1_RW"})]:
    fp = HERE / fn
    if fp.exists():
        ext = pd.read_csv(fp); ext["Tarih"] = pd.to_datetime(ext["Tarih"]); ext = ext.set_index("Tarih")
        for lbl, col in cols.items():
            if col in ext.columns:
                s = ext[col].reindex(actual.index)
                ref[lbl] = (rmse(s), rmse(s, True))

print("\n" + "=" * 78)
print(f"RESULTS — recursive OOS, n_full={int((actual.notna()&fc_all.notna()).sum())}, "
      f"n_exCOVID={int((actual.notna()&fc_all.notna()&~covid).sum())}")
print("=" * 78)
print(f"{'Model':<34}{'RMSE full':>11}{'RMSE exC':>11}{'DM vs AR':>11}{'CW vs AR':>11}")
print("-" * 78)
def line(lbl, f, nested=False):
    dm = dm_test(fc_ar, f)                  # AR worse than f  → positive favours f
    cw = cw_test(fc_ar, f) if nested else (np.nan, np.nan)
    dm_s = f"{dm[0]:+.2f}{'*' if dm[1]<0.05 else ''}"
    cw_s = f"{cw[0]:+.2f}{'**' if (not np.isnan(cw[1]) and cw[1]<0.05) else ('*' if (not np.isnan(cw[1]) and cw[1]<0.10) else '')}" if nested else "—"
    print(f"{lbl:<34}{rmse(f):>11.4f}{rmse(f,True):>11.4f}{dm_s:>11}{cw_s:>11}")

print(f"{'AR-only direct (p='+str(p_star)+', benchmark)':<34}{rmse(fc_ar):>11.4f}{rmse(fc_ar,True):>11.4f}{'—':>11}{'—':>11}")
line(f"ARDL all PCs (p={p_star},K={K_star})", fc_all, nested=True)
line(f"ARDL selected {'+'.join(SEL_PCS) if SEL_PCS else 'AR'} (p={p_star},K={K_star})", fc_sel, nested=True)
print("-" * 78)
for lbl, (rf, re) in ref.items():
    print(f"{'[ref] '+lbl:<34}{rf:>11.4f}{re:>11.4f}{'—':>11}{'—':>11}")
print("=" * 78)
print("DM: Diebold–Mariano t-stat (HAC, h-1 lags), >0 ⇒ ARDL beats AR-only; * p<.05.")
print("CW: Clark–West one-sided for the nested AR-only vs ARDL; * p<.10, ** p<.05.")


# ── Save outputs ─────────────────────────────────────────────────────────────────
pd.DataFrame(grid, columns=["p", "K", "n_params", "BIC_train"]).to_csv(HERE / "ardl_selection.csv", index=False)
fc = pd.DataFrame({"actual": actual,
                   "ARDL_all": fc_all,
                   "ARDL_sel": fc_sel,
                   "AR_direct": fc_ar})
fc.index.name = "Tarih"
fc.to_csv(HERE / "ardl_forecasts.csv")
print("\nSaved: ardl_selection.csv, ardl_forecasts.csv")


# ── Plot (displayed, not saved) ──────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(11, 4.4))
ax.axvspan(COVID_START, COVID_END, color="gold", alpha=0.18)
ax.plot(actual.index, actual.values, color="black", lw=2.0, label="Actual $u_t$")
ax.plot(fc_ar.index,  fc_ar.values,  color="#7F8C8D", lw=1.3, ls="--", label=f"AR-only direct (p={p_star})")
ax.plot(fc_all.index, fc_all.values, color="#00467F", lw=1.6, label=f"ARDL all PCs")
ax.plot(fc_sel.index, fc_sel.values, color="#C0392B", lw=1.6, ls="-.",
        label=f"ARDL selected ({'+'.join(SEL_PCS) if SEL_PCS else 'AR'})")
ax.set_ylabel("Non-agricultural unemployment (%)")
ax.set_title(f"Direct factor-augmented ARDL  (p={p_star}, K={K_star})  vs  AR-only")
ax.legend(loc="upper right"); ax.grid(axis="y", ls="--", alpha=0.35)
plt.tight_layout()
plt.show()
