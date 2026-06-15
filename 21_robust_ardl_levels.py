"""
EC 432 Project — Direct-Horizon Factor-Augmented ARDL on DESEASONALISED-LEVEL PCs
==================================================================================
Same direct ARDL as 20_robust_ardl_direct.py, but the factors are built WITHOUT
the 12-month (YoY) difference — the keywords are only deseasonalised (frozen STL),
standardised, then PCA → varimax on ALL 21 keywords (no LASSO).  So VPC1..VPC4
here are *level* factors rather than growth factors.

    u_t = α + Σ_{ℓ=2}^{p+1} φ_ℓ u_{t-ℓ} + Σ_{j∈S} Σ_{k=0}^{K} β_{jk} F_{j,t-k} + ε_t

NOTE: deseasonalised search *levels* are highly persistent (trending); the factors
and u therefore share low-frequency trends, so in-sample fit can look strong for
spurious reasons.  The recursive OOS RMSE (below) is the honest check.

Procedure: BIC lag selection over (p,K) with all PCs → general-to-specific PC
elimination → recursive pseudo-OOS (2018–2025, h=2) vs AR-only direct benchmark
with DM and Clark–West.  Preprocessing frozen at Dec-2017.

Output: ardl_levels_selection.csv, ardl_levels_forecasts.csv
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

FREEZE_DATE = pd.Timestamp("2017-12-31")
EVAL_START  = pd.Timestamp("2018-01-01")
EVAL_END    = pd.Timestamp("2025-12-01")
COVID_START = pd.Timestamp("2020-04-01")
COVID_END   = pd.Timestamp("2021-12-31")
PUB_LAG     = 2
N_TOP_PCS   = 4
MIN_KW      = 3
P_GRID      = [1, 2, 3, 4]
K_GRID      = [0, 1, 2, 3]
HAC_LAGS    = PUB_LAG - 1


def varimax(loadings, max_iter=1000, tol=1e-6):
    Phi = loadings.T; p, k = Phi.shape; R = np.eye(k)
    for _ in range(max_iter):
        Lam = Phi @ R
        grad = Phi.T @ (Lam**3 - Lam * (Lam**2).sum(axis=0) / p)
        U, _, Vt = np.linalg.svd(grad); Rn = U @ Vt
        if np.max(np.abs(Rn - R)) < tol:
            R = Rn; break
        R = Rn
    return (Phi @ R).T, R


def frozen_deseas(s, fz):
    tr = s[s.index <= fz].dropna()
    if len(tr) < 24:
        return s
    stl = STL(tr, period=12, robust=False).fit()
    pat = stl.seasonal.groupby(stl.seasonal.index.month).mean()
    return s - pd.Series(s.index.month, index=s.index).map(pat).values


print("Loading data & building frozen LEVEL VPCs (no YoY difference) …")
trends = pd.read_csv(HERE / "calibrated_trends_custom.csv", index_col=0,
                     parse_dates=True).resample("MS").first()
unemp = pd.read_csv(HERE / "unemp_csv.csv", sep=";", decimal=",",
                    encoding="utf-8-sig", index_col=0, parse_dates=False)
unemp.columns = ["u"]
unemp = unemp[unemp.index.notna() & (unemp.index.astype(str) != "nan")]
unemp.index = pd.to_datetime(unemp.index, format="%Y-%m")
u_full = unemp.resample("MS").first()["u"].astype(float).dropna()

deseas = pd.DataFrame(index=trends.index, columns=trends.columns, dtype=float)
for c in trends.columns:
    deseas[c] = frozen_deseas(trends[c].astype(float), FREEZE_DATE)
lvl = deseas.dropna()                    # <<< deseasonalised LEVELS (no .diff(12))
tm = lvl.index <= FREEZE_DATE

sc_pre = StandardScaler().fit(lvl[tm])
Xp = sc_pre.transform(lvl[tm])
ul = u_full.reindex(lvl[tm].index); ok = ul.notna().values
sel = lvl.columns[np.abs(LassoCV(cv=5, max_iter=10000, random_state=42)
                         .fit(Xp[ok], ul.dropna().values).coef_) > 0].tolist()
if len(sel) < MIN_KW:
    corr = [abs(np.corrcoef(Xp[ok][:, i], ul.dropna().values)[0, 1]) for i in range(Xp.shape[1])]
    sel = lvl.columns[np.argsort(corr)[::-1][:MIN_KW]].tolist()
sel = list(lvl.columns)   # NO-LASSO robustness: keep all 21 keywords (levels)
print(f"  Using {len(sel)} keywords (no-LASSO, levels)")

ss = StandardScaler().fit(lvl[sel][tm])
pca = PCA(n_components=min(N_TOP_PCS, len(sel) - 1)).fit(ss.transform(lvl[sel][tm]))
rot, Rv = varimax(pca.components_)
vpc = pd.DataFrame(pca.transform(ss.transform(lvl[sel])) @ Rv, index=lvl.index,
                   columns=[f"VPC{i+1}" for i in range(pca.n_components_)])
u_tr = u_full.reindex(vpc[tm].index).dropna()
for c in vpc.columns:
    if np.corrcoef(vpc.loc[tm, c].reindex(u_tr.index).values, u_tr.values)[0, 1] < 0:
        vpc[c] = -vpc[c]
ALL_PCS = list(vpc.columns)
print(f"  factors available: {ALL_PCS}")


# ── ARDL design ──────────────────────────────────────────────────────────────────
ADD_COVID = True     # include a COVID dummy (Apr-2020..Dec-2021) in every model

def build_design(pcs, p, K):
    idx = u_full.index.union(vpc.index)
    u = u_full.reindex(idx)
    d = pd.DataFrame({"y": u}, index=idx)
    for l in range(2, p + 2):
        d[f"u_l{l}"] = u.shift(l)
    for j in pcs:
        fj = vpc[j].reindex(idx)
        for k in range(0, K + 1):
            d[f"{j}_l{k}"] = fj.shift(k)
    if ADD_COVID:
        d["covid"] = ((idx >= COVID_START) & (idx <= COVID_END)).astype(float)
    feat = [c for c in d.columns if c != "y"]
    return d, feat

def ols_bic_train(pcs, p, K):
    d, feat = build_design(pcs, p, K)
    tr = d[d.index <= FREEZE_DATE].dropna()
    if len(tr) <= len(feat) + 2:
        return np.inf
    return sm.OLS(tr["y"].values, sm.add_constant(tr[feat].values)).fit().bic


# ── Step 1: lag selection ────────────────────────────────────────────────────────
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


# ── Step 2: general-to-specific PC elimination ──────────────────────────────────
print("\n" + "=" * 64)
print(f"STEP 2  General-to-specific PC elimination at (p,K)=({p_star},{K_star})")
print("=" * 64)
current = list(ALL_PCS)
print(f"  start  ALL: {'+'.join(current):<24}  BIC={ols_bic_train(current, p_star, K_star):.2f}")
while len(current) > 0:
    base = ols_bic_train(current, p_star, K_star)
    cand = [(ols_bic_train([x for x in current if x != pc], p_star, K_star),
             [x for x in current if x != pc], pc) for pc in current]
    best_bic, best_set, dropped = min(cand, key=lambda r: r[0])
    if best_bic < base - 1e-6:
        current = best_set
        print(f"  drop {dropped:<6} → {('+'.join(current) or '(AR-only)'):<22}  BIC={best_bic:.2f}")
    else:
        print(f"  no further drop improves BIC (stop). retained: {'+'.join(current)}")
        break
SEL_PCS = current
print(f"\n  → retained PCs: {SEL_PCS if SEL_PCS else '(none → pure AR)'}")


# ── Recursive OOS ────────────────────────────────────────────────────────────────
def recursive_ardl(pcs, p, K):
    d, feat = build_design(pcs, p, K)
    eval_dates = u_full[(u_full.index >= EVAL_START) & (u_full.index <= EVAL_END)].index
    out = pd.Series(index=eval_dates, dtype=float)
    for t in eval_dates:
        cutoff = t - pd.DateOffset(months=PUB_LAG)
        tr = d[d.index <= cutoff].dropna()
        xrow = d.loc[[t], feat]
        if len(tr) < len(feat) + 5 or xrow.isna().any(axis=1).iloc[0]:
            continue
        X = np.column_stack([np.ones(len(tr)), tr[feat].values])
        beta, *_ = np.linalg.lstsq(X, tr["y"].values, rcond=None)
        out[t] = float(np.r_[1.0, xrow.values.ravel()] @ beta)
    return out

print("\nRunning recursive pseudo-OOS (2018–2025, h=2) …")
fc_ar  = recursive_ardl([], p_star, K_star)
fc_all = recursive_ardl(ALL_PCS, p_star, K_star)
fc_sel = recursive_ardl(SEL_PCS, p_star, K_star) if SEL_PCS else fc_ar.copy()
actual = u_full.reindex(fc_all.index)
covid  = (actual.index >= COVID_START) & (actual.index <= COVID_END)


# ── Evaluation ────────────────────────────────────────────────────────────────────
def rmse(f, excl=False):
    m = actual.notna() & f.notna()
    if excl:
        m &= ~covid
    return float(np.sqrt(np.mean((actual[m] - f[m]) ** 2)))

def dm_test(f1, f2):
    m = actual.notna() & f1.notna() & f2.notna()
    d = (actual[m] - f1[m]) ** 2 - (actual[m] - f2[m]) ** 2
    r = sm.OLS(d.values, np.ones((len(d), 1))).fit(cov_type="HAC", cov_kwds={"maxlags": HAC_LAGS})
    return float(r.tvalues[0]), float(r.pvalues[0])

def cw_test(f_restr, f_unr):
    m = actual.notna() & f_restr.notna() & f_unr.notna()
    y, fr, fu = actual[m].values, f_restr[m].values, f_unr[m].values
    fterm = (y - fr) ** 2 - ((y - fu) ** 2 - (fr - fu) ** 2)
    r = sm.OLS(fterm, np.ones((len(fterm), 1))).fit(cov_type="HAC", cov_kwds={"maxlags": HAC_LAGS})
    return float(r.tvalues[0]), float(1 - norm.cdf(r.tvalues[0]))

ref = {}
for fn, cols in [("option_b_lag2_forecasts.csv", {"AR(1) recursive": "AR1", "B5 state space": "B5_VPC1_AR1"}),
                 ("ardl_forecasts.csv",          {"ARDL (YoY-diff PCs)": "ARDL_sel"})]:
    fp = HERE / fn
    if fp.exists():
        e = pd.read_csv(fp); e[e.columns[0]] = pd.to_datetime(e[e.columns[0]]); e = e.set_index(e.columns[0])
        for lbl, col in cols.items():
            if col in e.columns:
                s = e[col].reindex(actual.index); ref[lbl] = (rmse(s), rmse(s, True))

print("\n" + "=" * 80)
print(f"RESULTS — ARDL on deseasonalised-LEVEL PCs (p={p_star}, K={K_star}), "
      f"n_full={int((actual.notna()&fc_all.notna()).sum())}")
print("=" * 80)
print(f"{'Model':<32}{'RMSE full':>11}{'RMSE exC':>11}{'DM vs AR':>11}{'CW vs AR':>11}")
print("-" * 80)
print(f"{'AR-only direct (benchmark)':<32}{rmse(fc_ar):>11.4f}{rmse(fc_ar,True):>11.4f}{'—':>11}{'—':>11}")
for lbl, f in [(f"ARDL all PCs", fc_all),
               (f"ARDL selected {'+'.join(SEL_PCS) if SEL_PCS else 'AR'}", fc_sel)]:
    dm = dm_test(fc_ar, f); cw = cw_test(fc_ar, f)
    dm_s = f"{dm[0]:+.2f}{'*' if dm[1]<0.05 else ''}"
    cw_s = f"{cw[0]:+.2f}{'**' if cw[1]<0.05 else ('*' if cw[1]<0.10 else '')}"
    print(f"{lbl:<32}{rmse(f):>11.4f}{rmse(f,True):>11.4f}{dm_s:>11}{cw_s:>11}")
print("-" * 80)
for lbl, (rf, re) in ref.items():
    print(f"{'[ref] '+lbl:<32}{rf:>11.4f}{re:>11.4f}{'—':>11}{'—':>11}")
print("=" * 80)
print("DM>0 ⇒ ARDL beats AR-only (HAC, h-1 lags); * p<.05.  CW one-sided; * p<.10, ** p<.05.")

pd.DataFrame(grid, columns=["p", "K", "n_params", "BIC_train"]).to_csv(HERE / "ardl_levels_selection.csv", index=False)
pd.DataFrame({"actual": actual, "ARDL_all": fc_all, "ARDL_sel": fc_sel, "AR_direct": fc_ar}
             ).rename_axis("Tarih").to_csv(HERE / "ardl_levels_forecasts.csv")
print("\nSaved: ardl_levels_selection.csv, ardl_levels_forecasts.csv")

fig, ax = plt.subplots(figsize=(11, 4.4))
ax.axvspan(COVID_START, COVID_END, color="gold", alpha=0.18)
ax.plot(actual.index, actual.values, color="black", lw=2.0, label="Actual $u_t$")
ax.plot(fc_ar.index,  fc_ar.values,  color="#7F8C8D", lw=1.3, ls="--", label=f"AR-only direct (p={p_star})")
ax.plot(fc_all.index, fc_all.values, color="#00467F", lw=1.6, label="ARDL all level-PCs")
ax.plot(fc_sel.index, fc_sel.values, color="#C0392B", lw=1.6, ls="-.",
        label=f"ARDL selected ({'+'.join(SEL_PCS) if SEL_PCS else 'AR'})")
ax.set_ylabel("Non-agricultural unemployment (%)")
ax.set_title(f"ARDL on deseasonalised-level PCs (p={p_star}, K={K_star})")
ax.legend(loc="upper right"); ax.grid(axis="y", ls="--", alpha=0.35)
plt.tight_layout()
plt.show()
