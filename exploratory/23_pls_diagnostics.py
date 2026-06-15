"""
EC 432 Project — PLS extraction diagnostics (no forecasting yet)
================================================================
Replaces the ad hoc LASSO → PCA → varimax → sign-anchor chain with a single
supervised step: Partial Least Squares (PLS) extracts the keyword combinations
that maximise COVARIANCE with unemployment.  This script only INSPECTS the PLS
factors on the training window (≤ Dec-2017) — it does not forecast.

Reported:
  • cumulative R²(u) of k PLS components vs k PCA components  (does PLS explain u
    with fewer components?)
  • X-variance captured per PLS component
  • component-1 keyword weights (which terms drive the targeted factor)
  • correlation of the PLS factor with u, and with the old VPC1
All on the frozen training window (no look-ahead).
"""

import warnings
from pathlib import Path
import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import STL
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import LinearRegression, LassoCV

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent          # exploratory/ — this script's own outputs
ROOT = HERE.parent                              # project root — shared input CSVs
FREEZE = pd.Timestamp("2017-12-31")


def frozen_deseas(s, fz):
    tr = s[s.index <= fz].dropna()
    if len(tr) < 24:
        return s
    stl = STL(tr, period=12, robust=False).fit()
    pat = stl.seasonal.groupby(stl.seasonal.index.month).mean()
    return s - pd.Series(s.index.month, index=s.index).map(pat).values

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


# ── Data ─────────────────────────────────────────────────────────────────────────
print("Loading keywords + unemployment …")
trends = pd.read_csv(ROOT / "calibrated_trends_custom.csv", index_col=0, parse_dates=True).resample("MS").first()
unemp = pd.read_csv(ROOT / "unemp_csv.csv", sep=";", decimal=",",
                    encoding="utf-8-sig", index_col=0, parse_dates=False)
unemp.columns = ["u"]
unemp = unemp[unemp.index.notna() & (unemp.index.astype(str) != "nan")]
unemp.index = pd.to_datetime(unemp.index, format="%Y-%m")
u_full = unemp.resample("MS").first()["u"].astype(float).dropna()

deseas = pd.DataFrame(index=trends.index, columns=trends.columns, dtype=float)
for c in trends.columns:
    deseas[c] = frozen_deseas(trends[c].astype(float), FREEZE)
diff = deseas.diff(12).dropna()                      # YoY-differenced (same as paper)

tmask = diff.index <= FREEZE
scaler = StandardScaler().fit(diff[tmask])
Xall = pd.DataFrame(scaler.transform(diff), index=diff.index, columns=diff.columns)

# training rows where u is observed
idx_tr = diff.index[tmask]
y_tr = u_full.reindex(idx_tr)
ok = y_tr.notna()
Xtr = Xall.loc[idx_tr][ok.values]
ytr = y_tr[ok].values
ytr_c = ytr - ytr.mean()
print(f"  training: {Xtr.index[0].date()} → {Xtr.index[-1].date()}, n={len(Xtr)}, keywords={Xtr.shape[1]}")


# ── 1. Cumulative R²(u): PLS vs PCA, component by component ──────────────────────
print("\n" + "=" * 60)
print("Cumulative in-sample R²(u) by #components  (training ≤2017)")
print("=" * 60)
print(f"{'k':>3}  {'PLS R²(u)':>10}  {'PCA R²(u)':>10}")
maxk = 6
pca_full = PCA(n_components=maxk).fit(Xtr.values)
pcs_full = pca_full.transform(Xtr.values)
for k in range(1, maxk + 1):
    pls = PLSRegression(n_components=k, scale=False).fit(Xtr.values, ytr_c)
    r2_pls = pls.score(Xtr.values, ytr_c)
    lr = LinearRegression().fit(pcs_full[:, :k], ytr_c)
    r2_pca = lr.score(pcs_full[:, :k], ytr_c)
    print(f"{k:>3}  {r2_pls:>10.3f}  {r2_pca:>10.3f}")
print("(PLS targets u, so it should reach a given R² with fewer components.)")


# ── 2. X-variance captured per PLS component ────────────────────────────────────
pls5 = PLSRegression(n_components=5, scale=False).fit(Xtr.values, ytr_c)
xscore_var = np.var(pls5.x_scores_, axis=0)
xvar_share = xscore_var / Xtr.shape[1]               # standardized X total var = n_features
print("\nX-variance share captured per PLS component:")
for i, v in enumerate(xvar_share, 1):
    corr_u = np.corrcoef(pls5.x_scores_[:, i - 1], ytr_c)[0, 1]
    print(f"  PLS{i}:  X-var = {100*v:5.1f}%   corr with u = {corr_u:+.3f}")


# ── 3. Component-1 keyword weights (the targeted factor) ────────────────────────
w1 = pd.Series(pls5.x_weights_[:, 0], index=Xtr.columns).sort_values(key=np.abs, ascending=False)
print("\nPLS component-1 weights — top keywords driving the targeted factor:")
for kw, w in w1.head(10).items():
    print(f"  {kw:<22} {w:+.3f}")


# ── 4. Compare to the old VPC1 (LASSO→PCA→varimax) ──────────────────────────────
las = LassoCV(cv=5, max_iter=10000, random_state=42).fit(Xtr.values, ytr)
sel = Xtr.columns[np.abs(las.coef_) > 0].tolist()
if len(sel) < 3:
    sel = list(Xtr.columns[:3])
Xsel_tr = Xtr[sel].values
pca_sel = PCA(n_components=min(4, len(sel) - 1)).fit(Xsel_tr)
rot, Rv = varimax(pca_sel.components_)
vpc_tr = pca_sel.transform(Xsel_tr) @ Rv
vpc1 = vpc_tr[:, 0]
if np.corrcoef(vpc1, ytr)[0, 1] < 0:
    vpc1 = -vpc1
pls1 = pls5.x_scores_[:, 0]
print("\n" + "=" * 60)
print("PLS factor 1  vs  old VPC1 (LASSO→PCA→varimax)")
print("=" * 60)
print(f"  LASSO kept {len(sel)} keywords for the PCA: {sel}")
print(f"  corr(PLS1, u)  = {np.corrcoef(pls1, ytr)[0,1]:+.3f}")
print(f"  corr(VPC1, u)  = {np.corrcoef(vpc1, ytr)[0,1]:+.3f}")
print(f"  corr(PLS1, VPC1) = {np.corrcoef(pls1, vpc1)[0,1]:+.3f}")
r2_pls1 = PLSRegression(n_components=1, scale=False).fit(Xtr.values, ytr_c).score(Xtr.values, ytr_c)
r2_vpc1 = LinearRegression().fit(vpc1.reshape(-1, 1), ytr_c).score(vpc1.reshape(-1, 1), ytr_c)
print(f"  R²(u) from 1 PLS comp = {r2_pls1:.3f}   |   R²(u) from VPC1 only = {r2_vpc1:.3f}")
print("=" * 60)

# save the full-sample PLS scores (frozen on training) for later use
pls_full = pls5.transform(Xall.values)
pd.DataFrame(pls_full, index=Xall.index,
             columns=[f"PLS{i+1}" for i in range(pls_full.shape[1])]
             ).rename_axis("date").to_csv(HERE / "pls_scores.csv")
print("\nSaved full-sample PLS scores (frozen on training) → pls_scores.csv")
