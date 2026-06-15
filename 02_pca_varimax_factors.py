"""
EC 432 Project — Diagnostics for the no-LASSO PCA+varimax factor setup
======================================================================
Builds the factors WITHOUT LASSO: frozen deseasonalise → YoY diff → standardise
→ PCA(4) on ALL 21 keywords → varimax → sign-anchor (+corr with u).  Then shows
five diagnostic figures ON SCREEN (plt.show(); nothing written to disk):

  1. Scree            — variance explained per PC + cumulative
  2. Varimax loadings — 21 keywords × 4 rotated components (heatmap)
  3. Factor scores    — each VPC vs unemployment over time (2×2)
  4. |corr(VPC,u)|    — which rotated component is the labour-market factor
  5. Cross-correlation— lead/lag of the labour factor vs unemployment

Run:  python 02_pca_varimax_factors.py   (figures pop up one after another)
"""

import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import STL
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

warnings.filterwarnings("ignore")
HERE = Path(__file__).parent
FREEZE = pd.Timestamp("2017-12-31")
EVAL_START = pd.Timestamp("2018-01-01")
N_TOP_PCS = 4
BLUE, RED, GREEN, ORANGE = "#00467F", "#C0392B", "#27AE60", "#E67E22"


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
    if len(tr) < 24: return s
    stl = STL(tr, period=12, robust=False).fit()
    pat = stl.seasonal.groupby(stl.seasonal.index.month).mean()
    return s - pd.Series(s.index.month, index=s.index).map(pat).values


# ── Build factors (no LASSO, frozen at 2017) ────────────────────────────────────
trends = pd.read_csv(HERE / "calibrated_trends_custom.csv", index_col=0, parse_dates=True).resample("MS").first()
unemp = pd.read_csv(HERE / "unemp_csv.csv", sep=";", decimal=",",
                    encoding="utf-8-sig", index_col=0, parse_dates=False)
unemp.columns = ["u"]
unemp = unemp[unemp.index.notna() & (unemp.index.astype(str) != "nan")]
unemp.index = pd.to_datetime(unemp.index, format="%Y-%m")
u_full = unemp.resample("MS").first()["u"].astype(float).dropna()

deseas = pd.DataFrame(index=trends.index, columns=trends.columns, dtype=float)
for c in trends.columns:
    deseas[c] = frozen_deseas(trends[c].astype(float), FREEZE)
diff = deseas.diff(12).dropna()
kws = list(diff.columns)
tm = diff.index <= FREEZE
scaler = StandardScaler().fit(diff[tm].values)
pca = PCA(n_components=N_TOP_PCS).fit(scaler.transform(diff[tm].values))
rot, Rv = varimax(pca.components_)
scores = pca.transform(scaler.transform(diff.values)) @ Rv
VPC = pd.DataFrame(scores, index=diff.index, columns=[f"VPC{i+1}" for i in range(N_TOP_PCS)])
loadings = pd.DataFrame(rot.T, index=kws, columns=VPC.columns)   # 21 × 4
# variance explained by each rotated component (training)
rot_scores_tr = pca.transform(scaler.transform(diff[tm].values)) @ Rv
var_rot = rot_scores_tr.var(axis=0)
evr_rot = var_rot / diff.shape[1]                                # standardized total var = n_features
# sign-anchor each component to +corr with u (training)
ytr = u_full.reindex(diff.index[tm]); ok = ytr.notna()
corr_u = {}
for c in VPC.columns:
    cc = np.corrcoef(VPC.loc[diff.index[tm], c].values[ok.values], ytr[ok].values)[0, 1]
    if cc < 0:
        VPC[c] = -VPC[c]; loadings[c] = -loadings[c]; cc = -cc
    corr_u[c] = cc
labor = max(corr_u, key=corr_u.get)


# ── Printed diagnostics ──────────────────────────────────────────────────────────
print("=" * 64)
print("NO-LASSO PCA+varimax on all 21 keywords — numeric diagnostics")
print("=" * 64)
print("PCA explained-variance ratio (pre-rotation):",
      np.round(pca.explained_variance_ratio_, 3), " cum=", np.round(pca.explained_variance_ratio_.cumsum(), 3))
print("Variance share per VARIMAX component       :", np.round(evr_rot, 3))
print("\ncorr(VPC, u) on training:")
for c, v in corr_u.items():
    tag = "   <- labour factor" if c == labor else ""
    print(f"  {c}: {v:+.3f}{tag}")
print(f"\nLabour factor = {labor}.  Top keyword loadings:")
print(loadings[labor].sort_values(key=np.abs, ascending=False).head(8).round(3).to_string())
print(f"\nVPC1 (first component) top loadings:")
print(loadings['VPC1'].sort_values(key=np.abs, ascending=False).head(8).round(3).to_string())
print("=" * 64)


# ── FIGURE 1 — scree ─────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 4))
k = np.arange(1, len(pca.explained_variance_ratio_) + 1)
ax.bar(k, pca.explained_variance_ratio_, color=BLUE, alpha=0.8, label="per PC")
ax.plot(k, pca.explained_variance_ratio_.cumsum(), "o-", color=RED, label="cumulative")
ax.set_xlabel("principal component"); ax.set_ylabel("variance explained")
ax.set_title("Scree — PCA on all 21 keywords (pre-rotation)")
ax.set_xticks(k); ax.legend(); ax.grid(axis="y", ls="--", alpha=0.3)
plt.tight_layout(); plt.show()

# ── FIGURE 2 — varimax loadings heatmap ─────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 9))
im = ax.imshow(loadings.values, aspect="auto", cmap="RdBu_r", vmin=-0.6, vmax=0.6)
ax.set_xticks(range(N_TOP_PCS)); ax.set_xticklabels(VPC.columns)
ax.set_yticks(range(len(kws))); ax.set_yticklabels(kws, fontsize=8)
ax.set_title("Varimax loadings (21 keywords × rotated components)")
for i in range(len(kws)):
    for j in range(N_TOP_PCS):
        v = loadings.values[i, j]
        ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7,
                color="white" if abs(v) > 0.45 else "black")
fig.colorbar(im, ax=ax, shrink=0.5, label="loading")
plt.tight_layout(); plt.show()

# ── FIGURE 3 — factor scores vs unemployment (2×2) ──────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharex=True)
u_al = u_full.reindex(VPC.index)
for ax, c in zip(axes.ravel(), VPC.columns):
    ax.axvspan(pd.Timestamp("2020-04-01"), pd.Timestamp("2021-12-01"), color="gold", alpha=0.15)
    ax.axvline(FREEZE, color="grey", ls="--", lw=1)
    ax.plot(VPC.index, VPC[c], color=BLUE, lw=1.4, label=c)
    ax2 = ax.twinx()
    ax2.plot(u_al.index, u_al.values, color=RED, lw=1.4, alpha=0.8, label="u")
    ax.set_title(f"{c}   (corr with u = {corr_u[c]:+.2f})", fontsize=10)
    ax.tick_params(axis="y", labelcolor=BLUE); ax2.tick_params(axis="y", labelcolor=RED)
fig.suptitle("Rotated factor scores (blue) vs unemployment (red) — dashed=freeze, shaded=COVID")
plt.tight_layout(); plt.show()

# ── FIGURE 4 — |corr(VPC,u)| bar ────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6, 4))
cols = [GREEN if c == labor else BLUE for c in VPC.columns]
ax.bar(list(corr_u.keys()), list(corr_u.values()), color=cols)
ax.axhline(0.587, color=RED, ls="--", lw=1, label="LASSO-VPC1 (0.587)")
ax.set_ylabel("corr with unemployment (training)")
ax.set_title("Which rotated component is the labour factor?")
ax.legend(); ax.grid(axis="y", ls="--", alpha=0.3)
plt.tight_layout(); plt.show()

# ── FIGURE 5 — cross-correlation (lead/lag) of labour factor vs u ───────────────
fig, ax = plt.subplots(figsize=(7, 4))
x = VPC[labor].reindex(u_al.index); y = u_al
lags = range(-12, 13)
ccf = []
for L in lags:
    cc = x.corr(y.shift(-L))   # L>0: factor leads u by L months
    ccf.append(cc)
peakL = list(lags)[int(np.nanargmax(ccf))]
print(f"\nCCF: contemporaneous corr = {ccf[list(lags).index(0)]:+.3f}; "
      f"peak corr {np.nanmax(ccf):+.3f} at lag L={peakL} "
      f"({'factor leads u' if peakL>0 else 'contemporaneous/lagging'}).")
ax.bar(list(lags), ccf, color=GREEN, alpha=0.8)
ax.axvline(0, color="black", lw=0.8)
ax.set_xlabel("lag L  (L>0: factor leads unemployment by L months)")
ax.set_ylabel("correlation")
ax.set_title(f"Cross-correlation: {labor} (labour factor) vs unemployment")
ax.grid(axis="y", ls="--", alpha=0.3)
plt.tight_layout(); plt.show()

print("\nDisplayed 5 figures. (Run this file directly to view them.)")
