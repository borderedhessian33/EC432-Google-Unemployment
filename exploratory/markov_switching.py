"""
EC 432 Project — Markov-switching characterization of the search-informative regime
====================================================================================
Descriptive, FULL-SAMPLE / full-information complement to the out-of-sample
results.  It asks *when* the labour search factor co-moves with unemployment by
letting that relationship switch between two latent regimes.

Model (Markov-switching regression, 2 regimes):
    u_t = c_{S_t} + rho * u_{t-1} + beta_{S_t} * VPC3_t + eps,  eps ~ N(0, sigma2_{S_t})
  - the AR(1) persistence term (rho) is held common across regimes;
  - the labour-factor loading beta, the intercept, and the variance switch with
    the latent state S_t in {informative, uninformative};
  - S_t follows a first-order Markov chain.

This is NOT a forecasting model: both regimes are estimated with the whole sample
(smoothed inference).  Factors are the locked no-LASSO VPCs (frozen Dec-2017).

Output: printed regime estimates + transition matrix + smoothed-probability figure.
"""

import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression
from statsmodels.tsa.seasonal import STL
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent          # exploratory/ — this script's own outputs
ROOT = HERE.parent                              # project root — shared input CSVs
FREEZE = pd.Timestamp("2017-12-31")
COVID_START, COVID_END = pd.Timestamp("2020-04-01"), pd.Timestamp("2021-12-01")
N_TOP_PCS = 4


def varimax(L, max_iter=1000, tol=1e-6):
    Phi = L.T; p, k = Phi.shape; R = np.eye(k)
    for _ in range(max_iter):
        Lam = Phi @ R; g = Phi.T @ (Lam**3 - Lam * (Lam**2).sum(0) / p)
        U, _, Vt = np.linalg.svd(g); Rn = U @ Vt
        if np.max(np.abs(Rn - R)) < tol: R = Rn; break
        R = Rn
    return (Phi @ R).T, R

def frozen_deseas(s, fz):
    tr = s[s.index <= fz].dropna()
    if len(tr) < 24: return s
    st = STL(tr, period=12, robust=False).fit(); pat = st.seasonal.groupby(st.seasonal.index.month).mean()
    return s - pd.Series(s.index.month, index=s.index).map(pat).values


# ── Data + frozen no-LASSO labour factor (VPC3) ─────────────────────────────────
trends = pd.read_csv(ROOT / "calibrated_trends_custom.csv", index_col=0, parse_dates=True).resample("MS").first()
unemp = pd.read_csv(ROOT / "unemp_csv.csv", sep=";", decimal=",", encoding="utf-8-sig", index_col=0)
unemp.columns = ["u"]; unemp = unemp[unemp.index.notna() & (unemp.index.astype(str) != "nan")]
unemp.index = pd.to_datetime(unemp.index, format="%Y-%m")
u_full = unemp.resample("MS").first()["u"].astype(float).dropna()

ds = pd.DataFrame(index=trends.index, columns=trends.columns, dtype=float)
for c in trends.columns: ds[c] = frozen_deseas(trends[c].astype(float), FREEZE)
diff = ds.diff(12).dropna(); tm = diff.index <= FREEZE
sc = StandardScaler().fit(diff[tm].values)
pca = PCA(n_components=N_TOP_PCS).fit(sc.transform(diff[tm].values))
rot, Rv = varimax(pca.components_)
VPC = pd.DataFrame(pca.transform(sc.transform(diff.values)) @ Rv, index=diff.index,
                   columns=[f"VPC{i+1}" for i in range(N_TOP_PCS)])
ytr = u_full.reindex(diff.index[tm]); ok = ytr.notna()
for c in VPC.columns:
    if np.corrcoef(VPC.loc[diff.index[tm], c].values[ok.values], ytr[ok].values)[0, 1] < 0:
        VPC[c] = -VPC[c]


# ── Markov-switching regression (variables standardised for conditioning) ────────
# Co-movement specification: regress the YoY change in unemployment on the labour
# factor, letting the factor loading switch.  Differencing removes the near-unit-root
# persistence that otherwise swamps the factor; the regimes then capture *when* the
# search factor co-moves with unemployment rather than mere volatility shifts.
du  = (u_full.diff(12)).rename("du")          # YoY change: smoother, matches the YoY-diff factor
duz = (du - du.mean()) / du.std()
vz  = (VPC["VPC3"] - VPC["VPC3"].mean()) / VPC["VPC3"].std()
df = pd.DataFrame({"du": duz, "VPC3": vz}).dropna()
exog = df[["VPC3"]]
mod = MarkovRegression(df["du"], k_regimes=2, trend="c", exog=exog,
                       switching_trend=True,           # intercept switches
                       switching_exog=True,            # factor loading switches
                       switching_variance=True)
np.random.seed(42)
res = mod.fit()

print(res.summary())

# regime-specific labour-factor loading
beta = [res.params[f"x1[{r}]"] for r in range(2)]   # VPC3 is the only exog -> 'x1'
informative = int(np.argmax(np.abs(beta)))               # regime with the larger |beta|
P = res.regime_transition[:, :, 0]
dur = [1.0 / (1.0 - P[r, r]) for r in range(2)]
print("\n" + "=" * 60)
print(f"Labour-factor loading beta:  regime0 = {beta[0]:+.3f},  regime1 = {beta[1]:+.3f}")
print(f"=> 'search-informative' regime = regime {informative}")
print(f"Transition matrix P:\n{np.round(P,3)}")
print(f"Expected durations (months): regime0 = {dur[0]:.1f}, regime1 = {dur[1]:.1f}")
print("=" * 60)


# ── Smoothed probability of the search-informative regime ────────────────────────
prob = res.smoothed_marginal_probabilities[informative]
print("\nP(search-informative regime), annual mean:")
print(prob.groupby(prob.index.year).mean().round(2).to_string())
prob.rename("p_informative").rename_axis("date").to_csv(HERE / "markov_regime_prob.csv")
print("Saved: markov_regime_prob.csv")
fig, ax = plt.subplots(figsize=(10, 4.2))
ax.axvspan(COVID_START, COVID_END, color="gold", alpha=0.12, zorder=0, label="COVID window")
ax.axvline(FREEZE, color="grey", ls="--", lw=1, zorder=1, label="Preprocessing freeze")
ax.fill_between(prob.index, 0, prob.values, color="#00467F", alpha=0.30, zorder=2)
ax.plot(prob.index, prob.values, color="#00467F", lw=1.8, zorder=3,
        label="P(search-informative regime)")
ax.set_ylim(0, 1.02); ax.set_ylabel("Smoothed probability")
ax.legend(frameon=False, ncol=3, fontsize=9, loc="lower left")
ax.spines[["top", "right"]].set_visible(False)
ax.grid(axis="y", ls="--", alpha=0.3)
ax.margins(x=0.01)
plt.tight_layout()
plt.show()
