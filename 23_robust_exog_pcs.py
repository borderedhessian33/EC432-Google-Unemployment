"""
EC 432 Project — Option A: State-space regression with exogenous Google factors
================================================================================
A LIGHTER state space than the project's multivariate measurement model. Only
unemployment is observed; the PCs enter the *measurement* equation directly as
exogenous regressors (no second measurement system for the PCs):

    u_t   = μ_t + β' Z_t + ε^u_t          (measurement; Z_t = exogenous PC bridge)
    μ_t   = φ_1 μ_{t-1} + … + φ_p μ_{t-p} + η_t     (latent persistence)
    Z_t   = (F_t, F_{t-1}, …, F_{t-K})

    nowcast:  û_{t|t} = μ̂_{t|t} + β' Z_t

μ_t carries persistent unemployment dynamics; β'Z_t is the Google bridge specific
to unemployment.  Implemented with SARIMAX (AR(p) + measurement error + exog):
    SARIMAX(u, exog=Z, order=(p,0,0), measurement_error=True, trend='c')
which is exactly  u_t = c + β'Z_t + α_t + ε_t,  α_t ~ AR(p)  (α_t ≡ μ_t − c).

Lag/factor selection: per instruction we TRY A FEW specifications and keep the one
with the lowest recursive out-of-sample RMSE (full window).  Same frozen pipeline,
same 2-month publication lag, same expanding-window re-estimation as the project.

Output: statespace_exog_specs.csv, statespace_exog_forecasts.csv + printed tables.
"""

import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy.stats import norm
from statsmodels.tsa.statespace.sarimax import SARIMAX
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
HAC_LAGS    = PUB_LAG - 1
N_TOP_PCS   = 4
MIN_KW      = 3

# specifications to try  →  (PC list, p, K)
SPECS = [
    (["VPC1"],            1, 0),
    (["VPC1"],            2, 1),
    (["VPC1"],            2, 2),
    (["VPC1", "VPC3"],    2, 1),
    (["VPC1", "VPC3"],    2, 2),
    (["VPC1", "VPC2", "VPC3", "VPC4"], 2, 1),
]


# ── Frozen pipeline → VPCs (identical to 01/02/03/13) ───────────────────────────
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

print("Loading data & building frozen VPCs …")
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
diff = deseas.diff(12).dropna()
tm = diff.index <= FREEZE_DATE
Xp = StandardScaler().fit(diff[tm]).transform(diff[tm])
ul = u_full.reindex(diff[tm].index); ok = ul.notna().values
sel = diff.columns[np.abs(LassoCV(cv=5, max_iter=10000, random_state=42)
                          .fit(Xp[ok], ul.dropna().values).coef_) > 0].tolist()
if len(sel) < MIN_KW:
    corr = [abs(np.corrcoef(Xp[ok][:, i], ul.dropna().values)[0, 1]) for i in range(Xp.shape[1])]
    sel = diff.columns[np.argsort(corr)[::-1][:MIN_KW]].tolist()
sel = list(diff.columns)   # NO-LASSO robustness: keep all 21 keywords
ss = StandardScaler().fit(diff[sel][tm])
pca = PCA(n_components=min(N_TOP_PCS, len(sel) - 1)).fit(ss.transform(diff[sel][tm]))
rot, Rv = varimax(pca.components_)
vpc = pd.DataFrame(pca.transform(ss.transform(diff[sel])) @ Rv, index=diff.index,
                   columns=[f"VPC{i+1}" for i in range(pca.n_components_)])
u_tr = u_full.reindex(vpc[tm].index).dropna()
for c in vpc.columns:
    if np.corrcoef(vpc.loc[tm, c].reindex(u_tr.index).values, u_tr.values)[0, 1] < 0:
        vpc[c] = -vpc[c]
print(f"  factors available: {list(vpc.columns)}")


# ── Recursive Option-A nowcast for one specification ────────────────────────────
def build_Z(pcs, K):
    """exogenous regressor frame: F_{j,t-k}, k=0..K, monthly index."""
    Z = pd.DataFrame(index=vpc.index)
    for j in pcs:
        for k in range(0, K + 1):
            Z[f"{j}_l{k}"] = vpc[j].shift(k)
    return Z

def recursive_optionA(pcs, p, K):
    Z = build_Z(pcs, K)
    idx_all = u_full.index.intersection(Z.dropna().index)
    eval_dates = u_full[(u_full.index >= EVAL_START) & (u_full.index <= EVAL_END)].index
    out = pd.Series(index=eval_dates, dtype=float)
    prev = None
    for t in eval_dates:
        idx = idx_all[idx_all <= t]
        if len(idx) < 30:
            continue
        endog = u_full.reindex(idx).astype(float).copy()
        endog.loc[idx >= (t - pd.DateOffset(months=PUB_LAG - 1))] = np.nan  # mask u_{t-1}, u_t
        exog = Z.reindex(idx)
        try:
            mod = SARIMAX(endog, exog=exog, order=(p, 0, 0),
                          measurement_error=True, trend="c",
                          enforce_stationarity=True)
            if prev is not None and len(prev) == mod.k_params:
                res = mod.fit(start_params=prev, disp=False, method="lbfgs", maxiter=60)
            else:
                res = mod.fit(disp=False, method="lbfgs", maxiter=200)
            prev = res.params
            out[t] = float(res.get_prediction().predicted_mean.iloc[-1])  # û_{t|t} = μ̂+β'Z_t
        except Exception:
            prev = None
    return out


# ── Run all specs ────────────────────────────────────────────────────────────────
covid_mask = None
def rmse(actual, f, excl=False):
    m = actual.notna() & f.notna()
    if excl:
        m &= ~((actual.index >= COVID_START) & (actual.index <= COVID_END))
    return float(np.sqrt(np.mean((actual[m] - f[m]) ** 2)))

print("\nRunning recursive OOS for each specification (SARIMAX refit per month) …")
results, forecasts = [], {}
for pcs, p, K in SPECS:
    name = f"{'+'.join(pcs)} | p={p} | K={K}"
    fc = recursive_optionA(pcs, p, K)
    actual = u_full.reindex(fc.index)
    rf, re = rmse(actual, fc), rmse(actual, fc, True)
    results.append({"spec": name, "pcs": "+".join(pcs), "p": p, "K": K,
                    "rmse_full": rf, "rmse_excovid": re})
    forecasts[name] = fc
    print(f"  {name:<34}  RMSE_full={rf:.4f}  RMSE_exCOVID={re:.4f}")

res_df = pd.DataFrame(results).sort_values("rmse_full").reset_index(drop=True)
best = res_df.iloc[0]
best_fc = forecasts[best["spec"]]
actual = u_full.reindex(best_fc.index)


# ── Reference forecasts (AR(1), B5) for DM / CW and context ──────────────────────
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
fp = HERE / "option_b_lag2_forecasts.csv"
ar1 = None
if fp.exists():
    e = pd.read_csv(fp); e["Tarih"] = pd.to_datetime(e["Tarih"]); e = e.set_index("Tarih")
    ar1 = e["AR1"].reindex(actual.index)
    ref["AR(1) recursive"] = (rmse(actual, ar1), rmse(actual, ar1, True))
    if "B5_VPC1_AR1" in e:
        b5 = e["B5_VPC1_AR1"].reindex(actual.index)
        ref["B5 state space (full model)"] = (rmse(actual, b5), rmse(actual, b5, True))

print("\n" + "=" * 74)
print("OPTION A — specs ranked by full-window RMSE")
print("=" * 74)
print(f"{'spec':<36}{'RMSE full':>11}{'RMSE exC':>11}")
print("-" * 74)
for _, r in res_df.iterrows():
    star = "  ← best" if r["spec"] == best["spec"] else ""
    print(f"{r['spec']:<36}{r['rmse_full']:>11.4f}{r['rmse_excovid']:>11.4f}{star}")
print("-" * 74)
for lbl, (rf, re) in ref.items():
    print(f"{'[ref] '+lbl:<36}{rf:>11.4f}{re:>11.4f}")
print("=" * 74)

if ar1 is not None:
    dm = dm_test(ar1, best_fc); cw = cw_test(ar1, best_fc)
    print(f"\nBest Option-A spec: {best['spec']}")
    print(f"  vs AR(1):  DM t = {dm[0]:+.2f} (p={dm[1]:.3f}),  "
          f"Clark–West t = {cw[0]:+.2f} (p={cw[1]:.3f})")
    print("  DM>0 / CW>0 ⇒ Option A more accurate than the AR(1) benchmark.")


# ── Save + plot ──────────────────────────────────────────────────────────────────
res_df.to_csv(HERE / "statespace_exog_specs.csv", index=False)
fc_out = pd.DataFrame({"actual": actual, "OptionA_best": best_fc})
if ar1 is not None:
    fc_out["AR1"] = ar1
fc_out.index.name = "Tarih"
fc_out.to_csv(HERE / "statespace_exog_forecasts.csv")
print("\nSaved: statespace_exog_specs.csv, statespace_exog_forecasts.csv")

fig, ax = plt.subplots(figsize=(11, 4.4))
ax.axvspan(COVID_START, COVID_END, color="gold", alpha=0.18)
ax.plot(actual.index, actual.values, color="black", lw=2.0, label="Actual $u_t$")
if ar1 is not None:
    ax.plot(ar1.index, ar1.values, color="#7F8C8D", lw=1.3, ls="--", label="AR(1)")
ax.plot(best_fc.index, best_fc.values, color="#00467F", lw=1.7,
        label=f"Option A best ({best['spec']})")
ax.set_ylabel("Non-agricultural unemployment (%)")
ax.set_title("Option A: state space with exogenous Google factors (best spec)")
ax.legend(loc="upper right"); ax.grid(axis="y", ls="--", alpha=0.35)
plt.tight_layout()
plt.show()
