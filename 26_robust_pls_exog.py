"""
EC 432 Project — Frozen PLS factors inside the state space (candidates beyond B5)
=================================================================================
B5-PLS (PLS in the factor-measurement equation) lost to B5-PCA.  But PLS is a
SUPERVISED predictor of u, so its natural state-space role is as an EXOGENOUS
regressor in the measurement equation, not a latent-factor loading:

    u_t = c + μ_t + Σ_c Σ_{k=0}^{K} β_{ck} PLS^c_{t-k} + δ·D^covid_t + ε^u_t
    μ_t = φ_1 μ_{t-1} [+ φ_2 μ_{t-2}] + η_t
    nowcast  û_{t|t} = μ̂_{t|t} + Σ β·PLS_{t-k} + δ·covid     (predicted observation)

Implemented as SARIMAX(u, exog=[PLS-lags, covid], order=(p,0,0),
measurement_error=True, trend='c').  PLS is FROZEN at Dec-2017 (no look-ahead;
real-time).  Contemporaneous PLS (k=0) is allowed because it's available at the
origin and, unlike PCA, carries direct predictive content for u.

Sweeps several candidates; compares to AR(1), B5-PCA (0.534), B5-PLS (0.584).
Output: pls_statespace_forecasts.csv + printed table.
"""

import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import norm
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.seasonal import STL
from sklearn.preprocessing import StandardScaler
from sklearn.cross_decomposition import PLSRegression

warnings.filterwarnings("ignore")
HERE = Path(__file__).parent
EVAL_START  = pd.Timestamp("2018-01-01")
EVAL_END    = pd.Timestamp("2025-12-01")
FREEZE      = pd.Timestamp("2017-12-31")
COVID_START = pd.Timestamp("2020-04-01")
COVID_END   = pd.Timestamp("2021-12-01")
PUB_LAG     = 2
HAC_LAGS    = PUB_LAG - 1


def frozen_deseas(s, fz):
    tr = s[s.index <= fz].dropna()
    if len(tr) < 24:
        return s
    stl = STL(tr, period=12, robust=False).fit()
    pat = stl.seasonal.groupby(stl.seasonal.index.month).mean()
    return s - pd.Series(s.index.month, index=s.index).map(pat).values


print("Loading data & building FROZEN PLS factors (≤2017) …")
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
Dkw = deseas.diff(12).dropna()
tm = Dkw.index <= FREEZE
yf = u_full.reindex(Dkw.index[tm]); ok = yf.notna()
sc = StandardScaler().fit(Dkw.loc[Dkw.index[tm]][ok.values].values)
pls = PLSRegression(n_components=2, scale=False).fit(sc.transform(Dkw.loc[Dkw.index[tm]][ok.values].values),
                                                     (yf[ok] - yf[ok].mean()).values)
F = pls.transform(sc.transform(Dkw.values))
f1tr = pls.transform(sc.transform(Dkw.loc[Dkw.index[tm]][ok.values].values))[:, 0]
if np.corrcoef(f1tr, yf[ok].values)[0, 1] < 0:
    F[:, 0] *= -1
PLSf = pd.DataFrame(F, index=Dkw.index, columns=["PLS1", "PLS2"])
print(f"  frozen PLS1 corr with u (≤2017) = {np.corrcoef(f1tr, yf[ok].values)[0,1]:+.3f}")


# ── Recursive Option-A SARIMAX with frozen PLS exog ─────────────────────────────
def run_spec(pcs, p, K, use_covid=True):
    cols = []
    Z = pd.DataFrame(index=PLSf.index)
    for c in pcs:
        for k in range(0, K + 1):
            Z[f"{c}_l{k}"] = PLSf[c].shift(k); cols.append(f"{c}_l{k}")
    if use_covid:
        Z["covid"] = ((PLSf.index >= COVID_START) & (PLSf.index <= COVID_END)).astype(float)
        cols.append("covid")
    Z = Z.dropna()
    idx_all = u_full.index.intersection(Z.index)
    dates = u_full[(u_full.index >= EVAL_START) & (u_full.index <= EVAL_END)].index
    out, prev = pd.Series(index=dates, dtype=float), None
    for t in dates:
        idx = idx_all[idx_all <= t]
        if len(idx) < 30:
            continue
        endog = u_full.reindex(idx).astype(float).copy()
        endog.loc[idx >= (t - pd.DateOffset(months=PUB_LAG - 1))] = np.nan   # mask u_{t-1},u_t
        exog = Z.reindex(idx)[cols]
        try:
            mod = SARIMAX(endog, exog=exog, order=(p, 0, 0),
                          measurement_error=True, trend="c", enforce_stationarity=True)
            if prev is not None and len(prev) == mod.k_params:
                res = mod.fit(start_params=prev, disp=False, method="lbfgs", maxiter=80)
            else:
                res = mod.fit(disp=False, method="lbfgs", maxiter=200)
            prev = res.params
            out[t] = float(res.get_prediction().predicted_mean.iloc[-1])
        except Exception:
            prev = None
    return out


SPECS = [
    ("PLS1  K0 p1 (exog)",      ["PLS1"],         1, 0),
    ("PLS1  K1 p1 (exog)",      ["PLS1"],         1, 1),
    ("PLS1  K2 p1 (exog)",      ["PLS1"],         1, 2),
    ("PLS1  K2 p2 (exog)",      ["PLS1"],         2, 2),
    ("PLS1+2 K1 p1 (exog)",     ["PLS1", "PLS2"], 1, 1),
    ("PLS1+2 K2 p1 (exog)",     ["PLS1", "PLS2"], 1, 2),
]
print("\nRunning frozen-PLS state-space candidates …")
fcs = {}
for lbl, pcs, p, K in SPECS:
    fcs[lbl] = run_spec(pcs, p, K, True)
    print(f"  done: {lbl}")

# references
ar1 = b5pca = b5pls = None
fp = HERE / "option_b_lag2_forecasts.csv"
if fp.exists():
    e = pd.read_csv(fp); e["Tarih"] = pd.to_datetime(e["Tarih"]); e = e.set_index("Tarih")
    ar1 = e["AR1"]; b5pca = e["B5_VPC1_AR1"]
fp2 = HERE / "b5_pls_forecasts.csv"
if fp2.exists():
    e2 = pd.read_csv(fp2); e2["Tarih"] = pd.to_datetime(e2["Tarih"]); e2 = e2.set_index("Tarih")
    b5pls = e2["B5_PLS"]

anyfc = next(iter(fcs.values()))
actual = u_full.reindex(anyfc.index)
covid = (actual.index >= COVID_START) & (actual.index <= COVID_END)

def rmse(f, excl=False):
    f = f.reindex(actual.index); m = actual.notna() & f.notna()
    if excl:
        m &= ~covid
    return float(np.sqrt(np.mean((actual[m] - f[m]) ** 2)))

def cw(fr, fu):
    fr = fr.reindex(actual.index); fu = fu.reindex(actual.index)
    m = actual.notna() & fr.notna() & fu.notna()
    y, a, b = actual[m].values, fr[m].values, fu[m].values
    ft = (y - a) ** 2 - ((y - b) ** 2 - (a - b) ** 2)
    r = sm.OLS(ft, np.ones((len(ft), 1))).fit(cov_type="HAC", cov_kwds={"maxlags": HAC_LAGS})
    return float(r.tvalues[0]), float(1 - norm.cdf(r.tvalues[0]))

print("\n" + "=" * 78)
print("Frozen-PLS state-space candidates   (eval 2018–2025, n=96)")
print("=" * 78)
print(f"{'Model':<26}{'RMSE full':>11}{'RMSE exC':>11}{'CW vs AR(1)':>14}")
print("-" * 78)
for lbl, f in fcs.items():
    c = cw(ar1, f) if ar1 is not None else (np.nan, np.nan)
    print(f"{lbl:<26}{rmse(f):>11.4f}{rmse(f,True):>11.4f}{c[0]:>+11.2f} ({c[1]:.2f})")
print("-" * 78)
for lbl, f in [("AR(1) recursive", ar1), ("B5-PLS (factor-meas.)", b5pls), ("B5-PCA (PCA-VPC1)", b5pca)]:
    if f is not None:
        print(f"{lbl:<26}{rmse(f):>11.4f}{rmse(f,True):>11.4f}{'—':>14}")
print("=" * 78)
best = min(fcs.items(), key=lambda kv: rmse(kv[1]))
print(f"Best frozen-PLS SS spec: {best[0]} = {rmse(best[1]):.4f} full / {rmse(best[1],True):.4f} exC")
pd.DataFrame({"actual": actual, **{k: v.reindex(actual.index) for k, v in fcs.items()}}
             ).rename_axis("Tarih").to_csv(HERE / "pls_statespace_forecasts.csv")
print("Saved: pls_statespace_forecasts.csv")
