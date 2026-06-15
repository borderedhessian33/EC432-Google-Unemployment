"""
EC 432 Project — PLS specification search (look-ahead-free)
===========================================================
Frozen PLS-1 underperformed because the search→u relationship is non-stationary
(corr 0.74 on 2011–2017 → 0.18 full sample).  Two ideas that should suit PLS:
  • RECURSIVE PLS: re-fit the PLS direction each origin on data through t-2
    (expanding, no look-ahead) so it ADAPTS to the shifting relationship.
  • PLS as a DIRECT regressor in a factor-augmented ARDL (it's a supervised
    predictor of u, not a latent common factor), with its own AR(p) + lags.

Direct-horizon ARDL (fast OLS), publication lag h=2:
    u_t = α + Σ_{ℓ=2}^{p+1} φ_ℓ u_{t-ℓ} + Σ_c Σ_{k=0}^{K} β_{ck} PLS^c_{t-k} + ε_t
Recursive expanding window; preprocessing (deseasonalisation) frozen at Dec-2017;
PLS re-fit either once (frozen) or every origin (recursive), always on u≤t-2.

Compares many PLS specs to: AR-direct, AR(1), and the PCA-VPC1 B5 (0.534).
Output: pls_specs_forecasts.csv + printed table.
"""

import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import norm
from statsmodels.tsa.seasonal import STL
from sklearn.preprocessing import StandardScaler
from sklearn.cross_decomposition import PLSRegression

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent          # exploratory/ — this script's own outputs
ROOT = HERE.parent                              # project root — shared input CSVs
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


print("Loading data …")
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
Dkw = deseas.diff(12).dropna()                       # deseasonalised YoY-diff keywords (raw)


# ── PLS factor providers ────────────────────────────────────────────────────────
def fit_pls(Xfit_raw, yfit, n_comp):
    sc = StandardScaler().fit(Xfit_raw)
    pls = PLSRegression(n_components=n_comp, scale=False).fit(sc.transform(Xfit_raw),
                                                              (yfit - yfit.mean()))
    # sign-orient comp-1 to +corr with u on the fit sample
    f1 = pls.transform(sc.transform(Xfit_raw))[:, 0]
    sign = -1.0 if np.corrcoef(f1, yfit)[0, 1] < 0 else 1.0
    return sc, pls, sign

# frozen PLS (fit once on ≤2017, observed u)
_tm = Dkw.index <= FREEZE
_yf = u_full.reindex(Dkw.index[_tm]); _ok = _yf.notna()
_sc, _pls, _sgn = fit_pls(Dkw.loc[Dkw.index[_tm]][_ok.values].values, _yf[_ok].values, 2)
_FR = _pls.transform(_sc.transform(Dkw.values)); _FR[:, 0] *= _sgn
FROZEN = pd.DataFrame(_FR, index=Dkw.index, columns=["PLS1", "PLS2"])

def factor_frozen(t, n_comp):
    return FROZEN.loc[FROZEN.index <= t, ["PLS1", "PLS2"][:n_comp]]

def factor_recursive(t, n_comp):
    cutoff = t - pd.DateOffset(months=PUB_LAG)         # u observed only ≤ t-2
    fit_idx = Dkw.index[Dkw.index <= cutoff]
    yv = u_full.reindex(fit_idx); ok = yv.notna()
    if ok.sum() < 36:
        return None
    sc, pls, sgn = fit_pls(Dkw.loc[fit_idx][ok.values].values, yv[ok].values, n_comp)
    avail = Dkw.index[Dkw.index <= t]
    F = pls.transform(sc.transform(Dkw.loc[avail].values)); F[:, 0] *= sgn
    return pd.DataFrame(F, index=avail, columns=[f"PLS{i+1}" for i in range(n_comp)])


# ── Recursive ARDL with a PLS factor provider ───────────────────────────────────
def run_spec(mode, n_comp, p, K):
    provider = factor_frozen if mode == "frozen" else factor_recursive
    dates = u_full[(u_full.index >= EVAL_START) & (u_full.index <= EVAL_END)].index
    out = pd.Series(index=dates, dtype=float)
    for t in dates:
        F = provider(t, n_comp)
        if F is None:
            continue
        idx = F.index
        u = u_full.reindex(idx)
        d = pd.DataFrame({"y": u}, index=idx)
        for l in range(2, p + 2):
            d[f"u_l{l}"] = u.shift(l)
        feat = [f"u_l{l}" for l in range(2, p + 2)]
        for c in range(n_comp):
            for k in range(0, K + 1):
                nm = f"f{c}_l{k}"; d[nm] = F.iloc[:, c].shift(k); feat.append(nm)
        cutoff = t - pd.DateOffset(months=PUB_LAG)
        tr = d[d.index <= cutoff].dropna()
        xrow = d.loc[[t], feat]
        if len(tr) < len(feat) + 5 or xrow.isna().any(axis=1).iloc[0]:
            continue
        X = np.column_stack([np.ones(len(tr)), tr[feat].values])
        beta, *_ = np.linalg.lstsq(X, tr["y"].values, rcond=None)
        out[t] = float(np.r_[1.0, xrow.values.ravel()] @ beta)
    return out


# AR-only direct benchmark (no factor)
def run_ar(p):
    dates = u_full[(u_full.index >= EVAL_START) & (u_full.index <= EVAL_END)].index
    out = pd.Series(index=dates, dtype=float)
    u = u_full
    d = pd.DataFrame({"y": u}, index=u.index)
    for l in range(2, p + 2):
        d[f"u_l{l}"] = u.shift(l)
    feat = [f"u_l{l}" for l in range(2, p + 2)]
    for t in dates:
        cutoff = t - pd.DateOffset(months=PUB_LAG)
        tr = d[d.index <= cutoff].dropna(); xrow = d.loc[[t], feat]
        if len(tr) < len(feat) + 5 or xrow.isna().any(axis=1).iloc[0]:
            continue
        X = np.column_stack([np.ones(len(tr)), tr[feat].values])
        beta, *_ = np.linalg.lstsq(X, tr["y"].values, rcond=None)
        out[t] = float(np.r_[1.0, xrow.values.ravel()] @ beta)
    return out


# ── Specs to try ─────────────────────────────────────────────────────────────────
SPECS = [
    ("frozen PLS1   p3 K0", "frozen", 1, 3, 0),
    ("frozen PLS1   p3 K2", "frozen", 1, 3, 2),
    ("recursive PLS1 p3 K0", "recursive", 1, 3, 0),
    ("recursive PLS1 p3 K1", "recursive", 1, 3, 1),
    ("recursive PLS1 p3 K2", "recursive", 1, 3, 2),
    ("recursive PLS1 p2 K2", "recursive", 1, 2, 2),
    ("recursive PLS1+2 p3 K2", "recursive", 2, 3, 2),
]
print("Running PLS specs …")
fcs = {lbl: run_spec(mode, nc, p, K) for (lbl, mode, nc, p, K) in SPECS}
fc_ar = run_ar(3)

# references
ref = {}
fp = ROOT / "option_b_lag2_forecasts.csv"
if fp.exists():
    e = pd.read_csv(fp); e["Tarih"] = pd.to_datetime(e["Tarih"]); e = e.set_index("Tarih")
    ref["AR(1) recursive"] = e["AR1"]; ref["B5 (PCA-VPC1)"] = e["B5_VPC1_AR1"]

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
print("PLS specification search   (eval 2018–2025, n=96)")
print("=" * 78)
print(f"{'Spec':<26}{'RMSE full':>11}{'RMSE exC':>11}{'CW vs AR-dir':>14}")
print("-" * 78)
print(f"{'AR-direct (p=3)':<26}{rmse(fc_ar):>11.4f}{rmse(fc_ar,True):>11.4f}{'—':>14}")
for lbl, f in fcs.items():
    c = cw(fc_ar, f)
    print(f"{lbl:<26}{rmse(f):>11.4f}{rmse(f,True):>11.4f}{c[0]:>+11.2f} ({c[1]:.2f})")
print("-" * 78)
for lbl, f in ref.items():
    print(f"{lbl:<26}{rmse(f):>11.4f}{rmse(f,True):>11.4f}{'—':>14}")
print("=" * 78)

best = min(fcs.items(), key=lambda kv: rmse(kv[1]))
print(f"Best PLS spec by full RMSE: {best[0]}  ({rmse(best[1]):.4f})  |  B5-PCA = {rmse(ref['B5 (PCA-VPC1)']):.4f}")
pd.DataFrame({"actual": actual, **{k: v.reindex(actual.index) for k, v in fcs.items()}}
             ).rename_axis("Tarih").to_csv(HERE / "pls_specs_forecasts.csv")
print("Saved: pls_specs_forecasts.csv")
