"""
EC 432 Project — PLS transformation test: STL+YoY vs STL-only vs YoY-only
=========================================================================
Tests how the keyword transformation affects the FROZEN-PLS results, for the two
best PLS structures:
  • Option-A exog : u_t = μ_t + Σβ·PLS1_{t-k} (k=0..2) + δ·covid + ε,  μ~AR(1)
                    (SARIMAX exog; PLS1 enters as exogenous regressor)
  • B5 structure  : PLS1 in the factor-measurement eq + lagged bridge (AR1,ld2,K2,COVID)

Transforms (all frozen at Dec-2017 where seasonal):
  • stl_yoy  : STL-deseasonalise then 12-month YoY difference  (current pipeline)
  • stl_only : STL-deseasonalise only (levels, no differencing)
  • yoy_only : 12-month YoY difference of RAW trends (no STL)

Frozen PLS (≤2017) in every case. Compares to AR(1) and B5-PCA (0.534).
Output: pls_transforms_results.csv + printed table.
"""

import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import norm
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.statespace.mlemodel import MLEModel
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
N_BETA_LAGS = 2
HAC_LAGS    = PUB_LAG - 1


# ── B5 model (verbatim) ──────────────────────────────────────────────────────────
class OptionBModel(MLEModel):
    def __init__(self, endog, ar_order, lag_depth, lpm, n_beta_lags=2, covid_col=None):
        self.ar_order = ar_order; self.lag_depth = lag_depth
        self.n_pcs = endog.shape[1] - 1; self.n_beta_lags = n_beta_lags
        self.use_covid = covid_col is not None
        k_s = max(ar_order, lag_depth + 1)
        super().__init__(endog, k_states=k_s, k_posdef=1)
        lpm = np.asarray(lpm, float); self._lpm = np.where(np.isfinite(lpm), lpm, 0.0).T
        self._covid = (np.asarray(covid_col, float).flatten() if covid_col is not None else np.zeros(self.nobs))
        for i in range(1, k_s):
            self["transition", i, i - 1] = 1.0
        self["selection", 0, 0] = 1.0; self["design", 0, 0] = 1.0
        self["obs_intercept"] = np.zeros((1 + self.n_pcs, self.nobs))
        self.initialize_approximate_diffuse()
    @property
    def _noise_idx(self): return self.ar_order + self.n_pcs * (self.lag_depth + 1)
    @property
    def param_names(self):
        n = [f"phi{i+1}" for i in range(self.ar_order)]
        for p in range(self.n_pcs):
            for j in range(self.lag_depth + 1): n.append(f"l{chr(97+p)}{j+1}")
        n.append("sig_u")
        for p in range(self.n_pcs): n.append(f"sig_pc{chr(97+p)}")
        n.append("sig_eta")
        for p in range(self.n_pcs):
            for l in range(1, self.n_beta_lags + 1): n.append(f"beta_{chr(97+p)}_lag{l}")
        if self.use_covid: n.append("delta_covid")
        return n
    @property
    def start_params(self):
        parts = [0.7] + [0.1] * (self.ar_order - 1)
        for _ in range(self.n_pcs): parts += [0.5] + [0.1] * self.lag_depth
        parts += [0.3] * (1 + self.n_pcs) + [0.3] + [0.05] * (self.n_pcs * self.n_beta_lags)
        if self.use_covid: parts.append(0.0)
        return np.array(parts)
    def transform_params(self, u):
        p = u.copy()
        for i in range(self.ar_order): p[i] = np.tanh(u[i])
        ni = self._noise_idx
        for i in range(1 + self.n_pcs + 1): p[ni + i] = np.exp(u[ni + i])
        return p
    def untransform_params(self, c):
        p = c.copy()
        for i in range(self.ar_order): p[i] = np.arctanh(np.clip(c[i], -0.9999, 0.9999))
        ni = self._noise_idx
        for i in range(1 + self.n_pcs + 1): p[ni + i] = np.log(max(c[ni + i], 1e-12))
        return p
    def update(self, params, **kw):
        params = super().update(params, **kw)
        for i in range(self.ar_order): self["transition", 0, i] = params[i]
        idx = self.ar_order
        for p_i in range(self.n_pcs):
            for j in range(self.lag_depth + 1): self["design", p_i + 1, j] = params[idx]; idx += 1
        self["obs_cov", 0, 0] = params[idx] ** 2; idx += 1
        for p_i in range(self.n_pcs): self["obs_cov", p_i + 1, p_i + 1] = params[idx] ** 2; idx += 1
        self["state_cov", 0, 0] = params[idx] ** 2; idx += 1
        betas = params[idx: idx + self.n_pcs * self.n_beta_lags]; bc = betas @ self._lpm
        u_int = bc + params[idx + len(betas)] * self._covid if self.use_covid else bc
        oi = np.zeros((1 + self.n_pcs, self.nobs)); oi[0, :] = u_int
        self["obs_intercept"] = oi


def frozen_deseas(s, fz):
    tr = s[s.index <= fz].dropna()
    if len(tr) < 24: return s
    stl = STL(tr, period=12, robust=False).fit()
    pat = stl.seasonal.groupby(stl.seasonal.index.month).mean()
    return s - pd.Series(s.index.month, index=s.index).map(pat).values


# ── Data ─────────────────────────────────────────────────────────────────────────
trends = pd.read_csv(HERE / "calibrated_trends_custom.csv", index_col=0, parse_dates=True).resample("MS").first()
unemp = pd.read_csv(HERE / "unemp_csv.csv", sep=";", decimal=",",
                    encoding="utf-8-sig", index_col=0, parse_dates=False)
unemp.columns = ["u"]
unemp = unemp[unemp.index.notna() & (unemp.index.astype(str) != "nan")]
unemp.index = pd.to_datetime(unemp.index, format="%Y-%m")
u_full = unemp.resample("MS").first()["u"].astype(float).dropna()
unemp_raw = unemp.resample("MS").first()

deseas = pd.DataFrame(index=trends.index, columns=trends.columns, dtype=float)
for c in trends.columns:
    deseas[c] = frozen_deseas(trends[c].astype(float), FREEZE)

def make_Dkw(mode):
    if mode == "stl_yoy":  return deseas.diff(12).dropna()
    if mode == "stl_only": return deseas.dropna()
    if mode == "yoy_only": return trends.diff(12).dropna()

def frozen_pls(Dkw):
    tm = Dkw.index <= FREEZE
    yf = u_full.reindex(Dkw.index[tm]); ok = yf.notna()
    sc = StandardScaler().fit(Dkw.loc[Dkw.index[tm]][ok.values].values)
    pls = PLSRegression(n_components=2, scale=False).fit(sc.transform(Dkw.loc[Dkw.index[tm]][ok.values].values),
                                                         (yf[ok] - yf[ok].mean()).values)
    F = pls.transform(sc.transform(Dkw.values))
    f1 = pls.transform(sc.transform(Dkw.loc[Dkw.index[tm]][ok.values].values))[:, 0]
    if np.corrcoef(f1, yf[ok].values)[0, 1] < 0: F[:, 0] *= -1
    corr = np.corrcoef(F[tm][ok.values, 0], yf[ok].values)[0, 1]
    return pd.DataFrame(F, index=Dkw.index, columns=["PLS1", "PLS2"]), corr


# ── Engine 1: Option-A exog (SARIMAX) ───────────────────────────────────────────
def run_optionA(PLSf, K=2, p=1):
    Z = pd.DataFrame(index=PLSf.index)
    cols = []
    for k in range(0, K + 1):
        Z[f"PLS1_l{k}"] = PLSf["PLS1"].shift(k); cols.append(f"PLS1_l{k}")
    Z["covid"] = ((PLSf.index >= COVID_START) & (PLSf.index <= COVID_END)).astype(float); cols.append("covid")
    Z = Z.dropna()
    idx_all = u_full.index.intersection(Z.index)
    dates = u_full[(u_full.index >= EVAL_START) & (u_full.index <= EVAL_END)].index
    out, prev = pd.Series(index=dates, dtype=float), None
    for t in dates:
        idx = idx_all[idx_all <= t]
        if len(idx) < 30: continue
        endog = u_full.reindex(idx).astype(float).copy()
        endog.loc[idx >= (t - pd.DateOffset(months=PUB_LAG - 1))] = np.nan
        exog = Z.reindex(idx)[cols]
        try:
            mod = SARIMAX(endog, exog=exog, order=(p, 0, 0), measurement_error=True,
                          trend="c", enforce_stationarity=True)
            res = (mod.fit(start_params=prev, disp=False, method="lbfgs", maxiter=80)
                   if prev is not None and len(prev) == mod.k_params
                   else mod.fit(disp=False, method="lbfgs", maxiter=200))
            prev = res.params
            out[t] = float(res.get_prediction().predicted_mean.iloc[-1])
        except Exception:
            prev = None
    return out


# ── Engine 2: B5 structure (OptionBModel) ───────────────────────────────────────
def run_b5(PLSf):
    scores_df = PLSf[["PLS1"]]
    dates = u_full[(u_full.index >= EVAL_START) & (u_full.index <= EVAL_END)].index
    out, prev = pd.Series(index=dates, dtype=float), None
    for t in dates:
        u_vis = unemp_raw.loc[unemp_raw.index <= t, "u"].copy()
        lag_cutoff = t - pd.DateOffset(months=PUB_LAG - 1)
        if u_vis.loc[u_vis.index < lag_cutoff].dropna().shape[0] < 12: continue
        pc_avail = scores_df.loc[scores_df.index <= t]
        try:
            comb = pc_avail[["PLS1"]].join(u_vis.rename("u"), how="left").dropna(subset=["PLS1"])
            comb = comb[["u", "PLS1"]]
            if len(comb) < 24: continue
            arr = comb.values.astype(float); arr[comb.index >= lag_cutoff, 0] = np.nan
            tr = comb.index < lag_cutoff
            mu_e = np.nanmean(arr[tr], axis=0); sd_e = np.nanstd(arr[tr], axis=0); sd_e[sd_e < 1e-8] = 1.0
            endog = (arr - mu_e) / sd_e
            n = len(comb); lpm = np.zeros((n, N_BETA_LAGS)); pc = endog[:, 1]
            for l in range(1, N_BETA_LAGS + 1): lpm[l:, l - 1] = pc[:-l]
            covid = ((comb.index >= COVID_START) & (comb.index <= COVID_END)).astype(float) / sd_e[0]
            mdl = OptionBModel(endog, 1, 2, lpm, N_BETA_LAGS, covid)
            res = (mdl.fit(start_params=prev, method="powell", maxiter=600, disp=False)
                   if prev is not None and len(prev) == len(mdl.start_params)
                   else mdl.fit(start_params=mdl.fit(method="nm", maxiter=3000, disp=False).params,
                                method="powell", maxiter=1000, disp=False))
            prev = res.params
            out[t] = float(res.filter_results.filtered_state[0, -1]) * sd_e[0] + mu_e[0]
        except Exception:
            prev = None
    return out


# ── Run all transforms × both engines ───────────────────────────────────────────
ar1 = b5pca = None
fp = HERE / "option_b_lag2_forecasts.csv"
if fp.exists():
    e = pd.read_csv(fp); e["Tarih"] = pd.to_datetime(e["Tarih"]); e = e.set_index("Tarih")
    ar1 = e["AR1"]; b5pca = e["B5_VPC1_AR1"]
actual = u_full[(u_full.index >= EVAL_START) & (u_full.index <= EVAL_END)]
covid = (actual.index >= COVID_START) & (actual.index <= COVID_END)

def rmse(f, excl=False):
    f = f.reindex(actual.index); m = actual.notna() & f.notna()
    if excl: m &= ~covid
    return float(np.sqrt(np.mean((actual[m] - f[m]) ** 2)))

rows = []
for mode in ["stl_yoy", "stl_only", "yoy_only"]:
    Dkw = make_Dkw(mode)
    PLSf, corr = frozen_pls(Dkw)
    print(f"\n[{mode}]  frozen PLS1 corr with u (≤2017) = {corr:+.3f}")
    fa = run_optionA(PLSf); print(f"   Option-A exog done")
    fb = run_b5(PLSf);      print(f"   B5-PLS done")
    rows.append((mode, "Option-A exog (K2,p1)", rmse(fa), rmse(fa, True)))
    rows.append((mode, "B5 structure",          rmse(fb), rmse(fb, True)))

print("\n" + "=" * 72)
print("PLS transformation test   (frozen PLS, eval 2018–2025, n=96)")
print("=" * 72)
print(f"{'transform':<12}{'model':<24}{'RMSE full':>11}{'RMSE exC':>11}")
print("-" * 72)
for mode, model, rf, re in rows:
    print(f"{mode:<12}{model:<24}{rf:>11.4f}{re:>11.4f}")
print("-" * 72)
print(f"{'ref':<12}{'AR(1)':<24}{rmse(ar1):>11.4f}{rmse(ar1,True):>11.4f}")
print(f"{'ref':<12}{'B5-PCA':<24}{rmse(b5pca):>11.4f}{rmse(b5pca,True):>11.4f}")
print("=" * 72)
pd.DataFrame(rows, columns=["transform", "model", "rmse_full", "rmse_exc"]).to_csv(
    HERE / "pls_transforms_results.csv", index=False)
print("Saved: pls_transforms_results.csv")
