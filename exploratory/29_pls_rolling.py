"""
EC 432 Project — Rolling/expanding selection over a WIDE grid of PLS state-space models
=======================================================================================
Frozen PLS factors (YoY differencing only — STL is provably redundant).  Candidate
grid spans two state-space structures × factor count × AR order × lag depth × COVID:

  • Option-A exog : u_t = c + μ_t + Σ β·PLS_{t-k} (+δ covid) + ε,  μ~AR(p)
                    (SARIMAX; PLS as exogenous regressor — natural for a predictor)
  • B5 structure  : PLS in the factor-measurement eq + lagged bridge (AR p, ld2, COVID)

Each candidate is a genuine recursive expanding-window nowcast (2-month pub lag,
filtered-state extraction).  Then ROLLING selection: each test year, re-rank
candidates on all prior months and forecast with the leader.  Eval 2019–2025,
Clark–West vs AR(1).  Look-ahead-free (frozen PLS uses u only ≤ t in fitting; the
recursive selection uses only prior months).

Output: pls_rolling_forecasts.csv, pls_rolling_menu.csv, pls_rolling_picks.csv
"""

import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import norm
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.statespace.mlemodel import MLEModel
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


# ── B5-type model ────────────────────────────────────────────────────────────────
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


# ── Data + frozen PLS (YoY only) ─────────────────────────────────────────────────
trends = pd.read_csv(ROOT / "calibrated_trends_custom.csv", index_col=0, parse_dates=True).resample("MS").first()
unemp = pd.read_csv(ROOT / "unemp_csv.csv", sep=";", decimal=",",
                    encoding="utf-8-sig", index_col=0, parse_dates=False)
unemp.columns = ["u"]
unemp = unemp[unemp.index.notna() & (unemp.index.astype(str) != "nan")]
unemp.index = pd.to_datetime(unemp.index, format="%Y-%m")
unemp_raw = unemp.resample("MS").first()
u_full = unemp_raw["u"].astype(float).dropna()

Dkw = trends.diff(12).dropna()                         # YoY only (== STL+YoY)
tm = Dkw.index <= FREEZE
yf = u_full.reindex(Dkw.index[tm]); ok = yf.notna()
sc = StandardScaler().fit(Dkw.loc[Dkw.index[tm]][ok.values].values)
pls = PLSRegression(n_components=2, scale=False).fit(sc.transform(Dkw.loc[Dkw.index[tm]][ok.values].values),
                                                     (yf[ok] - yf[ok].mean()).values)
F = pls.transform(sc.transform(Dkw.values))
f1 = pls.transform(sc.transform(Dkw.loc[Dkw.index[tm]][ok.values].values))[:, 0]
if np.corrcoef(f1, yf[ok].values)[0, 1] < 0: F[:, 0] *= -1
PLSf = pd.DataFrame(F, index=Dkw.index, columns=["PLS1", "PLS2"])
print(f"frozen PLS1 corr with u (≤2017) = {np.corrcoef(f1, yf[ok].values)[0,1]:+.3f}")


# ── Engines ──────────────────────────────────────────────────────────────────────
def run_optionA(pcs, p, K, covid):
    Z = pd.DataFrame(index=PLSf.index); cols = []
    for c in pcs:
        for k in range(0, K + 1):
            Z[f"{c}_l{k}"] = PLSf[c].shift(k); cols.append(f"{c}_l{k}")
    if covid:
        Z["covid"] = ((PLSf.index >= COVID_START) & (PLSf.index <= COVID_END)).astype(float); cols.append("covid")
    Z = Z.dropna(); idx_all = u_full.index.intersection(Z.index)
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
            res = (mod.fit(start_params=prev, disp=False, method="lbfgs", maxiter=60)
                   if prev is not None and len(prev) == mod.k_params
                   else mod.fit(disp=False, method="lbfgs", maxiter=200))
            prev = res.params
            out[t] = float(res.get_prediction().predicted_mean.iloc[-1])
        except Exception:
            prev = None
    return out

def run_b5(pcs, ar, Kb, covid):
    scores = PLSf[pcs]
    dates = u_full[(u_full.index >= EVAL_START) & (u_full.index <= EVAL_END)].index
    out, prev = pd.Series(index=dates, dtype=float), None
    for t in dates:
        u_vis = unemp_raw.loc[unemp_raw.index <= t, "u"].copy()
        lc = t - pd.DateOffset(months=PUB_LAG - 1)
        if u_vis.loc[u_vis.index < lc].dropna().shape[0] < 12: continue
        pa = scores.loc[scores.index <= t]
        try:
            comb = pa[pcs].join(u_vis.rename("u"), how="left").dropna(subset=pcs)
            comb = comb[["u"] + pcs]
            if len(comb) < 24: continue
            arr = comb.values.astype(float); arr[comb.index >= lc, 0] = np.nan
            tr = comb.index < lc
            mu_e = np.nanmean(arr[tr], axis=0); sd_e = np.nanstd(arr[tr], axis=0); sd_e[sd_e < 1e-8] = 1.0
            endog = (arr - mu_e) / sd_e
            n = len(comb); npc = len(pcs); lpm = np.zeros((n, npc * Kb))
            for pi in range(npc):
                pc = endog[:, pi + 1]
                for l in range(1, Kb + 1): lpm[l:, pi * Kb + (l - 1)] = pc[:-l]
            covcol = (((comb.index >= COVID_START) & (comb.index <= COVID_END)).astype(float) / sd_e[0]
                      if covid else None)
            mdl = OptionBModel(endog, ar, 2, lpm, Kb, covcol)
            res = (mdl.fit(start_params=prev, method="powell", maxiter=600, disp=False)
                   if prev is not None and len(prev) == len(mdl.start_params)
                   else mdl.fit(start_params=mdl.fit(method="nm", maxiter=3000, disp=False).params,
                                method="powell", maxiter=1000, disp=False))
            prev = res.params
            out[t] = float(res.filter_results.filtered_state[0, -1]) * sd_e[0] + mu_e[0]
        except Exception:
            prev = None
    return out


# ── Wide candidate grid ──────────────────────────────────────────────────────────
GRID = [
    # Option-A exog (SARIMAX)
    ("A:PLS1 p1 K0 cov",   "A", ["PLS1"], 1, 0, True),
    ("A:PLS1 p1 K1 cov",   "A", ["PLS1"], 1, 1, True),
    ("A:PLS1 p1 K2 cov",   "A", ["PLS1"], 1, 2, True),
    ("A:PLS1 p2 K2 cov",   "A", ["PLS1"], 2, 2, True),
    ("A:PLS1 p1 K2 nocov", "A", ["PLS1"], 1, 2, False),
    ("A:PLS12 p1 K1 cov",  "A", ["PLS1", "PLS2"], 1, 1, True),
    ("A:PLS12 p1 K2 cov",  "A", ["PLS1", "PLS2"], 1, 2, True),
    # B5 structure (custom MLE)
    ("B:PLS1 p1 K2 cov",   "B", ["PLS1"], 1, 2, True),
    ("B:PLS1 p2 K2 cov",   "B", ["PLS1"], 2, 2, True),
    ("B:PLS1 p1 K3 cov",   "B", ["PLS1"], 1, 3, True),
    ("B:PLS12 p1 K2 cov",  "B", ["PLS1", "PLS2"], 1, 2, True),
]
print(f"Running {len(GRID)} PLS state-space candidates …")
cand = {}
for lbl, eng, pcs, p, K, cov in GRID:
    cand[lbl] = run_optionA(pcs, p, K, cov) if eng == "A" else run_b5(pcs, p, K, cov)
    print(f"  done: {lbl}")


# ── References + rolling selection (LASSO-free: clean benchmark + headline) ──────
b = pd.read_csv(ROOT / "ar_benchmarks.csv", index_col=0, parse_dates=True)
h = pd.read_csv(ROOT / "candidate_forecasts.csv", index_col=0, parse_dates=True)
AR1 = b["AR1"]; B5PCA = h["VPC3+VPC4_nocov"]   # B5PCA now = the no-LASSO headline
actual = u_full[(u_full.index >= EVAL_START) & (u_full.index <= EVAL_END)]
covid = (actual.index >= COVID_START) & (actual.index <= COVID_END)

def rmse(f, lo, hi, excl=False):
    f = f.reindex(actual.index); m = (actual.index >= lo) & (actual.index <= hi) & actual.notna() & f.notna()
    if excl: m &= ~covid
    return float(np.sqrt(np.mean((actual[m] - f[m]) ** 2)))

# rolling expanding annual selection (over PLS candidates only)
fc = pd.Series(index=actual.index, dtype=float); picks = {}
for yr in range(2019, 2026):
    sel_hi = pd.Timestamp(f"{yr-1}-12-01")
    val = pd.Series({n: rmse(f, actual.index[0], sel_hi) for n, f in cand.items()}).sort_values()
    pick = val.index[0]; picks[yr] = pick
    ym = (actual.index >= pd.Timestamp(f"{yr}-01-01")) & (actual.index <= pd.Timestamp(f"{yr}-12-01"))
    fc[ym] = cand[pick].reindex(actual.index)[ym]

lo, hi = pd.Timestamp("2019-01-01"), pd.Timestamp("2025-12-01")
def cw(fr, fu):
    fr = fr.reindex(actual.index); fu = fu.reindex(actual.index)
    m = (actual.index >= lo) & (actual.index <= hi) & actual.notna() & fr.notna() & fu.notna()
    y, a, b = actual[m].values, fr[m].values, fu[m].values
    ft = (y - a) ** 2 - ((y - b) ** 2 - (a - b) ** 2)
    r = sm.OLS(ft, np.ones((len(ft), 1))).fit(cov_type="HAC", cov_kwds={"maxlags": HAC_LAGS})
    return float(r.tvalues[0]), float(1 - norm.cdf(r.tvalues[0]))

print("\n" + "=" * 70)
print("PLS candidate menu — full-window RMSE (eval 2018–2025)")
print("=" * 70)
menu = pd.Series({n: rmse(f, EVAL_START, EVAL_END) for n, f in cand.items()}).sort_values()
menu_exc = {n: rmse(cand[n], EVAL_START, EVAL_END, True) for n in cand}
for n, r in menu.items():
    print(f"  {n:<22}{r:>9.4f}  (exC {menu_exc[n]:.4f})")

print("\nRolling yearly picks:")
for yr, p in picks.items():
    print(f"  {yr}: {p}")

print("\n" + "=" * 70)
print("ROLLING-SELECTED PLS state space — test 2019–2025")
print("=" * 70)
print(f"{'Model':<30}{'RMSE full':>11}{'RMSE exC':>11}")
print("-" * 70)
print(f"{'Rolling-selected PLS':<30}{rmse(fc,lo,hi):>11.4f}{rmse(fc,lo,hi,True):>11.4f}")
print(f"{'AR(1)':<30}{rmse(AR1,lo,hi):>11.4f}{rmse(AR1,lo,hi,True):>11.4f}")
print(f"{'B5-PCA (PCA rolling≈B5)':<30}{rmse(B5PCA,lo,hi):>11.4f}{rmse(B5PCA,lo,hi,True):>11.4f}")
print("-" * 70)
c = cw(AR1, fc)
print(f"Rolling-PLS vs AR(1): Clark–West t={c[0]:+.2f} (p={c[1]:.3f})")
print("=" * 70)

pd.DataFrame({"actual": actual, "rolling_pls": fc, "AR1": AR1, "B5_PCA": B5PCA}
             ).rename_axis("Tarih").to_csv(HERE / "pls_rolling_forecasts.csv")
pd.DataFrame({"rmse_full": menu, "rmse_exc": pd.Series(menu_exc)}).to_csv(HERE / "pls_rolling_menu.csv")
pd.Series(picks, name="pick").rename_axis("year").to_frame().to_csv(HERE / "pls_rolling_picks.csv")
print("\nSaved: pls_rolling_forecasts.csv, pls_rolling_menu.csv, pls_rolling_picks.csv")
