"""
EC 432 Project — Rolling selection over no-LASSO candidates: VPC1–VPC4 × SS specs
==================================================================================
LASSO is unnecessary (shown in 30): the labour factor is recoverable from PCA+
varimax on ALL 21 keywords by selecting the u-relevant component.  So the candidate
grid is the all-keyword VPC1–VPC4 crossed with the B5-type state-space specs, and a
look-ahead-free ROLLING/EXPANDING selection chooses among them.

Candidates (state space = OptionB: AR(1), ld=2, β-lags=2, factor-measurement + bridge):
    factor ∈ {VPC1, VPC2, VPC3, VPC4}  ×  COVID ∈ {dummy, none}      = 8
    + AR(1), AR(2) benchmarks (so the selector can decline Trends)

Pipeline: frozen deseasonalise → YoY diff → standardise → PCA(4) on all 21 → varimax
→ sign-anchor (all frozen at Dec-2017).  Recursive expanding window, 2-month pub lag.
Rolling: each test year, re-rank candidates on all prior months; forecast with the
leader.  Eval 2019–2025, Clark–West vs AR(1).

Output: nolasso_rolling_forecasts.csv, nolasso_rolling_menu.csv, nolasso_rolling_picks.csv
"""

import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import norm
from statsmodels.tsa.statespace.mlemodel import MLEModel
from statsmodels.tsa.seasonal import STL
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent          # exploratory/ — this script's own outputs
ROOT = HERE.parent                              # project root — shared input CSVs
EVAL_START  = pd.Timestamp("2018-01-01")
EVAL_END    = pd.Timestamp("2025-12-01")
FREEZE      = pd.Timestamp("2017-12-31")
COVID_START = pd.Timestamp("2020-04-01")
COVID_END   = pd.Timestamp("2021-12-01")
PUB_LAG     = 2
N_BETA_LAGS = 2
N_TOP_PCS   = 4
HAC_LAGS    = PUB_LAG - 1


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


# ── Data + no-LASSO VPC1–VPC4 (frozen) ──────────────────────────────────────────
trends = pd.read_csv(ROOT / "calibrated_trends_custom.csv", index_col=0, parse_dates=True).resample("MS").first()
unemp = pd.read_csv(ROOT / "unemp_csv.csv", sep=";", decimal=",",
                    encoding="utf-8-sig", index_col=0, parse_dates=False)
unemp.columns = ["u"]
unemp = unemp[unemp.index.notna() & (unemp.index.astype(str) != "nan")]
unemp.index = pd.to_datetime(unemp.index, format="%Y-%m")
unemp_raw = unemp.resample("MS").first()
u_full = unemp_raw["u"].astype(float).dropna()

deseas = pd.DataFrame(index=trends.index, columns=trends.columns, dtype=float)
for c in trends.columns:
    deseas[c] = frozen_deseas(trends[c].astype(float), FREEZE)
diff = deseas.diff(12).dropna()
tm = diff.index <= FREEZE
scaler = StandardScaler().fit(diff[tm].values)
pca = PCA(n_components=N_TOP_PCS).fit(scaler.transform(diff[tm].values))
rot, Rv = varimax(pca.components_)
VPC = pd.DataFrame(pca.transform(scaler.transform(diff.values)) @ Rv,
                   index=diff.index, columns=[f"VPC{i+1}" for i in range(N_TOP_PCS)])
ytr = u_full.reindex(diff.index[tm]); ok = ytr.notna()
for c in VPC.columns:
    if np.corrcoef(VPC.loc[diff.index[tm], c].values[ok.values], ytr[ok].values)[0, 1] < 0:
        VPC[c] = -VPC[c]
print("Built no-LASSO VPC1–VPC4 (frozen).")


def run_b5(factor_col, ar, covid):
    scores_df = VPC[[factor_col]].rename(columns={factor_col: "F"})
    dates = u_full[(u_full.index >= EVAL_START) & (u_full.index <= EVAL_END)].index
    out, prev = pd.Series(index=dates, dtype=float), None
    for t in dates:
        u_vis = unemp_raw.loc[unemp_raw.index <= t, "u"].copy()
        lc = t - pd.DateOffset(months=PUB_LAG - 1)
        if u_vis.loc[u_vis.index < lc].dropna().shape[0] < 12: continue
        pa = scores_df.loc[scores_df.index <= t]
        try:
            comb = pa[["F"]].join(u_vis.rename("u"), how="left").dropna(subset=["F"])[["u", "F"]]
            if len(comb) < 24: continue
            arr = comb.values.astype(float); arr[comb.index >= lc, 0] = np.nan
            tr = comb.index < lc
            mu_e = np.nanmean(arr[tr], axis=0); sd_e = np.nanstd(arr[tr], axis=0); sd_e[sd_e < 1e-8] = 1.0
            endog = (arr - mu_e) / sd_e
            n = len(comb); lpm = np.zeros((n, N_BETA_LAGS)); pc = endog[:, 1]
            for l in range(1, N_BETA_LAGS + 1): lpm[l:, l - 1] = pc[:-l]
            cov = (((comb.index >= COVID_START) & (comb.index <= COVID_END)).astype(float) / sd_e[0]
                   if covid else None)
            mdl = OptionBModel(endog, ar, 2, lpm, N_BETA_LAGS, cov)
            res = (mdl.fit(start_params=prev, method="powell", maxiter=600, disp=False)
                   if prev is not None and len(prev) == len(mdl.start_params)
                   else mdl.fit(start_params=mdl.fit(method="nm", maxiter=3000, disp=False).params,
                                method="powell", maxiter=1000, disp=False))
            prev = res.params
            out[t] = float(res.filter_results.filtered_state[0, -1]) * sd_e[0] + mu_e[0]
        except Exception:
            prev = None
    return out


# ── Candidate grid: VPC1–4 × {COVID dummy, none}, AR(1), K=2 ─────────────────────
GRID = []
for f in ["VPC1", "VPC2", "VPC3", "VPC4"]:
    GRID.append((f"{f} AR1 cov",   f, 1, True))
    GRID.append((f"{f} AR1 nocov", f, 1, False))
print(f"Running {len(GRID)} no-LASSO state-space candidates …")
cand = {}
for lbl, f, ar, cov in GRID:
    cand[lbl] = run_b5(f, ar, cov)
    print(f"  done: {lbl}")

# AR benchmarks in the pool (LASSO-free: from the standalone 10_benchmarks.py)
e = pd.read_csv(ROOT / "ar_benchmarks.csv", index_col=0, parse_dates=True)
AR1 = e["AR1"]; AR2 = e["AR2"]
cand["AR(1) benchmark"] = AR1
cand["AR(2) benchmark"] = AR2

actual = u_full[(u_full.index >= EVAL_START) & (u_full.index <= EVAL_END)]
covid = (actual.index >= COVID_START) & (actual.index <= COVID_END)

def rmse(f, lo, hi, excl=False):
    f = f.reindex(actual.index); m = (actual.index >= lo) & (actual.index <= hi) & actual.notna() & f.notna()
    if excl: m &= ~covid
    return float(np.sqrt(np.mean((actual[m] - f[m]) ** 2)))

# rolling expanding annual selection
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

print("\n" + "=" * 64)
print("Candidate menu — full-window RMSE (eval 2018–2025)")
print("=" * 64)
menu = pd.Series({n: rmse(f, EVAL_START, EVAL_END) for n, f in cand.items()}).sort_values()
for n, r in menu.items():
    print(f"  {n:<20}{r:>9.4f}  (exC {rmse(cand[n],EVAL_START,EVAL_END,True):.4f})")

print("\nRolling yearly picks:")
for yr, p in picks.items():
    print(f"  {yr}: {p}")

print("\n" + "=" * 64)
print("ROLLING-SELECTED (no-LASSO grid) — test 2019–2025")
print("=" * 64)
print(f"{'Model':<30}{'RMSE full':>11}{'RMSE exC':>11}")
print("-" * 64)
print(f"{'Rolling-selected (no-LASSO)':<30}{rmse(fc,lo,hi):>11.4f}{rmse(fc,lo,hi,True):>11.4f}")
print(f"{'AR(1)':<30}{rmse(AR1,lo,hi):>11.4f}{rmse(AR1,lo,hi,True):>11.4f}")
print(f"{'AR(2)':<30}{rmse(AR2,lo,hi):>11.4f}{rmse(AR2,lo,hi,True):>11.4f}")
print("-" * 64)
c = cw(AR1, fc)
print(f"Rolling vs AR(1): Clark–West t={c[0]:+.2f} (p={c[1]:.3f})")
print("=" * 64)

pd.DataFrame({"actual": actual, "rolling_nolasso": fc, "AR1": AR1, "AR2": AR2}
             ).rename_axis("Tarih").to_csv(HERE / "nolasso_rolling_forecasts.csv")
menu.to_frame("rmse_full").to_csv(HERE / "nolasso_rolling_menu.csv")
pd.Series(picks, name="pick").rename_axis("year").to_frame().to_csv(HERE / "nolasso_rolling_picks.csv")
print("\nSaved: nolasso_rolling_forecasts.csv, nolasso_rolling_menu.csv, nolasso_rolling_picks.csv")
