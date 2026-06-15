"""
EC 432 Project — B5 with the PLS targeted factor (instead of PCA-VPC1)
======================================================================
Keep B5's structure EXACTLY (VPC→ now PLS1; AR(1) latent state; factor companion
ld=2; β-lags=2; COVID dummy; 2-month publication lag; recursive expanding window;
filtered-state extraction).  The ONLY change: the Google factor is the first PLS
component (supervised, targets u) rather than the LASSO→PCA→varimax VPC1.

PLS is fit on the frozen training window (≤Dec-2017) and applied forward — same
frozen discipline as the PCA pipeline (no look-ahead; PLS uses u only through 2017).

Compares B5-PLS to the published B5 (PCA-VPC1, 0.534) and AR(1).
Output: b5_pls_forecasts.csv + printed comparison.
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
from sklearn.cross_decomposition import PLSRegression

warnings.filterwarnings("ignore")
HERE = Path(__file__).parent

EVAL_START   = pd.Timestamp("2018-01-01")
EVAL_END     = pd.Timestamp("2025-12-01")
FREEZE_DATE  = EVAL_START - pd.DateOffset(months=1)
PUB_LAG      = 2
N_BETA_LAGS  = 2
COVID_START  = pd.Timestamp("2020-04-01")
COVID_END    = pd.Timestamp("2021-12-01")
HAC_LAGS     = PUB_LAG - 1
N_PLS        = 1          # single targeted factor = direct B5 analog


# ── B5 state-space model — verbatim from 11_candidate_grid.py ─────────────────
class OptionBModel(MLEModel):
    def __init__(self, endog, ar_order, lag_depth, lagged_pc_matrix,
                 n_beta_lags=2, covid_col=None):
        self.ar_order = ar_order; self.lag_depth = lag_depth
        self.n_pcs = endog.shape[1] - 1; self.n_beta_lags = n_beta_lags
        self.use_covid = covid_col is not None
        k_s = max(ar_order, lag_depth + 1)
        super().__init__(endog, k_states=k_s, k_posdef=1)
        lpm = np.asarray(lagged_pc_matrix, dtype=float)
        self._lpm = np.where(np.isfinite(lpm), lpm, 0.0).T
        self._covid = (np.asarray(covid_col, dtype=float).flatten()
                       if covid_col is not None else np.zeros(self.nobs))
        for i in range(1, k_s):
            self["transition", i, i - 1] = 1.0
        self["selection", 0, 0] = 1.0
        self["design", 0, 0] = 1.0
        self["obs_intercept"] = np.zeros((1 + self.n_pcs, self.nobs))
        self.initialize_approximate_diffuse()

    @property
    def _noise_idx(self):
        return self.ar_order + self.n_pcs * (self.lag_depth + 1)

    @property
    def param_names(self):
        names = [f"phi{i+1}" for i in range(self.ar_order)]
        for p in range(self.n_pcs):
            for j in range(self.lag_depth + 1):
                names.append(f"l{chr(ord('a')+p)}{j+1}")
        names.append("sig_u")
        for p in range(self.n_pcs):
            names.append(f"sig_pc{chr(ord('a')+p)}")
        names.append("sig_eta")
        for p in range(self.n_pcs):
            for l in range(1, self.n_beta_lags + 1):
                names.append(f"beta_{chr(ord('a')+p)}_lag{l}")
        if self.use_covid:
            names.append("delta_covid")
        return names

    @property
    def start_params(self):
        parts = [0.7] + [0.1] * (self.ar_order - 1)
        for _ in range(self.n_pcs):
            parts += [0.5] + [0.1] * self.lag_depth
        parts += [0.3] * (1 + self.n_pcs) + [0.3]
        parts += [0.05] * (self.n_pcs * self.n_beta_lags)
        if self.use_covid:
            parts.append(0.0)
        return np.array(parts)

    def transform_params(self, u):
        p = u.copy()
        for i in range(self.ar_order):
            p[i] = np.tanh(u[i])
        ni = self._noise_idx
        for i in range(1 + self.n_pcs + 1):
            p[ni + i] = np.exp(u[ni + i])
        return p

    def untransform_params(self, c):
        p = c.copy()
        for i in range(self.ar_order):
            p[i] = np.arctanh(np.clip(c[i], -0.9999, 0.9999))
        ni = self._noise_idx
        for i in range(1 + self.n_pcs + 1):
            p[ni + i] = np.log(max(c[ni + i], 1e-12))
        return p

    def update(self, params, **kwargs):
        params = super().update(params, **kwargs)
        for i in range(self.ar_order):
            self["transition", 0, i] = params[i]
        idx = self.ar_order
        for p_i in range(self.n_pcs):
            for j in range(self.lag_depth + 1):
                self["design", p_i + 1, j] = params[idx]; idx += 1
        self["obs_cov", 0, 0] = params[idx] ** 2; idx += 1
        for p_i in range(self.n_pcs):
            self["obs_cov", p_i + 1, p_i + 1] = params[idx] ** 2; idx += 1
        self["state_cov", 0, 0] = params[idx] ** 2; idx += 1
        betas = params[idx: idx + self.n_pcs * self.n_beta_lags]
        bc = betas @ self._lpm
        u_int = bc + params[idx + len(betas)] * self._covid if self.use_covid else bc
        oi = np.zeros((1 + self.n_pcs, self.nobs)); oi[0, :] = u_int
        self["obs_intercept"] = oi


def frozen_deseas(s, fz):
    tr = s[s.index <= fz].dropna()
    if len(tr) < 24:
        return s
    stl = STL(tr, period=12, robust=False).fit()
    pat = stl.seasonal.groupby(stl.seasonal.index.month).mean()
    return s - pd.Series(s.index.month, index=s.index).map(pat).values


# ── Data + frozen PLS factor ─────────────────────────────────────────────────────
print("Loading data & building frozen PLS-1 factor …")
trends = pd.read_csv(HERE / "calibrated_trends_custom.csv", index_col=0, parse_dates=True)
trends.index = pd.to_datetime(trends.index).to_period("M").to_timestamp()
trends = trends.resample("MS").first()
unemp_raw = pd.read_csv(HERE / "unemp_csv.csv", sep=";", decimal=",",
                        encoding="utf-8-sig", index_col=0, parse_dates=False)
unemp_raw.columns = ["u"]
unemp_raw = unemp_raw[unemp_raw.index.notna() & (unemp_raw.index.astype(str) != "nan")]
unemp_raw.index = pd.to_datetime(unemp_raw.index, format="%Y-%m")
unemp_raw = unemp_raw.resample("MS").first()
unemp_raw["u"] = unemp_raw["u"].astype(float)
u_full = unemp_raw["u"].dropna()

deseas = pd.DataFrame(index=trends.index, columns=trends.columns, dtype=float)
for c in trends.columns:
    deseas[c] = frozen_deseas(trends[c].astype(float), FREEZE_DATE)
diff = deseas.diff(12).dropna()
tmask = diff.index <= FREEZE_DATE
scaler = StandardScaler().fit(diff[tmask])
Xall = pd.DataFrame(scaler.transform(diff), index=diff.index, columns=diff.columns)

# DIAGNOSTIC: fit PLS on the WHOLE sample (LOOK-AHEAD — upper bound, not valid OOS)
y_fit = u_full.reindex(diff.index)
okf = y_fit.notna().values                       # every month with observed u (2011..2025)
pls = PLSRegression(n_components=max(N_PLS, 1), scale=False).fit(
        Xall.values[okf], (y_fit[okf] - y_fit[okf].mean()).values)
scores_full = pls.transform(Xall.values)
if np.corrcoef(scores_full[okf, 0], y_fit[okf].values)[0, 1] < 0:
    scores_full[:, 0] = -scores_full[:, 0]
scores_df = pd.DataFrame(scores_full[:, :N_PLS], index=diff.index,
                         columns=[f"PLS{i+1}" for i in range(N_PLS)])
print("  *** PLS fit on FULL sample — LOOK-AHEAD diagnostic, not a valid OOS result ***")
print(f"  PLS-1 corr with u (full sample) = "
      f"{np.corrcoef(scores_full[okf,0], y_fit[okf].values)[0,1]:+.3f}")


# ── Recursive OOS for B5-PLS (single factor, AR(1), ld=2, K=2, COVID) ───────────
def recursive_b5_pls():
    dates = u_full[(u_full.index >= EVAL_START) & (u_full.index <= EVAL_END)].index
    out, prev = pd.Series(index=dates, dtype=float), None
    pc_cols = list(scores_df.columns)
    for t in dates:
        u_vis = unemp_raw.loc[unemp_raw.index <= t, "u"].copy()
        lag_cutoff = t - pd.DateOffset(months=PUB_LAG - 1)
        if u_vis.loc[u_vis.index < lag_cutoff].dropna().shape[0] < 12:
            continue
        pc_avail = scores_df.loc[scores_df.index <= t]
        try:
            comb = pc_avail[pc_cols].join(u_vis.rename("u"), how="left").dropna(subset=pc_cols)
            comb = comb[["u"] + pc_cols]
            if len(comb) < 24:
                continue
            arr = comb.values.astype(float)
            arr[comb.index >= lag_cutoff, 0] = np.nan
            tr = comb.index < lag_cutoff
            mu_e = np.nanmean(arr[tr], axis=0); sd_e = np.nanstd(arr[tr], axis=0)
            sd_e[sd_e < 1e-8] = 1.0
            endog = (arr - mu_e) / sd_e
            n = len(comb); npc = len(pc_cols)
            lpm = np.zeros((n, npc * N_BETA_LAGS))
            for p_i in range(npc):
                pc_std = endog[:, p_i + 1]
                for l in range(1, N_BETA_LAGS + 1):
                    lpm[l:, p_i * N_BETA_LAGS + (l - 1)] = pc_std[:-l]
            covid = ((comb.index >= COVID_START) & (comb.index <= COVID_END)).astype(float) / sd_e[0]
            mdl = OptionBModel(endog, 1, 2, lpm, N_BETA_LAGS, covid)
            if prev is not None and len(prev) == len(mdl.start_params):
                res = mdl.fit(start_params=prev, method="powell", maxiter=600, disp=False)
            else:
                r0 = mdl.fit(start_params=mdl.start_params, method="nm", maxiter=3000, disp=False)
                res = mdl.fit(start_params=r0.params, method="powell", maxiter=1000, disp=False)
            prev = res.params
            out[t] = float(res.filter_results.filtered_state[0, -1]) * sd_e[0] + mu_e[0]
        except Exception:
            prev = None
    return out

print(f"\nRecursive OOS: B5 with PLS-{N_PLS} factor (2018–2025) …")
b5_pls = recursive_b5_pls()

actual = u_full.reindex(b5_pls.index)
covid = (actual.index >= COVID_START) & (actual.index <= COVID_END)


def rmse(f, excl=False):
    m = actual.notna() & f.notna()
    if excl:
        m &= ~covid
    return float(np.sqrt(np.mean((actual[m] - f[m]) ** 2)))

def dm(f1, f2):
    m = actual.notna() & f1.notna() & f2.notna()
    d = (actual[m] - f1[m]) ** 2 - (actual[m] - f2[m]) ** 2
    r = sm.OLS(d.values, np.ones((len(d), 1))).fit(cov_type="HAC", cov_kwds={"maxlags": HAC_LAGS})
    return float(r.tvalues[0]), float(r.pvalues[0])

def cw(fr, fu):
    m = actual.notna() & fr.notna() & fu.notna()
    y, a, b = actual[m].values, fr[m].values, fu[m].values
    ft = (y - a) ** 2 - ((y - b) ** 2 - (a - b) ** 2)
    r = sm.OLS(ft, np.ones((len(ft), 1))).fit(cov_type="HAC", cov_kwds={"maxlags": HAC_LAGS})
    return float(r.tvalues[0]), float(1 - norm.cdf(r.tvalues[0]))

ar1 = b5_pca = None
fp = HERE / "option_b_lag2_forecasts.csv"
if fp.exists():
    e = pd.read_csv(fp); e["Tarih"] = pd.to_datetime(e["Tarih"]); e = e.set_index("Tarih")
    ar1 = e["AR1"].reindex(actual.index); b5_pca = e["B5_VPC1_AR1"].reindex(actual.index)

print("\n" + "=" * 70)
print("B5-PLS  vs  B5-PCA  vs  AR(1)   (eval 2018–2025, n=96)")
print("=" * 70)
print(f"{'Model':<34}{'RMSE full':>11}{'RMSE exC':>11}")
print("-" * 70)
if ar1 is not None:
    print(f"{'AR(1) benchmark':<34}{rmse(ar1):>11.4f}{rmse(ar1,True):>11.4f}")
if b5_pca is not None:
    print(f"{'B5  (PCA-VPC1, published)':<34}{rmse(b5_pca):>11.4f}{rmse(b5_pca,True):>11.4f}")
print(f"{'B5-PLS (targeted factor)':<34}{rmse(b5_pls):>11.4f}{rmse(b5_pls,True):>11.4f}")
print("-" * 70)
if ar1 is not None:
    d, c = dm(ar1, b5_pls), cw(ar1, b5_pls)
    print(f"B5-PLS vs AR(1):   DM t={d[0]:+.2f} (p={d[1]:.3f}) ;  CW t={c[0]:+.2f} (p={c[1]:.3f})")
if b5_pca is not None:
    d = dm(b5_pca, b5_pls)
    print(f"B5-PLS vs B5-PCA:  DM t={d[0]:+.2f} (p={d[1]:.3f})   (t>0 ⇒ PLS better)")
print("=" * 70)

pd.DataFrame({"actual": actual, "B5_PLS": b5_pls, "B5_PCA": b5_pca, "AR1": ar1}
             ).rename_axis("Tarih").to_csv(HERE / "b5_pls_forecasts.csv")
print("\nSaved: b5_pls_forecasts.csv")
