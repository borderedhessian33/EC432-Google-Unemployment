"""
EC 432 Project — Candidate models (locked L=2 grid, LASSO-free)
================================================================
The canonical candidate pool for the paper.  Factors are built WITHOUT LASSO:
frozen deseasonalise -> 12-month YoY diff -> standardise -> PCA(4) on ALL 21
keywords -> varimax -> sign-anchor (+corr with u), all frozen at Dec-2017.

Every candidate is the same B5-type state space with the fixed backbone
    AR order p = 1,  loading depth D = 2,  bridge depth L = 2,
estimated recursively out-of-sample (expanding window, 2-month publication lag),
2018-01 ... 2025-12.  Candidates differ ONLY in:
    • factor set : 4 singletons {VPC1..VPC4} + 6 pairs            (10 sets)
    • COVID term : constant dummy  or  none                       (2 options)
=> 10 x 2 = 20 candidates.

No special variants: the time-varying-COVID model and the PLS factor are NOT
candidates and are excluded by design.  Benchmarks (AR1/AR2) live in 10_benchmarks.py.

Output: candidate_forecasts.csv  (20 columns; incremental cache — delete to rebuild)
"""

import warnings
from pathlib import Path
import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.mlemodel import MLEModel
from statsmodels.tsa.seasonal import STL
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

warnings.filterwarnings("ignore")
HERE = Path(__file__).parent
EVAL_START  = pd.Timestamp("2018-01-01")
EVAL_END    = pd.Timestamp("2025-12-01")
FREEZE      = pd.Timestamp("2017-12-31")
COVID_START = pd.Timestamp("2020-04-01")
COVID_END   = pd.Timestamp("2021-12-01")
PUB_LAG     = 2
N_BETA_LAGS = 2          # bridge depth L
N_TOP_PCS   = 4
CACHE = HERE / "candidate_forecasts.csv"


# ── State-space model (AR(p) latent factor; u and VPCs load on it; lagged-VPC bridge) ──
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


# ── Data + frozen no-LASSO VPC factors (all 21 keywords) ────────────────────────
trends = pd.read_csv(HERE / "calibrated_trends_custom.csv", index_col=0, parse_dates=True).resample("MS").first()
unemp = pd.read_csv(HERE / "unemp_csv.csv", sep=";", decimal=",",
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
print("No-LASSO VPC1-4 factors ready (frozen Dec-2017).")


# ── Recursive OOS nowcast for one candidate (AR1, D=2, L=2; COVID dummy optional) ──
def run_candidate(pcs, covid):
    scores = VPC[pcs]
    dates = u_full[(u_full.index >= EVAL_START) & (u_full.index <= EVAL_END)].index
    out, prev = pd.Series(index=dates, dtype=float), None
    for t in dates:
        u_vis = unemp_raw.loc[unemp_raw.index <= t, "u"].copy()
        lc = t - pd.DateOffset(months=PUB_LAG - 1)
        if u_vis.loc[u_vis.index < lc].dropna().shape[0] < 12: continue
        pa = scores.loc[scores.index <= t]
        try:
            comb = pa[pcs].join(u_vis.rename("u"), how="left").dropna(subset=pcs)[["u"] + pcs]
            if len(comb) < 24: continue
            arr = comb.values.astype(float); arr[comb.index >= lc, 0] = np.nan
            tr = comb.index < lc
            mu_e = np.nanmean(arr[tr], axis=0); sd_e = np.nanstd(arr[tr], axis=0); sd_e[sd_e < 1e-8] = 1.0
            endog = (arr - mu_e) / sd_e
            n = len(comb); npc = len(pcs); lpm = np.zeros((n, npc * N_BETA_LAGS))
            for pi in range(npc):
                pc = endog[:, pi + 1]
                for l in range(1, N_BETA_LAGS + 1): lpm[l:, pi * N_BETA_LAGS + (l - 1)] = pc[:-l]
            cov = (((comb.index >= COVID_START) & (comb.index <= COVID_END)).astype(float) / sd_e[0]
                   if covid else None)
            mdl = OptionBModel(endog, 1, 2, lpm, N_BETA_LAGS, cov)
            res = (mdl.fit(start_params=prev, method="powell", maxiter=600, disp=False)
                   if prev is not None and len(prev) == len(mdl.start_params)
                   else mdl.fit(start_params=mdl.fit(method="nm", maxiter=3000, disp=False).params,
                                method="powell", maxiter=1000, disp=False))
            prev = res.params
            out[t] = float(res.filter_results.filtered_state[0, -1]) * sd_e[0] + mu_e[0]
        except Exception:
            prev = None
    return out


# ── The 20 locked candidates: 10 factor sets x {COVID dummy, none} ───────────────
SINGLES = [["VPC1"], ["VPC2"], ["VPC3"], ["VPC4"]]
PAIRS   = [["VPC1","VPC2"], ["VPC1","VPC3"], ["VPC1","VPC4"],
           ["VPC2","VPC3"], ["VPC2","VPC4"], ["VPC3","VPC4"]]
FACTOR_SETS = SINGLES + PAIRS

specs = {}
for pcs in FACTOR_SETS:
    base = "+".join(pcs)
    specs[base + "_cov"]   = (pcs, True)
    specs[base + "_nocov"] = (pcs, False)

# incremental cache: load existing, compute only candidates not yet present
if CACHE.exists():
    cand_df = pd.read_csv(CACHE, index_col=0, parse_dates=True)
    cand = {c: cand_df[c] for c in cand_df.columns}
else:
    cand = {}
missing = [lbl for lbl in specs if lbl not in cand]
if missing:
    print(f"Computing {len(missing)} candidate(s): {missing}")
    for lbl in missing:
        pcs, covid = specs[lbl]
        cand[lbl] = run_candidate(pcs, covid); print(f"  done {lbl}")
    pd.DataFrame({k: cand[k] for k in specs}).to_csv(CACHE)
    print(f"  saved -> {CACHE.name}")
else:
    print("All 20 candidates already cached.")


# ── Sanity menu (RMSE full / ex-COVID) ──────────────────────────────────────────
actual = u_full[(u_full.index >= EVAL_START) & (u_full.index <= EVAL_END)]
covid_mask = (actual.index >= COVID_START) & (actual.index <= COVID_END)
def rmse(f, excl=False):
    f = f.reindex(actual.index); m = actual.notna() & f.notna()
    if excl: m &= ~covid_mask
    return float(np.sqrt(np.mean((actual[m] - f[m]) ** 2)))

menu = pd.Series({lbl: rmse(cand[lbl]) for lbl in specs}).sort_values()
print("\n" + "=" * 52)
print(f"LOCKED CANDIDATE GRID — {len(specs)} models (eval 2018–2025)")
print("=" * 52)
print(f"{'Model':<20}{'RMSE full':>11}{'RMSE exC':>11}")
print("-" * 52)
for lbl, r in menu.items():
    print(f"{lbl:<20}{r:>11.4f}{rmse(cand[lbl], True):>11.4f}")
print("=" * 52)
