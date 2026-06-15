"""
EC 432 Project — Bridge-depth robustness sweep  (PAPER TABLE 8)
===============================================================
Re-estimates the locked B5-type state space at three bridge depths

    L = 0  (no bridge: factors reach u only through the shared latent state)
    L = 1  (one lagged factor in the u-equation)
    L = 2  (two lagged factors — the headline depth, spans the 2-month gap)

for the labour-relevant and the weak (non-labour) factor sets, holding the rest of
the backbone fixed at (p, D) = (1, 2) with a constant COVID dummy.  Everything else
(frozen no-LASSO PCA+varimax factors, recursive 2-month-publication-lag design,
eval 2018-2025) is identical to 11_candidate_grid.py — only L changes.

The L = 2 column reproduces the COVID-dummy candidates of Table 2 exactly; the
L = 0 / L = 1 columns are the bridge-depth robustness check.

Output: bridge_depth_rmse.csv  (RMSE by factor set × L) ;
        bridge_depth_forecasts.csv  (incremental nowcast cache — delete to rebuild)
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
N_TOP_PCS   = 4
CACHE = HERE / "bridge_depth_forecasts.csv"


# ── State-space model (latent AR(1); u & VPCs load on it; L-lag VPC bridge) ──────
class OptionBModel(MLEModel):
    def __init__(self, endog, ar_order, lag_depth, lpm, n_beta_lags=2, covid_col=None):
        self.ar_order = ar_order; self.lag_depth = lag_depth
        self.n_pcs = endog.shape[1] - 1; self.n_beta_lags = n_beta_lags
        self.use_covid = covid_col is not None
        k_s = max(ar_order, lag_depth + 1)
        super().__init__(endog, k_states=k_s, k_posdef=1)
        lpm = np.asarray(lpm, float)
        self._lpm = np.where(np.isfinite(lpm), lpm, 0.0).T if lpm.size else np.zeros((0, self.nobs))
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
        nb = self.n_pcs * self.n_beta_lags
        betas = params[idx: idx + nb]; bc = betas @ self._lpm if nb else np.zeros(self.nobs)
        u_int = bc + params[idx + nb] * self._covid if self.use_covid else bc
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


# ── Recursive OOS nowcast for one (factor set, COVID, bridge depth L) ────────────
def run_candidate(pcs, covid, L):
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
            n = len(comb); npc = len(pcs); lpm = np.zeros((n, npc * L))
            for pi in range(npc):
                pc = endog[:, pi + 1]
                for l in range(1, L + 1): lpm[l:, pi * L + (l - 1)] = pc[:-l]
            cov = (((comb.index >= COVID_START) & (comb.index <= COVID_END)).astype(float) / sd_e[0]
                   if covid else None)
            mdl = OptionBModel(endog, 1, 2, lpm, L, cov)
            res = (mdl.fit(start_params=prev, method="powell", maxiter=600, disp=False)
                   if prev is not None and len(prev) == len(mdl.start_params)
                   else mdl.fit(start_params=mdl.fit(method="nm", maxiter=3000, disp=False).params,
                                method="powell", maxiter=1000, disp=False))
            prev = res.params
            out[t] = float(res.filter_results.filtered_state[0, -1]) * sd_e[0] + mu_e[0]
        except Exception:
            prev = None
    return out


# ── Factor sets × bridge depths (all COVID-dummy specs) ──────────────────────────
LABOUR = {"VPC3": ["VPC3"], "VPC3+VPC4": ["VPC3", "VPC4"], "VPC1+VPC3": ["VPC1", "VPC3"],
          "VPC4": ["VPC4"], "VPC1": ["VPC1"]}
WEAK   = {"VPC2": ["VPC2"], "VPC2+VPC4": ["VPC2", "VPC4"]}
SETS = {**LABOUR, **WEAK}
DEPTHS = [0, 1, 2]

# incremental cache keyed "set@L"
if CACHE.exists():
    fc_df = pd.read_csv(CACHE, index_col=0, parse_dates=True)
    fcs = {c: fc_df[c] for c in fc_df.columns}
else:
    fcs = {}
todo = [(s, L) for s in SETS for L in DEPTHS if f"{s}@L{L}" not in fcs]
if todo:
    print(f"Computing {len(todo)} (set, L) nowcasts …")
    for s, L in todo:
        fcs[f"{s}@L{L}"] = run_candidate(SETS[s], True, L)
        print(f"  done {s} @ L={L}")
    pd.DataFrame({f"{s}@L{L}": fcs[f"{s}@L{L}"] for s in SETS for L in DEPTHS}).to_csv(CACHE)
    print(f"  saved -> {CACHE.name}")
else:
    print("All bridge-depth nowcasts already cached.")


# ── RMSE table (PAPER TABLE 8) ───────────────────────────────────────────────────
actual = u_full[(u_full.index >= EVAL_START) & (u_full.index <= EVAL_END)]
def rmse(f):
    f = f.reindex(actual.index); m = actual.notna() & f.notna()
    return float(np.sqrt(np.mean((actual[m] - f[m]) ** 2)))

ar1 = rmse(pd.read_csv(HERE / "ar_benchmarks.csv", index_col=0, parse_dates=True)["AR1"])
tbl = pd.DataFrame({f"L={L}": {s: rmse(fcs[f"{s}@L{L}"]) for s in SETS} for L in DEPTHS})

print("\n" + "=" * 54)
print(f"TABLE 8 — bridge-depth RMSE, 2018-2025  (AR(1) = {ar1:.3f})")
print("=" * 54)
print("Labour-relevant factors")
print(tbl.loc[list(LABOUR)].round(3).to_string())
print("\nWeak (non-labour) factors")
print(tbl.loc[list(WEAK)].round(3).to_string())
print("=" * 54)
print("Best depth per set:")
for s in SETS:
    best = tbl.loc[s].idxmin()
    print(f"  {s:<12} -> {best}  ({tbl.loc[s, best]:.3f})")

tbl.round(4).to_csv(HERE / "bridge_depth_rmse.csv")
print("\nSaved: bridge_depth_rmse.csv")
