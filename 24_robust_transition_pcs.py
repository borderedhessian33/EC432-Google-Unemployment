"""
EC 432 Project — Google factors in the TRANSITION equation
===========================================================
Variant of Option A in which the PCs drive the latent state instead of the
observation.  The search signal shifts the persistent unemployment factor and
then propagates through its AR dynamics (a distributed-lag effect), rather than
entering the measurement equation as a static add-on:

    u_t   = μ_t + ε^u_t                                   (measurement: u loads on μ only)
    μ_t   = φ_1 μ_{t-1} + … + φ_p μ_{t-p} + γ' Z_t + η_t  (PCs enter the TRANSITION)
    Z_t   = (F_t, F_{t-1}, …, F_{t-K})

    nowcast:  û_{t|t} = μ̂_{t|t}   (μ̂ already contains the Google input γ'Z_t)

Implemented as a custom statsmodels MLEModel with a time-varying state intercept
c_{t-1}[0] = γ'Z_t (statsmodels timing α_t = T α_{t-1} + c_{t-1} + R η).
Same frozen pipeline, 2-month publication lag, expanding-window re-estimation.
Specs are screened by lowest recursive OOS RMSE (as requested).

Output: statespace_transition_specs.csv, statespace_transition_forecasts.csv
"""

import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy.stats import norm
from statsmodels.tsa.statespace.mlemodel import MLEModel
from statsmodels.tsa.statespace.tools import (constrain_stationary_univariate,
                                              unconstrain_stationary_univariate)
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

SPECS = [
    (["VPC1"],         1, 0),
    (["VPC1"],         2, 1),
    (["VPC1", "VPC3"], 1, 1),
    (["VPC1", "VPC3"], 2, 1),
]


# ── Frozen pipeline → VPCs (identical to 15) ────────────────────────────────────
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


# ── Custom state space: PCs drive the transition ────────────────────────────────
class TransitionExog(MLEModel):
    """u_t = μ_t + ε ;  μ_t = φ(L)μ + γ'Z_t + η  (Z enters the state intercept)."""
    def __init__(self, endog, exog, ar_order):
        self.ar_order = ar_order
        self.q = exog.shape[1]
        self._exog = np.asarray(exog, dtype=float)
        super().__init__(endog, k_states=ar_order, k_posdef=1)
        for i in range(1, ar_order):                 # companion sub-diagonal
            self["transition", i, i - 1] = 1.0
        self["selection", 0, 0] = 1.0
        self["design", 0, 0] = 1.0                   # u loads on μ_t only
        self.initialize_approximate_diffuse()

    @property
    def param_names(self):
        return ([f"phi{i+1}" for i in range(self.ar_order)] +
                [f"gamma{i+1}" for i in range(self.q)] + ["sig_u", "sig_eta"])

    @property
    def start_params(self):
        phi = [0.7] + [0.0] * (self.ar_order - 1)
        return np.array(phi + [0.0] * self.q + [0.4, 0.3])

    def transform_params(self, u):
        p = u.copy()
        p[:self.ar_order] = constrain_stationary_univariate(u[:self.ar_order])
        p[-2:] = np.exp(u[-2:])
        return p

    def untransform_params(self, c):
        p = c.copy()
        p[:self.ar_order] = unconstrain_stationary_univariate(c[:self.ar_order])
        p[-2:] = np.log(np.maximum(c[-2:], 1e-10))
        return p

    def update(self, params, **kwargs):
        params = super().update(params, **kwargs)
        phi   = params[:self.ar_order]
        gamma = params[self.ar_order:self.ar_order + self.q]
        sig_u, sig_eta = params[-2], params[-1]
        self["transition", 0, :self.ar_order] = phi
        gZ = self._exog @ gamma                      # contemporaneous Google input
        c = np.zeros((self.ar_order, self.nobs))
        c[0, :-1] = gZ[1:]                           # c_{t-1}[0] = γ'Z_t  (timing)
        c[0, -1]  = gZ[-1]
        self["state_intercept"] = c
        self["obs_cov", 0, 0]   = sig_u ** 2
        self["state_cov", 0, 0] = sig_eta ** 2


def fit_model(endog, exog, p, prev):
    mdl = TransitionExog(endog, exog, p)
    try:
        if prev is not None and len(prev) == len(mdl.start_params):
            res = mdl.fit(start_params=prev, method="lbfgs", maxiter=100, disp=False)
        else:
            r0  = mdl.fit(method="nm", maxiter=3000, disp=False)
            res = mdl.fit(start_params=r0.params, method="lbfgs", maxiter=300, disp=False)
        if not np.isfinite(res.llf):
            return None
        return res
    except Exception:
        return None


def build_Z(pcs, K):
    Z = pd.DataFrame(index=vpc.index)
    for j in pcs:
        for k in range(0, K + 1):
            Z[f"{j}_l{k}"] = vpc[j].shift(k)
    return Z


def recursive_transition(pcs, p, K):
    Z = build_Z(pcs, K)
    idx_all = u_full.index.intersection(Z.dropna().index)
    eval_dates = u_full[(u_full.index >= EVAL_START) & (u_full.index <= EVAL_END)].index
    out = pd.Series(index=eval_dates, dtype=float)
    prev = None
    for t in eval_dates:
        idx = idx_all[idx_all <= t]
        if len(idx) < 30:
            continue
        cutoff = t - pd.DateOffset(months=PUB_LAG - 1)        # mask u_{t-1}, u_t
        u_arr = u_full.reindex(idx).values.astype(float)
        train = idx < cutoff
        mu_e, sd_e = np.nanmean(u_arr[train]), np.nanstd(u_arr[train])
        sd_e = sd_e if sd_e > 1e-8 else 1.0
        endog = (u_arr - mu_e) / sd_e
        endog[idx >= cutoff] = np.nan
        exog = Z.reindex(idx).values
        res = fit_model(endog, exog, p, prev)
        if res is None:
            prev = None
            continue
        prev = res.params
        mu_hat = res.filter_results.filtered_state[0, -1]      # μ̂_{t|t} (incl. γ'Z_t)
        out[t] = mu_hat * sd_e + mu_e
    return out


# ── Run specs ────────────────────────────────────────────────────────────────────
def rmse(actual, f, excl=False):
    m = actual.notna() & f.notna()
    if excl:
        m &= ~((actual.index >= COVID_START) & (actual.index <= COVID_END))
    return float(np.sqrt(np.mean((actual[m] - f[m]) ** 2)))

print("\nRunning recursive OOS for each specification (custom MLE refit per month) …")
results, forecasts = [], {}
for pcs, p, K in SPECS:
    name = f"{'+'.join(pcs)} | p={p} | K={K}"
    fc = recursive_transition(pcs, p, K)
    actual = u_full.reindex(fc.index)
    rf, re = rmse(actual, fc), rmse(actual, fc, True)
    results.append({"spec": name, "pcs": "+".join(pcs), "p": p, "K": K,
                    "rmse_full": rf, "rmse_excovid": re,
                    "n": int((actual.notna() & fc.notna()).sum())})
    forecasts[name] = fc
    print(f"  {name:<26}  RMSE_full={rf:.4f}  RMSE_exCOVID={re:.4f}  (n={results[-1]['n']})")

res_df = pd.DataFrame(results).sort_values("rmse_full").reset_index(drop=True)
best = res_df.iloc[0]; best_fc = forecasts[best["spec"]]
actual = u_full.reindex(best_fc.index)


# ── DM / CW vs AR(1) + reference RMSEs ───────────────────────────────────────────
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

ref, ar1 = {}, None
fp = HERE / "option_b_lag2_forecasts.csv"
if fp.exists():
    e = pd.read_csv(fp); e["Tarih"] = pd.to_datetime(e["Tarih"]); e = e.set_index("Tarih")
    ar1 = e["AR1"].reindex(actual.index)
    ref["AR(1) recursive"] = (rmse(actual, ar1), rmse(actual, ar1, True))
    if "B5_VPC1_AR1" in e:
        b5 = e["B5_VPC1_AR1"].reindex(actual.index)
        ref["B5 full multivariate SS"] = (rmse(actual, b5), rmse(actual, b5, True))
sp = HERE / "statespace_exog_forecasts.csv"
if sp.exists():
    a = pd.read_csv(sp); a["Tarih"] = pd.to_datetime(a["Tarih"]); a = a.set_index("Tarih")
    if "OptionA_best" in a:
        oa = a["OptionA_best"].reindex(actual.index)
        ref["Option A (exog in measurement)"] = (rmse(actual, oa), rmse(actual, oa, True))

print("\n" + "=" * 74)
print("PCs IN TRANSITION — specs ranked by full-window RMSE")
print("=" * 74)
print(f"{'spec':<28}{'RMSE full':>11}{'RMSE exC':>11}")
print("-" * 74)
for _, r in res_df.iterrows():
    star = "  ← best" if r["spec"] == best["spec"] else ""
    print(f"{r['spec']:<28}{r['rmse_full']:>11.4f}{r['rmse_excovid']:>11.4f}{star}")
print("-" * 74)
for lbl, (rf, re) in ref.items():
    print(f"{'[ref] '+lbl:<28}{rf:>11.4f}{re:>11.4f}")
print("=" * 74)
if ar1 is not None:
    dm = dm_test(ar1, best_fc); cw = cw_test(ar1, best_fc)
    print(f"\nBest spec: {best['spec']}")
    print(f"  vs AR(1):  DM t = {dm[0]:+.2f} (p={dm[1]:.3f}),  "
          f"Clark–West t = {cw[0]:+.2f} (p={cw[1]:.3f})")

res_df.to_csv(HERE / "statespace_transition_specs.csv", index=False)
fc_out = pd.DataFrame({"actual": actual, "Transition_best": best_fc})
if ar1 is not None:
    fc_out["AR1"] = ar1
fc_out.index.name = "Tarih"
fc_out.to_csv(HERE / "statespace_transition_forecasts.csv")
print("\nSaved: statespace_transition_specs.csv, statespace_transition_forecasts.csv")

fig, ax = plt.subplots(figsize=(11, 4.4))
ax.axvspan(COVID_START, COVID_END, color="gold", alpha=0.18)
ax.plot(actual.index, actual.values, color="black", lw=2.0, label="Actual $u_t$")
if ar1 is not None:
    ax.plot(ar1.index, ar1.values, color="#7F8C8D", lw=1.3, ls="--", label="AR(1)")
ax.plot(best_fc.index, best_fc.values, color="#8E44AD", lw=1.7,
        label=f"PCs-in-transition best ({best['spec']})")
ax.set_ylabel("Non-agricultural unemployment (%)")
ax.set_title("Google factors in the transition equation (best spec)")
ax.legend(loc="upper right"); ax.grid(axis="y", ls="--", alpha=0.35)
plt.tight_layout()
plt.show()
