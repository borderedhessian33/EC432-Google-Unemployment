"""
EC 432 Project — In-sample estimates of the headline-family models (two-step)
==============================================================================
Joint MLE leaves the bridge coefficients weakly identified (the factor channel
absorbs the variation, so the likelihood is flat in beta).  We therefore report
in-sample estimates with the two-step estimator:

  Step 1 — Factor model (MLE, Kalman):
       u_t      = mu_t + eps_u
       VPC_{j,t}= sum_d lambda_{j,d} mu_{t-d} + eps_j ,   mu_t = phi mu_{t-1} + eta
     All parameters well identified (no bridge present).  Yields smoothed mu_hat.

  Step 2 — Bridge regression (OLS, Newey-West HAC):
       (u_t - mu_hat_t) = sum_{j,l} beta_{j,l} VPC_{j,t-l} (+ delta COVID_t) + nu
     Valid t-stats for the bridge because mu_hat is fixed before estimation.

Three nested specifications:
  (1) VPC3                 (2) VPC3 + COVID        (3) VPC3+VPC4 (headline)

Factors are the locked no-LASSO VPCs (frozen Dec-2017); estimation is in-sample
on the full 2011–2024 record.

Output: printed Step-1 / Step-2 tables + insample_step1.csv, insample_step2.csv
"""

import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.statespace.mlemodel import MLEModel
from statsmodels.tsa.seasonal import STL
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

warnings.filterwarnings("ignore")
HERE = Path(__file__).parent
FREEZE = pd.Timestamp("2017-12-31")
COVID_START = pd.Timestamp("2020-04-01")
COVID_END   = pd.Timestamp("2021-12-01")
N_BETA_LAGS = 2
N_TOP_PCS = 4
HAC = 6                                   # Newey-West lags for Step 2


# ── Step-1 factor model: u & VPCs load on a latent AR(1) factor (NO bridge) ──────
class FactorModel(MLEModel):
    def __init__(self, endog, ar_order=1, lag_depth=2):
        self.ar_order = ar_order; self.lag_depth = lag_depth
        self.n_pcs = endog.shape[1] - 1
        k_s = max(ar_order, lag_depth + 1)
        super().__init__(endog, k_states=k_s, k_posdef=1)
        for i in range(1, k_s): self["transition", i, i - 1] = 1.0
        self["selection", 0, 0] = 1.0; self["design", 0, 0] = 1.0
        self.initialize_approximate_diffuse()
    @property
    def _noise_idx(self): return self.ar_order + self.n_pcs * (self.lag_depth + 1)
    @property
    def param_names(self):
        n = [f"phi{i+1}" for i in range(self.ar_order)]
        for p in range(self.n_pcs):
            for j in range(self.lag_depth + 1): n.append(f"lambda_{chr(97+p)}{j}")
        n.append("sig_u")
        for p in range(self.n_pcs): n.append(f"sig_pc{chr(97+p)}")
        n.append("sig_eta")
        return n
    @property
    def start_params(self):
        parts = [0.7] + [0.1] * (self.ar_order - 1)
        for _ in range(self.n_pcs): parts += [0.5] + [0.1] * self.lag_depth
        parts += [0.3] * (1 + self.n_pcs) + [0.3]
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
        self["state_cov", 0, 0] = params[idx] ** 2


def varimax(L, max_iter=1000, tol=1e-6):
    Phi = L.T; p, k = Phi.shape; R = np.eye(k)
    for _ in range(max_iter):
        Lam = Phi @ R; g = Phi.T @ (Lam**3 - Lam * (Lam**2).sum(0) / p)
        U, _, Vt = np.linalg.svd(g); Rn = U @ Vt
        if np.max(np.abs(Rn - R)) < tol: R = Rn; break
        R = Rn
    return (Phi @ R).T, R

def frozen_deseas(s, fz):
    tr = s[s.index <= fz].dropna()
    if len(tr) < 24: return s
    st = STL(tr, period=12, robust=False).fit(); pat = st.seasonal.groupby(st.seasonal.index.month).mean()
    return s - pd.Series(s.index.month, index=s.index).map(pat).values


# ── Data + frozen no-LASSO VPC factors ───────────────────────────────────────────
trends = pd.read_csv(HERE / "calibrated_trends_custom.csv", index_col=0, parse_dates=True).resample("MS").first()
unemp = pd.read_csv(HERE / "unemp_csv.csv", sep=";", decimal=",", encoding="utf-8-sig", index_col=0)
unemp.columns = ["u"]; unemp = unemp[unemp.index.notna() & (unemp.index.astype(str) != "nan")]
unemp.index = pd.to_datetime(unemp.index, format="%Y-%m")
u_full = unemp.resample("MS").first()["u"].astype(float).dropna()

ds = pd.DataFrame(index=trends.index, columns=trends.columns, dtype=float)
for c in trends.columns: ds[c] = frozen_deseas(trends[c].astype(float), FREEZE)
diff = ds.diff(12).dropna(); tm = diff.index <= FREEZE
sc = StandardScaler().fit(diff[tm].values)
pca = PCA(n_components=N_TOP_PCS).fit(sc.transform(diff[tm].values))
rot, Rv = varimax(pca.components_)
VPC = pd.DataFrame(pca.transform(sc.transform(diff.values)) @ Rv, index=diff.index,
                   columns=[f"VPC{i+1}" for i in range(N_TOP_PCS)])
ytr = u_full.reindex(diff.index[tm]); ok = ytr.notna()
for c in VPC.columns:
    if np.corrcoef(VPC.loc[diff.index[tm], c].values[ok.values], ytr[ok].values)[0, 1] < 0:
        VPC[c] = -VPC[c]


def twostep(pcs, covid, label):
    comb = VPC[pcs].join(u_full.rename("u"), how="inner").dropna()[["u"] + pcs]
    arr = comb.values.astype(float)
    mu, sd = arr.mean(0), arr.std(0); sd[sd < 1e-8] = 1.0
    endog = (arr - mu) / sd
    # Step 1: factor model
    m1 = FactorModel(endog, 1, 2)
    r0 = m1.fit(method="nm", maxiter=4000, disp=False)
    r1 = m1.fit(start_params=r0.params, method="bfgs", maxiter=2000, disp=False)
    mu_hat = m1.smooth(r1.params).smoothed_state[0]
    s1 = pd.DataFrame({"coef": r1.params, "se": r1.bse, "t": r1.tvalues, "p": r1.pvalues},
                      index=m1.param_names)
    # Step 2: bridge regression on residual, Newey-West
    resid = endog[:, 0] - mu_hat
    Z, names = [], []
    for pi, name in enumerate(pcs):
        for l in range(1, N_BETA_LAGS + 1):
            Z.append(pd.Series(endog[:, pi + 1], index=comb.index).shift(l).values)
            names.append(f"{name}_lag{l}")
    if covid:
        Z.append(((comb.index >= COVID_START) & (comb.index <= COVID_END)).astype(float))
        names.append("COVID")
    X = np.column_stack([np.ones(len(comb))] + Z)
    keep = ~np.isnan(X).any(1)
    ols = sm.OLS(resid[keep], X[keep]).fit(cov_type="HAC", cov_kwds={"maxlags": HAC})
    s2 = pd.DataFrame({"coef": ols.params, "se": ols.bse, "t": ols.tvalues, "p": ols.pvalues},
                      index=["const"] + names)
    print(f"\n{'='*64}\n{label}   (n={len(comb)})\n{'='*64}")
    print("STEP 1 — factor model (sigma_u = %.3f):" % r1.params[m1._noise_idx]); print(s1.round(3).to_string())
    print("\nSTEP 2 — bridge regression (Newey-West, HAC=%d):" % HAC); print(s2.round(3).to_string())
    s1.insert(0, "model", label); s2.insert(0, "model", label)
    return s1, s2


specs = [(["VPC3"], False, "VPC3"),
         (["VPC3"], True,  "VPC3 + COVID"),
         (["VPC3", "VPC4"], False, "VPC3+VPC4 (headline)")]
results = [twostep(p, c, l) for p, c, l in specs]
S1 = pd.concat([r[0] for r in results])
S2 = pd.concat([r[1] for r in results])
S1.rename_axis("param").to_csv(HERE / "insample_step1.csv")
S2.rename_axis("param").to_csv(HERE / "insample_step2.csv")
print("\nSaved: insample_step1.csv, insample_step2.csv")
