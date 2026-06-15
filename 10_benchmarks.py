"""
EC 432 Project — Benchmarks (LASSO-free, the comparison set for all candidates)
================================================================================
Google-free benchmark nowcasts against which every candidate is judged, all on
the same recursive design (expanding window, 2-month publication lag, eval
2018-01 … 2025-12):

  AR1, AR2        naive autoregressions on unemployment (no state space)
  NoTrends        state-space filtering structure with NO Google factor
  NoTrends_cov    same + constant COVID dummy

The no-Trends models share the candidates' state-space machinery (latent AR(1)
factor + measurement error, filtered-state nowcast) but omit the search factor,
so the AR(1) → no-Trends → candidate ladder isolates the filtering structure from
the Google signal.  Nothing here uses Google Trends, LASSO, or PCA.

Output: ar_benchmarks.csv  (actual, AR1, AR2, NoTrends, NoTrends_cov)
"""

import warnings
from pathlib import Path
import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.statespace.mlemodel import MLEModel

warnings.filterwarnings("ignore")
HERE = Path(__file__).parent
EVAL_START  = pd.Timestamp("2018-01-01")
EVAL_END    = pd.Timestamp("2025-12-01")
COVID_START = pd.Timestamp("2020-04-01")
COVID_END   = pd.Timestamp("2021-12-01")
PUB_LAG     = 2


# ── Unemployment series ──────────────────────────────────────────────────────────
unemp = pd.read_csv(HERE / "unemp_csv.csv", sep=";", decimal=",",
                    encoding="utf-8-sig", index_col=0, parse_dates=False)
unemp.columns = ["u"]
unemp = unemp[unemp.index.notna() & (unemp.index.astype(str) != "nan")]
unemp.index = pd.to_datetime(unemp.index, format="%Y-%m")
u_full = unemp.resample("MS").first()["u"].astype(float).dropna()
dates = u_full[(u_full.index >= EVAL_START) & (u_full.index <= EVAL_END)].index


# ── AR(1)/AR(2): 2-step-ahead forecast from data through t-2 ─────────────────────
def recursive_ar(order):
    out = pd.Series(index=dates, dtype=float)
    for t in dates:
        cutoff = t - pd.DateOffset(months=PUB_LAG - 1)
        u_tr = u_full[u_full.index < cutoff]
        if len(u_tr) >= 6:
            try:
                out[t] = float(SARIMAX(u_tr, order=(order, 0, 0), trend="c")
                               .fit(disp=False, method="lbfgs").forecast(PUB_LAG).iloc[-1])
            except Exception:
                pass
    return out


# ── No-Trends state space: latent AR(1) + measurement error (+ optional COVID) ──
class NoTrendsSS(MLEModel):
    def __init__(self, endog, covid_col=None):
        self.use_covid = covid_col is not None
        super().__init__(endog, k_states=1, k_posdef=1)
        self._covid = (np.asarray(covid_col, float).flatten()
                       if covid_col is not None else np.zeros(self.nobs))
        self["selection", 0, 0] = 1.0
        self["design", 0, 0] = 1.0
        self["obs_intercept"] = np.zeros((1, self.nobs))
        self.initialize_approximate_diffuse()
    @property
    def param_names(self):
        return ["phi", "sig_u", "sig_eta"] + (["delta_covid"] if self.use_covid else [])
    @property
    def start_params(self):
        return np.array([0.7, 0.3, 0.3] + ([0.0] if self.use_covid else []))
    def transform_params(self, u):
        p = u.copy(); p[0] = np.tanh(u[0]); p[1] = np.exp(u[1]); p[2] = np.exp(u[2])
        return p
    def untransform_params(self, c):
        p = c.copy(); p[0] = np.arctanh(np.clip(c[0], -0.9999, 0.9999))
        p[1] = np.log(max(c[1], 1e-12)); p[2] = np.log(max(c[2], 1e-12))
        return p
    def update(self, params, **kwargs):
        params = super().update(params, **kwargs)
        self["transition", 0, 0] = params[0]
        self["obs_cov", 0, 0] = params[1] ** 2
        self["state_cov", 0, 0] = params[2] ** 2
        oi = np.zeros((1, self.nobs))
        if self.use_covid:
            oi[0, :] = params[3] * self._covid
        self["obs_intercept"] = oi


def recursive_notrends(use_covid):
    out, prev = pd.Series(index=dates, dtype=float), None
    for t in dates:
        idx = u_full.index[u_full.index <= t]
        lag_cutoff = t - pd.DateOffset(months=PUB_LAG - 1)        # mask u_{t-1}, u_t
        if (idx < lag_cutoff).sum() < 24:
            continue
        u_arr = u_full.reindex(idx).values.astype(float)
        train = idx < lag_cutoff
        mu_e, sd_e = np.nanmean(u_arr[train]), np.nanstd(u_arr[train])
        sd_e = sd_e if sd_e > 1e-8 else 1.0
        endog = (u_arr - mu_e) / sd_e
        endog[idx >= lag_cutoff] = np.nan
        covid = (((idx >= COVID_START) & (idx <= COVID_END)).astype(float) / sd_e
                 if use_covid else None)
        try:
            mdl = NoTrendsSS(endog, covid)
            if prev is not None and len(prev) == len(mdl.start_params):
                res = mdl.fit(start_params=prev, method="lbfgs", maxiter=100, disp=False)
            else:
                r0 = mdl.fit(method="nm", maxiter=3000, disp=False)
                res = mdl.fit(start_params=r0.params, method="lbfgs", maxiter=300, disp=False)
            prev = res.params
            out[t] = res.filter_results.filtered_state[0, -1] * sd_e + mu_e
        except Exception:
            prev = None
    return out


# ── Build all benchmarks ─────────────────────────────────────────────────────────
print("AR(1)/AR(2) …")
bench = pd.DataFrame({"actual": u_full.reindex(dates)})
bench["AR1"] = recursive_ar(1)
bench["AR2"] = recursive_ar(2)
print("no-Trends SS (no COVID) …")
bench["NoTrends"] = recursive_notrends(False)
print("no-Trends SS (+ COVID dummy) …")
bench["NoTrends_cov"] = recursive_notrends(True)
bench.to_csv(HERE / "ar_benchmarks.csv")

actual = bench["actual"]
def rmse(c):
    e = (actual - bench[c]).dropna()
    return float(np.sqrt((e ** 2).mean()))
for c in ["AR1", "AR2", "NoTrends", "NoTrends_cov"]:
    print(f"{c:<14} RMSE(full) = {rmse(c):.4f}")
print("Saved: ar_benchmarks.csv")
