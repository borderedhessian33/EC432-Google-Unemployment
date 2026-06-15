# Paper State & Direction — Nowcasting Turkish Unemployment with Google Trends

_Working summary of decisions, conclusions, and the writing plan. Last updated from the analysis session._

---

## 1. One-sentence thesis
GTAB-calibrated Google Trends carry **statistically significant, ~2-month-leading** predictive content for Turkish unemployment (Clark–West), and labour-market search factors **consistently rank best out-of-sample**; but the gains are **modest, regime-dependent, and not decisive under a multiplicity-controlled confidence set** — they concentrate in turbulent periods.

Lead with **rigour and honesty**, not "Google beats AR." The careful real-time design + candid assessment is the contribution.

---

## 2. What we concluded (the defensible findings)

1. **Headline model = state space with a shared latent factor (B5).** Latent AR(1) factor μ; unemployment and the Google factor both load on μ (the factor-measurement equation is the channel that does the work); lagged Google enters the u-equation as a bridge; COVID dummy. **OOS RMSE ≈ 0.534 (full) / 0.444 (ex-COVID)** vs AR(1) 0.586 / 0.469.

2. **The gain is essentially all-Google.** The same state space with **no Trends** (latent AR + measurement error) ≈ AR(1) (0.595). Adding the Google factor is what moves RMSE, and it is **significant (DM t = 2.15, p = 0.03)**. The filtering structure alone buys nothing — this is the cleanest single piece of evidence.

3. **Honest real-time evaluation = rolling/expanding selection.** Re-select the model each year on all prior data, forecast that year, evaluate 2019–2025. Beats AR(1) (**CW p ≈ 0.04**), ties the ex-post best model, and **converges to the labour-factor model from 2021** — so the "best" model is not a hindsight cheat.

4. **Model Confidence Set (the sobering, honest result).** At n ≈ 96, the **90% MCS contains the entire pool, including AR(1)**. So: gains are CW-significant (nested test) but **not decisive under multiplicity control**. Report this — it is what makes the paper credible. Reconcile: CW (powerful, nested, one-sided) detects content; MCS (joint, conservative) cannot separate at this sample size.

5. **Non-stationarity is the central explanatory finding.** The search→unemployment correlation falls from **0.74 (2011–17) to 0.18 (full sample)** across COVID. This explains *why* gains are modest and regime-dependent, why frozen factors go stale, and why supervised PLS overfits. The labour factor **leads unemployment by 2 months** (cross-correlation peak at L = 2) — the empirical backbone of the nowcast and of the K = 2 bridge.

6. **Pipeline can be simplified — two ad hoc steps are removable:**
   - **STL is mathematically redundant** after a 12-month (YoY) difference (the fixed monthly seasonal cancels exactly; verified, max diff 1e-13).
   - **LASSO is dispensable**: PCA + varimax on all 21 keywords + selecting the u-relevant rotated component (VPC3) recovers B5 exactly (0.5355 ≈ 0.5343, DM p = 0.93).
   - Caveat: some factor-sharpening still helps the *rolling selector* adopt Trends earlier on short windows.

7. **PCA beats PLS out of sample.** PLS fits u better in-sample (corr 0.74 vs 0.59) but **overfits** and loses OOS (B5-PLS 0.584 > B5-PCA 0.534, DM p = 0.03), because the relationship is non-stationary. **PLS-exog is the best ex-COVID/calm-regime model (0.4365)** but does not beat PCA on the full window. Use as a robustness/regime point.

8. **Unemployment is already seasonally adjusted** (month-dummy R² = 0.003; seasonal amplitude 0.16 pp vs sd 1.66). The code correctly does **not** deseasonalize u. Seasonality that matters is in the **keywords**, not u — fix the slide claim "u is strongly seasonal."

9. **COVID is a transitory spike-and-recovery, not a permanent break in the u process.** Chow at 2020-04: F = 1.92, **p = 0.15 (not significant)**; COVID pulse and post-COVID step dummies insignificant; Quandt–Andrews sup-F = 10.56 at 2021-06 (< 11.8 c.v.). So the *unemployment process* is stable; what breaks is the *search→u relationship* (point 5). Keep these two distinct.

10. **All single-equation / lighter alternatives corroborate the state-space choice** (they give the gain back): direct ARDL (0.638), regularized keyword ARDL (0.685), exogenous-measurement Option A (0.584), factors-in-transition (0.592), robust Student-t errors (no change), second-stage bridge correction (worse). The shared-factor structure is what works.

---

## 3. Models decided on

| Role | Model | Where |
|---|---|---|
| **Headline (ex-post best individual)** | B5 = VPC1, AR(1), ld=2, K=2, COVID dummy | `06_oos_bridge_models.py` |
| **Headline (real-time, look-ahead-free)** | Rolling/expanding selection over the no-LASSO grid | `33_nolasso_grid_rolling.py` |
| **Equivalent no-LASSO factor** | VPC3 (labour factor), AR(1), K=2, COVID | `33_…` / `30_b5_nolasso.py` |
| **Time-varying COVID variant** | TV3 / VPC3 δ_t∼RW | `08_oos_timevarying_covid.py` |
| **Calm-regime / ex-COVID best** | PLS-exog (SARIMAX, PLS1, K=2, +COVID) | `26_pls_statespace.py` |
| **Benchmarks** | AR(1), AR(2) | inside every OOS script |

**Pre-registered candidate grid** (≈15 models): factor ∈ {VPC1..VPC4 singles + VPC1+VPC3, VPC3+VPC4} × COVID ∈ {dummy, none, RW} × structure ∈ {B5 factor-measurement, exogenous}, with AR benchmarks. Stop here — more candidates invite selection overfitting (we saw VPC2-COVID and dual-PLS blow up).

---

## 4. Key numbers (eval 2018–2025, n = 96)

| Model | RMSE full | RMSE ex-COVID |
|---|---|---|
| AR(1) | 0.586 | 0.469 |
| AR(2) | 0.606 | 0.447 |
| **B5 (state space, headline)** | **0.534** | **0.444** |
| Rolling-selected (real-time, no-LASSO) | 0.552 | 0.436 |
| no-Trends state space (decomposition) | 0.595 | — |
| PLS-exog (calm-regime best) | 0.578 | **0.437** |

Significance: rolling vs AR(1) **CW p ≈ 0.04**; Google contribution (no-Trends→B5) **DM p = 0.03**; DM pairwise mostly insignificant; **90% MCS = whole pool incl. AR(1)**.

---

## 5. Paper structure & what to write per section

- **Introduction** — ragged-edge / publication-lag problem (TÜİK t+3); contribution = (i) first GTAB application to Turkish unemployment, (ii) strictly real-time look-ahead-free evaluation with rolling selection, (iii) honest decomposition + MCS. Preview headline number **and** the candid caveat.
- **Literature** — Choi–Varian; D'Amuri–Marcucci; Chadwick–Şengül (Turkey); Borup–Schütte. Position as the rigorously-evaluated emerging-market entry.
- **Data** — unemployment (SA, 2-month lag); 21 keywords + GTAB calibration; figures from `35_data_plots.py` + raw-vs-calibrated GTAB. Note COVID upfront; **drop "u is strongly seasonal."**
- **Methodology** — (4.1) preprocessing: YoY-diff → standardize → PCA + varimax → select factor (state STL & LASSO are removable in robustness); (4.2) the state-space/B5 model, emphasize the factor-measurement channel; (4.3) real-time design: frozen-at-2017, 2-month masking, recursive expanding window; (4.4) **pre-registered grid + rolling selection**; (4.5) inference: DM, Clark–West, MCS (note they answer different questions).
- **Results** — (5.1) factor diagnostics + 2-month lead; (5.2) headline OOS + no-Trends decomposition; (5.3) MCS, stated straight.
- **Robustness** (your biggest asset) — pipeline ablations (no-LASSO, no-STL, levels vs diff); alternative models (13–21); subsample tables (pre/COVID/post); incremental-over-macro **(to do)**.
- **Discussion** — non-stationarity as the organizing finding; reconcile CW vs MCS; "Google helps in turbulence."
- **Conclusion / limitations / future work** — short OOS, non-stationarity, real-time vintages, single country; future: regional/weekly MIDAS, DMA, larger GTAB panel.

**Writing order:** lock Methodology + Results tables first; pre-register the grid before reporting; one headline number stated consistently; turn every null (MCS, PLS, modest gain) into a rigour point.

---

## 6. Important .py files (renamed 2026-06: filename = task + paper artifact)

See `README.md` for the full table. The ones that matter:

**Core pipeline / headline**
- `00_fetch_gtab_trends.py` — builds `calibrated_trends_custom.csv` (GTAB)
- `02_pca_varimax_factors.py` — frozen PCA+varimax factors + §4.1 diagnostics
- `10_benchmarks.py` — AR(1)/AR(2)/no-Trends benchmarks (Table 2 Panel C)
- `11_candidate_grid.py` — **20 locked candidates (Table 2 RMSE)** → `candidate_forecasts.csv`
- `12_oos_accuracy_table.py` — **full Table 2** (RMSE + DM + CW)
- `13_decomposition.py` — **no-Trends decomposition (gain is all-Google)** (Table 3)
- `14_realtime_selection.py` — **rolling/expanding real-time selection** (Table 4)
- `15_model_confidence_set.py` — **Model Confidence Set** (Table 5)
- `16_subsample_rmse.py` — sub-period RMSE + rolling-RMSE figure (Table 6, Fig 11)
- `17_bridge_depth.py` — **bridge-depth sweep** (Table 8)
- `18_insample_twostep.py` — in-sample Step-1/Step-2 estimates (§5.7)

**Robustness / alternatives (Table 7, cite don't headline)**
- `20_robust_ardl_direct.py`, `21_robust_ardl_levels.py`, `22_robust_elasticnet.py`
- `23_robust_exog_pcs.py` (Option A), `24_robust_transition_pcs.py` (transition)
- `25_robust_pls_factor.py`, `26_robust_pls_exog.py`, `27_robust_stl_vs_yoy.py` (PLS + STL-redundancy)

**Figures**
- `01_data_plots.py` — data-section figures (Figs 1–3)
- `30_fig_oos_paths.py` (Fig 9), `31_fig_realtime_path.py` (Fig 10)
- factor diagnostics figures come from `02_pca_varimax_factors.py` (Figs 4–7)

**`exploratory/`** — informed the analysis but feed no paper table/figure:
`23_pls_diagnostics`, `25_pls_specs`, `29_pls_rolling`, `32_nolasso_rolling`
(superseded by `14`), `markov_switching`.

_(`hw3_var.py`, `var_analysis.R` in the other folder are an unrelated VAR homework — ignore.)_

---

## 7. Open items / before submission

**Code↔paper cross-check (done 2026-06):** every empirical result reproduces from
the code, most to the exact decimal — Tables 2 (incl. DM/CW), 3, 4, 5 (MCS), 6, 7,
8, plus §4.1 (~58% variance, lag-2 peak) and the channel decomposition. Two
**manuscript-text** numbers are NOT supported by the code and must be fixed:
- **§4.1 VPC2 corr with u = 0.40 → should be 0.30** (VPC1 0.48, VPC3 0.61, VPC4 0.07 all match).
- **§5.5 "0.74 → 0.18" corr decay** is the *supervised PLS1* factor's number; the
  headline labour factor **VPC3 decays 0.61 → 0.34**. Re-label or restate.

Earlier (pre-rename) cross-check finds still relevant:
- **Incremental-over-macro test** (highest value): does Google add CW-significant content **on top of** IP, USD/TRY, CBRT confidence? Affects the referee verdict. If not run, flag as a limitation.
- "n=96" training-window label (in-sample tables are full-sample n=180); the factor equation shows one loading but the code uses ld=2 (three loadings); "u strongly seasonal" is false.
- Optional rigor upgrades: Dynamic Model Averaging (robust "don't select"); formal break test on the bridge coefficient to document the non-stationarity.
