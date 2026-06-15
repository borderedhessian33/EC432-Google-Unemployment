# Nowcasting Turkish Unemployment with GTAB-Calibrated Google Trends

Code for *"Ask Google Before Statistics: Nowcasting Unemployment in Turkey Using
Google Trends."* Scripts are numbered in run order; each filename names the task
it performs and (where relevant) the paper table/figure it produces. Plotting
scripts display figures on screen (`plt.show()`) — nothing is written to disk as
an image.

## Requirements
```
pip install -r requirements.txt
```
(pandas, numpy, scikit-learn, statsmodels, scipy, matplotlib, gtab.)

## Input data
| File | Description |
|------|-------------|
| `calibrated_trends_custom.csv` | GTAB-calibrated Google Trends panel (21 keywords, monthly, 2010–2025). Produced by `00_fetch_gtab_trends.py`. |
| `unemp_csv.csv` | TÜİK / EVDS monthly unemployment rate (`;`-separated, decimal `,`). |

Preprocessing is **frozen at Dec-2017**; the OOS engines re-estimate the state
space each month on an expanding window, masking `u_{t-1}` and `u_t` (2-month lag).

## Pipeline (run in order)

### Data & factors
| Script | Task | Paper artifact |
|--------|------|----------------|
| `00_fetch_gtab_trends.py` | Fetch & GTAB-calibrate Google Trends → `calibrated_trends_custom.csv` (needs internet; slow) | — |
| `01_data_plots.py` | Data-section figures (unemployment, raw vs GTAB) | Figures 1–3 |
| `02_pca_varimax_factors.py` | YoY-diff → standardise → PCA(4) → varimax → sign-anchor; factor diagnostics, VPC↔u correlations, 2-month lead | §4.1, Figures 4–7 |

### Core estimation & main results
| Script | Task | Paper artifact |
|--------|------|----------------|
| `10_benchmarks.py` | AR(1), AR(2) and the two no-Trends state spaces → `ar_benchmarks.csv` | Table 2 (Panel C) |
| `11_candidate_grid.py` | The 20 locked candidates (10 factor sets × {COVID dummy, none}), recursive OOS → `candidate_forecasts.csv` | Table 2 (RMSE) |
| `12_oos_accuracy_table.py` | Assembles Table 2: RMSE, RMSE/AR(1), Harvey-corrected DM, one-sided Clark–West → `oos_accuracy_table.csv` | **Table 2 (full)** |
| `13_decomposition.py` | AR(1) → no-Trends → headline ladder; filtering-vs-Google split (DM isolating Google) → `decomposition.csv` | Table 3 |
| `14_realtime_selection.py` | Look-ahead-free expanding annual model selection; CW vs AR(1) → `rolling_*.csv` | Table 4 |
| `15_model_confidence_set.py` | Hansen–Lunde–Nason MCS over the 24-model pool (Tₘₐₓ, moving-block bootstrap) → `mcs_results_{full,exc}.csv` | Table 5 |
| `16_subsample_rmse.py` | Pre-COVID / COVID / post-COVID RMSE + 12-month rolling RMSE figure → `subsample_rmse.csv` | Table 6, Figure 11 |
| `17_bridge_depth.py` | Bridge-depth sweep L∈{0,1,2} per factor set → `bridge_depth_rmse.csv` | **Table 8** |
| `18_insample_twostep.py` | In-sample two-step (Kalman factor + Newey–West bridge) estimates → `insample_step{1,2}.csv` | §5.7 in-sample |

### Robustness — alternative specifications (Table 7)
| Script | Alternative | Table 7 row |
|--------|-------------|-------------|
| `20_robust_ardl_direct.py` | Direct factor-augmented ARDL (OLS, no Kalman) | Direct ARDL, OLS |
| `21_robust_ardl_levels.py` | Factors from deseasonalised *levels* (no YoY diff) | Level factors |
| `22_robust_elasticnet.py` | Raw 21 keywords + elastic net (no PCA) | Raw keywords, elastic net |
| `23_robust_exog_pcs.py` | Option A: PCs as exogenous measurement regressors | PCs as exogenous |
| `24_robust_transition_pcs.py` | PCs enter the latent transition equation | PCs in transition |
| `25_robust_pls_factor.py` | PLS targeted factor inside the B5 state space | PLS factor (state space) |
| `26_robust_pls_exog.py` | PLS factor as exogenous (Option-A) state space | PLS-exog state space |
| `27_robust_stl_vs_yoy.py` | STL+YoY vs STL-only vs YoY-only (STL redundancy) | footnote 1 |

### Figures
| Script | Task | Paper artifact |
|--------|------|----------------|
| `30_fig_oos_paths.py` | Top-3 candidates vs benchmarks vs realized u | Figure 9 |
| `31_fig_realtime_path.py` | Real-time rolling-selected nowcast path | Figure 10 |

## Dependency order
```
00 → calibrated_trends_custom.csv
01, 02            read trends + unemp (independent)
10, 11            read trends + unemp → ar_benchmarks.csv, candidate_forecasts.csv
12, 13, 14, 15, 16   read 10 + 11 outputs (thin readers)
17                re-estimates Kalman (own cache: bridge_depth_forecasts.csv)
18                in-sample, reads trends + unemp
20–27             each re-runs its own alternative (read trends + unemp [+ 10/11])
30, 31            read 11 / 14 outputs
```

## Reproducibility
Every table and figure in the paper reproduces from this code — Tables 2–8, the
§4.1 factor diagnostics, and the DM / Clark–West / MCS statistics — verified by
re-running the full pipeline. Each script names the paper artifact it produces
(see the tables above).

## `exploratory/`
Scripts and outputs that informed the analysis but feed **no** paper table or
figure: PLS extraction/spec/rolling studies (`23/25/29_pls_*`), the superseded
8-candidate rolling selector (`32_nolasso_rolling`), and a descriptive
Markov-switching regime model (`markov_switching`). They read shared inputs from
the project root and write their own outputs into `exploratory/`. Kept for
reference; not part of the reproducible pipeline.

## Notes
- Preprocessing is frozen at Dec-2017; the recursive OOS engines re-estimate only
  the state-space coefficients each month (expanding window, 2-month publication lag).
- Plotting scripts display figures with `plt.show()` rather than writing image files.
