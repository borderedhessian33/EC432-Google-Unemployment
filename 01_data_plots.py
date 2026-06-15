"""
EC 432 Project — Data-section figures (display only; no titles, LaTeX-ready)
===========================================================================
Five clean figures shown on screen (plt.show(); nothing saved).  No plot titles
(captions go in LaTeX).  Run:  python 01_data_plots.py

  1. Unemployment rate over time (COVID shaded, 2018 crisis marked)
  2. Unemployment seasonality (trend-removed, by calendar month)
  3. Representative calibrated keyword series (small multiples)
  4. Keyword correlation heatmap (YoY-differenced, all 21)
  5. Search vs unemployment co-movement (standardised overlay)
"""

import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

warnings.filterwarnings("ignore")
HERE = Path(__file__).parent
BLUE, RED, GREY, GOLD = "#00467F", "#C0392B", "#7F8C8D", "gold"
COVID_S, COVID_E = pd.Timestamp("2020-04-01"), pd.Timestamp("2021-12-01")
CRISIS = pd.Timestamp("2018-08-01")

plt.rcParams.update({
    "font.family": "sans-serif",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.9, "axes.labelsize": 12,
    "xtick.labelsize": 10, "ytick.labelsize": 10,
    "legend.fontsize": 10, "legend.frameon": False,
    "figure.dpi": 130,
})

def yearaxis(ax, step=2):
    ax.xaxis.set_major_locator(mdates.YearLocator(step))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))


# ── Load ─────────────────────────────────────────────────────────────────────────
trends = pd.read_csv(HERE / "calibrated_trends_custom.csv", index_col=0, parse_dates=True).resample("MS").first()
unemp = pd.read_csv(HERE / "unemp_csv.csv", sep=";", decimal=",",
                    encoding="utf-8-sig", index_col=0, parse_dates=False)
unemp.columns = ["u"]
unemp = unemp[unemp.index.notna() & (unemp.index.astype(str) != "nan")]
unemp.index = pd.to_datetime(unemp.index, format="%Y-%m")
u = unemp.resample("MS").first()["u"].astype(float).dropna()


# ── FIGURE 1 — unemployment rate over time ──────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 3.6))
ax.axvspan(COVID_S, COVID_E, color=GOLD, alpha=0.20, lw=0)
ax.axvline(CRISIS, color=GREY, lw=1.0, ls="--")
ax.plot(u.index, u.values, color=BLUE, lw=1.8)
ax.fill_between(u.index, u.values, u.min() - 0.5, color=BLUE, alpha=0.06)
ax.set_ylabel("Unemployment rate (%)")
ax.set_ylim(u.min() - 0.5, u.max() + 0.7)
ax.annotate("2018 currency crisis", xy=(CRISIS, u.max() + 0.2), fontsize=9,
            color=GREY, ha="center")
ax.annotate("COVID-19", xy=(pd.Timestamp("2021-01-01"), u.max() + 0.2),
            fontsize=9, color="goldenrod", ha="center")
yearaxis(ax)
ax.set_xlim(u.index[0], u.index[-1])
ax.grid(axis="y", ls="--", alpha=0.30)
plt.tight_layout(); plt.show()


# ── FIGURE 2 — seasonality (trend-removed), by calendar month ───────────────────
detr = (u - u.rolling(12, center=True).mean()).dropna()      # remove slow trend
by_month = [detr[detr.index.month == m].values for m in range(1, 13)]
fig, ax = plt.subplots(figsize=(7, 3.6))
bp = ax.boxplot(by_month, patch_artist=True, widths=0.6,
                medianprops=dict(color=RED, lw=1.5),
                flierprops=dict(marker="o", markersize=3, alpha=0.4))
for patch in bp["boxes"]:
    patch.set(facecolor=BLUE, alpha=0.35, edgecolor=BLUE)
ax.axhline(0, color="black", lw=0.8)
ax.set_xticklabels(["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"])
ax.set_ylabel("Unemployment, trend removed (pp)")
ax.set_xlabel("Calendar month")
ax.grid(axis="y", ls="--", alpha=0.30)
plt.tight_layout(); plt.show()


# ── FIGURE 3 — representative calibrated keyword series (small multiples) ────────
pref = ["işsizlik", "iş ilanı", "işkur", "kıdem tazminatı", "kariyer.net", "memur alımı"]
sel = [k for k in pref if k in trends.columns]
if len(sel) < 6:
    extra = trends.var().sort_values(ascending=False).index
    sel += [k for k in extra if k not in sel][: 6 - len(sel)]
sel = sel[:6]
fig, axes = plt.subplots(2, 3, figsize=(11, 5.2), sharex=True)
for ax, kw in zip(axes.ravel(), sel):
    s = trends[kw].astype(float)
    ax.axvspan(COVID_S, COVID_E, color=GOLD, alpha=0.18, lw=0)
    ax.plot(s.index, s.values, color=BLUE, lw=1.3)
    ax.text(0.03, 0.92, kw, transform=ax.transAxes, fontsize=10,
            fontweight="bold", color=BLUE, va="top")
    ax.grid(axis="y", ls="--", alpha=0.25)
    yearaxis(ax, step=3)
    ax.set_xlim(s.index[0], s.index[-1])
for ax in axes[:, 0]:
    ax.set_ylabel("Search intensity")
plt.tight_layout(); plt.show()


# ── FIGURE 4 — keyword correlation heatmap (YoY-differenced, all 21) ─────────────
D = trends.diff(12).dropna()
C = D.corr().values
fig, ax = plt.subplots(figsize=(8.5, 7.5))
im = ax.imshow(C, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
ax.set_xticks(range(len(D.columns))); ax.set_xticklabels(D.columns, rotation=90, fontsize=7)
ax.set_yticks(range(len(D.columns))); ax.set_yticklabels(D.columns, fontsize=7)
fig.colorbar(im, ax=ax, shrink=0.7, label="correlation")
plt.tight_layout(); plt.show()


# ── FIGURE 5 — search vs unemployment co-movement (standardised) ────────────────
def z(s): return (s - s.mean()) / s.std()
overlay_kw = [k for k in ["işsizlik", "iş ilanı"] if k in trends.columns][:2]
fig, ax = plt.subplots(figsize=(9, 3.6))
ax.axvspan(COVID_S, COVID_E, color=GOLD, alpha=0.18, lw=0)
ax.plot(u.index, z(u), color="black", lw=2.0, label="Unemployment rate")
cols = [BLUE, RED]
for kw, c in zip(overlay_kw, cols):
    s = trends[kw].astype(float).reindex(u.index)
    ax.plot(s.index, z(s), color=c, lw=1.4, alpha=0.85, label=f"Search: {kw}")
ax.set_ylabel("Standardised (z-score)")
yearaxis(ax)
ax.set_xlim(u.index[0], u.index[-1])
ax.legend(loc="upper left", ncol=3)
ax.grid(axis="y", ls="--", alpha=0.30)
plt.tight_layout(); plt.show()

print("Displayed 5 data figures (run this file directly to view).")
print(f"Keywords shown in Fig 3: {sel}")
print(f"Co-movement overlay (Fig 5): {overlay_kw}")
