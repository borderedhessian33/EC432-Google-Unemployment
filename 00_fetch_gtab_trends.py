import os
import time
import random
import inspect
import gtab
import pandas as pd

# ── Configuration ─────────────────────────────────────────────────────────────
GEO          = "TR"
TIMEFRAME    = "2010-01-01 2024-12-31"
SLEEP_MIN    = 30
SLEEP_JITTER = 10

anchor_candidates = [
    "hava durumu",
    "google",
    "youtube",
    "haberler",
    "spor",
    "sinema",
    "netflix",
    "trendyol",
    "fenerbahçe",
    "galatasaray",
    "gs",
    "dolar",
    "borsa",
    "konut",
    "istanbul",
    "migros",
    "beşiktaş",
    "araba",
    "futbol",
    "yemek tarifleri",
    "asgari ücret",
    "enflasyon",
    "tatil",
]

keywords = [
    "cv",
    "cv örnekleri",
    "eleman aranıyor",
    "iş",
    "iş arama",
    "iş ilanı",
    "iş ilanları",
    "iş arayanlar",
    "iş arıyorum",
    "iş bulma",
    "işci bulma kurumu",
    "işkur",
    "işsizlik",
    "işsizlik sigortası",
    "kariyer",
    "kariyer.net",
    "kariyer net",
    "kariyernet",
    "personal alımı",
    "kariyer merkezi",
    "kariyer iş ilanları",
    "iş başvurusu",
    "iş başvuru",
    "işe alım",
    "çok para kazanan meslekler",
    "en çok para kazanan meslekler",
    "en iyi meslekler",
    "online iş",
    "online iş ilanları",
    "online eleman",
    "uzaktan iş",
    "uzaktan iş ilanları",
    "part time iş",
    "kamu iş ilanı",
    "devlet iş ilanı",
    "kamu iş ilanları",
    "devlet iş ilanları",
    "memur alımı",
    "devlette iş",
]

# ── Write anchor candidates file into G-TAB's own directory ───────────────────
gtab_dir    = os.path.dirname(inspect.getfile(gtab))
anchor_file = "anchor_candidate_list.txt"
anchor_path = os.path.join(gtab_dir, anchor_file)

with open(anchor_path, "w", encoding="utf-8") as f:
    for term in anchor_candidates:
        f.write(term + "\n")

print(f"Anchor candidates written to: {anchor_path}")
print(f"Total anchor candidates: {len(anchor_candidates)}")
print(f"Total keywords to calibrate: {len(keywords)}")
print()

# ── Verify cache is clean ──────────────────────────────────────────────────────
print("Proceeding with run...")
print()

# ── Init ───────────────────────────────────────────────────────────────────────
t = gtab.GTAB()

NUM_CANDIDATES = len(anchor_candidates)   # 23
NUM_ANCHORS    = NUM_CANDIDATES // 2      # 11 — always less than num_candidates

t.set_options(
    pytrends_config={
        "geo": GEO,
        "timeframe": TIMEFRAME,
    },
    gtab_config={
        "anchor_candidates_file": anchor_file,
        "num_anchor_candidates": NUM_CANDIDATES,
        "num_anchors": NUM_ANCHORS,
        "sleep": 30,
    },
    conn_config={
        "retries": 3,
        "backoff_factor": 2.0,
        "timeout": [30, 30],
    }
)

# ── Verify ────────────────────────────────────────────────────────────────────
print("PYTRENDS config:", t.CONFIG["PYTRENDS"])
print("GTAB config:    ", t.CONFIG["GTAB"])
print()

# ── Phase 1: Build the anchor bank ────────────────────────────────────────────
print("Building anchor bank — do not interrupt...")
print("This will take a long time due to sleep intervals.")
t.create_anchorbank()
print("Anchor bank built successfully.")
print()
print("Sleeping 3 minutes before keyword queries...")
time.sleep(180)

# ── Phase 2: Calibrate keywords one by one ────────────────────────────────────
calibrated = {}
failed     = []

for i, kw in enumerate(keywords):
    try:
        print(f"[{i+1}/{len(keywords)}] Querying: {kw}")
        calibrated[kw] = t.new_query(kw)

        sleep_time = SLEEP_MIN + random.uniform(0, SLEEP_JITTER)
        print(f"  Sleeping {sleep_time:.1f}s...")
        time.sleep(sleep_time)

    except Exception as e:
        print(f"  FAILED: {kw} — {e}")
        failed.append(kw)
        print("  Cooling down for 5 minutes...")
        time.sleep(300)

# ── Save results ───────────────────────────────────────────────────────────────
failed_calibration = []
clean_calibrated = {}

for kw, result in calibrated.items():
    if isinstance(result, pd.Series):
        clean_calibrated[kw] = result
    else:
        failed_calibration.append((kw, result))

if failed_calibration:
    print("\nKeywords that returned non-series results (likely uncalibratable):")
    for kw, val in failed_calibration:
        print(f"  {kw}: {val}")

results_df = pd.DataFrame(clean_calibrated)
results_df.to_csv("calibrated_trends_custom.csv")
print()
print(f"Results saved to calibrated_trends_custom.csv")
print(f"Successfully calibrated: {len(clean_calibrated)}/{len(calibrated)} keywords")

if failed:
    print(f"\nFailed during querying (retry manually): {failed}")
    with open("failed_keywords.txt", "w", encoding="utf-8") as f:
        for kw in failed:
            f.write(kw + "\n")