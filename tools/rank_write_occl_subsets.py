python - <<'PY'
import math, pandas as pd, numpy as np

CONF_PATH = "/workspace/datasets/inhard/indices/val_conf_agg.csv"
OUT_DIR   = "/workspace/datasets/inhard/indices"
TOP_PCT   = 0.20   # top 20% is "high occlusion"

df = pd.read_csv(CONF_PATH)

# Keep only clips that actually have frames aggregated
df = df.copy()
df["pct_low_frames"] = pd.to_numeric(df["pct_low_frames"], errors="coerce")
df["mean_conf"]      = pd.to_numeric(df["mean_conf"], errors="coerce")
df["mean_frac_low_joints"] = pd.to_numeric(df["mean_frac_low_joints"], errors="coerce")
df = df.dropna(subset=["pct_low_frames", "mean_conf"])

# Rank: higher pct_low_frames is harder; tie-break: lower mean_conf is harder
df = df.sort_values(["pct_low_frames", "mean_conf"], ascending=[False, True]).reset_index(drop=True)

# Decide cutoff
k = max(1, math.floor(TOP_PCT * len(df)))

high = df.iloc[:k].copy()
low  = df.iloc[k:].copy()

# Write minimal split CSVs (label, clip)
high[["label","clip"]].to_csv(f"{OUT_DIR}/val_high_occ.csv", index=False)
low[["label","clip"]].to_csv(f"{OUT_DIR}/val_low_occ.csv", index=False)

# Optional summary table with key metrics (sorted hardest→easiest)
summary_cols = ["label","clip","n_frames","mean_conf","mean_frac_low_joints","pct_low_frames","low_thr","frame_low_frac"]
high[summary_cols].to_csv(f"{OUT_DIR}/val_high_occ_summary.csv", index=False)

# Also keep a ranked full table for reference
df[summary_cols].to_csv(f"{OUT_DIR}/val_conf_agg_ranked.csv", index=False)

print("✔ Wrote:")
print(f"  {OUT_DIR}/val_high_occ.csv   (n={len(high)})")
print(f"  {OUT_DIR}/val_low_occ.csv    (n={len(low)})")
print(f"  {OUT_DIR}/val_high_occ_summary.csv")
print(f"  {OUT_DIR}/val_conf_agg_ranked.csv")
PY
