#!/usr/bin/env python3
import argparse, math
from pathlib import Path
import pandas as pd

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", type=Path, required=True,
                    help="Path to baseline_skel_val_bg.csv (has clip,view,label,split,path_vjepa,path_skel)")
    ap.add_argument("--conf-agg", type=Path, required=True,
                    help="Path to val_conf_agg.csv produced by the aggregator (label, clip, pct_low_frames, mean_conf, etc.)")
    ap.add_argument("--out-dir", type=Path, required=True,
                    help="Directory to write val_high_occ.csv, val_low_occ.csv, val_high_occ_summary.csv, val_conf_agg_ranked.csv")
    ap.add_argument("--top-pct", type=float, default=0.20, help="Top fraction for high occlusion split (default 0.20)")
    ap.add_argument("--primary", type=str, default="pct_low_frames", help="Primary ranking metric (default pct_low_frames)")
    ap.add_argument("--tie", type=str, default="mean_conf", help="Tie-breaker metric (ascending; default mean_conf)")
    args = ap.parse_args()

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    base = pd.read_csv(args.baseline)
    conf = pd.read_csv(args.conf_agg)

    # expected baseline columns
    base_cols = ["clip","view","label","split","path_vjepa","path_skel"]
    miss = [c for c in base_cols if c not in base.columns]
    if miss:
        raise SystemExit(f"Baseline is missing required columns: {miss}")

    # numeric safety
    conf["pct_low_frames"] = pd.to_numeric(conf.get("pct_low_frames"), errors="coerce")
    conf["mean_conf"]      = pd.to_numeric(conf.get("mean_conf"), errors="coerce")

    # rank hardest → easiest
    conf_ranked = conf.dropna(subset=[args.primary, args.tie]).sort_values(
        [args.primary, args.tie], ascending=[False, True]
    ).reset_index(drop=True)

    k = max(1, math.floor(args.top_pct * len(conf_ranked)))
    high_keys = conf_ranked.iloc[:k][["label","clip"]].copy()
    low_keys  = conf_ranked.iloc[k:][["label","clip"]].copy()

    # use only 'val' split rows from baseline (if present)
    base_val = base.copy()
    if "split" in base_val.columns:
        base_val = base_val[base_val["split"].astype(str).str.lower().eq("val")]

    # merge to bring view/split/paths onto the keys
    high = high_keys.merge(base_val[base_cols], on=["label","clip"], how="left")
    low  = low_keys.merge(base_val[base_cols],  on=["label","clip"], how="left")

    # write splits matching baseline schema
    (out_dir / "val_high_occ.csv").write_text(high[base_cols].to_csv(index=False))
    (out_dir / "val_low_occ.csv").write_text(low[base_cols].to_csv(index=False))

    # summary (add metrics for the high set)
    summary_cols = base_cols + ["n_frames","mean_conf","mean_frac_low_joints","pct_low_frames","low_thr","frame_low_frac"]
    conf_small = conf_ranked[["label","clip","n_frames","mean_conf","mean_frac_low_joints","pct_low_frames","low_thr","frame_low_frac"]]
    high_summary = high.merge(conf_small, on=["label","clip"], how="left")
    (out_dir / "val_high_occ_summary.csv").write_text(high_summary[summary_cols].to_csv(index=False))

    # full ranked table with paths (nice for inspection)
    full_ranked = conf_ranked.merge(base_val[base_cols], on=["label","clip"], how="left")
    (out_dir / "val_conf_agg_ranked.csv").write_text(full_ranked[summary_cols].to_csv(index=False))

    print(f"✔ wrote: {out_dir/'val_high_occ.csv'} (rows={len(high)})")
    print(f"✔ wrote: {out_dir/'val_low_occ.csv'} (rows={len(low)})")
    print(f"✔ wrote: {out_dir/'val_high_occ_summary.csv'}")
    print(f"✔ wrote: {out_dir/'val_conf_agg_ranked.csv'}")

if __name__ == "__main__":
    main()
