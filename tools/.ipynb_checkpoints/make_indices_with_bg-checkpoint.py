#!/usr/bin/env python
import os, re, csv, argparse
from pathlib import Path
import pandas as pd
from collections import defaultdict

def read_ids(txt_path):
    ids = []
    with open(txt_path, "r") as f:
        for line in f:
            s = line.strip()
            if s:
                ids.append(s)
    return set(ids)

def norm2(v):
    # format seconds to match your saved clip names (two decimals, zero-padded)
    return f"{float(v):0.2f}"

def extract_base_id(s):
    m = re.search(r'(P\d{2}_R\d{2})', str(s))
    return m.group(1) if m else None

def build_file_index(root, suffix=".pt"):
    # map clip_id (e.g., P11_R01_0491.52_0493.52_left) -> full path
    mapping = {}
    root = Path(root)
    if not root.exists():
        return mapping
    for p in root.rglob(f"*{suffix}"):
        clip_id = p.stem  # filename without extension
        mapping[clip_id] = str(p)
    return mapping

def main(args):
    outdir = Path(args.out_dir); outdir.mkdir(parents=True, exist_ok=True)

    # 1) official splits
    S_train = read_ids(args.train_ids)
    S_val   = read_ids(args.val_ids)

    # 2) scan features already extracted
    vjepa_idx = build_file_index(args.vjepa_dir, ".pt")
    skel_idx  = build_file_index(args.skel_dir,  ".pt")
    print(f"[scan] V-JEPA features: {len(vjepa_idx)}  |  skeletons: {len(skel_idx)}")

    # 3) metadata with No action included
    meta = pd.read_csv(args.all_csv)
    # try to normalize column names
    # expect something like: File / Filename, Action_label or Meta_action_label, Action_start_rgb_sec, Action_end_rgb_sec
    cols = {c.lower(): c for c in meta.columns}
    file_col  = cols.get("file") or cols.get("filename") or cols.get("file_name")
    label_col = cols.get("action_label") or cols.get("meta_action_label") or cols.get("action")
    t0_col    = cols.get("action_start_rgb_sec") or cols.get("start") or cols.get("action_start")
    t1_col    = cols.get("action_end_rgb_sec")   or cols.get("end")   or cols.get("action_end")

    if not (file_col and label_col and t0_col and t1_col):
        raise SystemExit(f"❌ Could not find expected columns in {args.all_csv}. Got: {meta.columns.tolist()}")

    meta["base_id"] = meta[file_col].apply(extract_base_id)
    meta = meta.dropna(subset=["base_id", label_col, t0_col, t1_col]).copy()

    # 4) build clip_ids for each view (left/right/top) and keep only those we actually have on disk
    records = []
    views = ["left","right","top"]
    for _, r in meta.iterrows():
        base = r["base_id"]
        label = str(r[label_col])
        t0 = norm2(r[t0_col])
        t1 = norm2(r[t1_col])
        split = "train" if base in S_train else ("val" if base in S_val else None)
        if split is None:
            # skip anything not in official lists
            continue
        for view in views:
            clip_id = f"{base}_{t0}_{t1}_{view}"
            p_v = vjepa_idx.get(clip_id, "")
            p_s = skel_idx.get(clip_id, "")
            # For video index: require V-JEPA present
            if p_v:
                records.append({
                    "clip": clip_id,
                    "view": view,
                    "label": label,
                    "split": split,
                    "path_vjepa": p_v,
                    "path_skel": p_s,   # may be empty; fine for video baseline
                })

    df = pd.DataFrame.from_records(records)
    if df.empty:
        raise SystemExit("No matching clips found. Check path formatting & rounding.")

    # 5) write video indices (only require path_vjepa)
    vid_train = df[(df["split"]=="train") & (df["path_vjepa"]!="")].copy()
    vid_val   = df[(df["split"]=="val")   & (df["path_vjepa"]!="")].copy()

    # 6) write skeleton indices (require path_skel)
    sk_train = df[(df["split"]=="train") & (df["path_skel"]!="")].copy()
    sk_val   = df[(df["split"]=="val")   & (df["path_skel"]!="")].copy()

    # optional filtering: min duration
    if args.min_duration > 0:
        def dur_from_clip(clip):
            # Pxx_Rxx_t0_t1_view
            m = re.search(r'_([0-9]+\.[0-9]+)_([0-9]+\.[0-9]+)_(left|right|top)$', clip)
            if not m: return 0.0
            t0, t1 = float(m.group(1)), float(m.group(2))
            return max(0.0, t1 - t0)
        for d in (vid_train, vid_val, sk_train, sk_val):
            d["duration"] = d["clip"].apply(dur_from_clip)
            d.drop(d[d["duration"] < args.min_duration].index, inplace=True)
            d.drop(columns=["duration"], inplace=True)

    # 7) class stats
    def stats(tag, d):
        by_lbl = d["label"].value_counts().sort_index()
        print(f"\n[{tag}] {len(d)} rows, {d['label'].nunique()} classes")
        print(by_lbl.head(30))

    stats("video_train", vid_train)
    stats("video_val",   vid_val)
    stats("skel_train",  sk_train)
    stats("skel_val",    sk_val)

    # 8) save
    vt = outdir / "baseline_video_train_bg.csv"
    vv = outdir / "baseline_video_val_bg.csv"
    st = outdir / "baseline_skel_train_bg.csv"
    sv = outdir / "baseline_skel_val_bg.csv"

    vid_train.to_csv(vt, index=False)
    vid_val.to_csv(vv, index=False)
    sk_train.to_csv(st, index=False)
    sk_val.to_csv(sv, index=False)
    print(f"\n✅ Wrote:\n  {vt}\n  {vv}\n  {st}\n  {sv}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--all_csv",   type=str, default="/workspace/datasets/InHARD/InHARD_All_No_Action.csv")
    ap.add_argument("--train_ids", type=str, default="/workspace/datasets/inhard/splits/train_ids.txt")
    ap.add_argument("--val_ids",   type=str, default="/workspace/datasets/inhard/splits/val_ids.txt")
    ap.add_argument("--vjepa_dir", type=str, default="/workspace/results/vjepa")
    ap.add_argument("--skel_dir",  type=str, default="/workspace/results/comotion")
    ap.add_argument("--out_dir",   type=str, default="/workspace/datasets/inhard/indices")
    ap.add_argument("--min_duration", type=float, default=0.0)
    args = ap.parse_args()
    main(args)
