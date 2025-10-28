#!/usr/bin/env python
import os, re, argparse, random
from pathlib import Path
import pandas as pd

RNG = random.Random(42)

def read_ids(txt_path):
    ids = set()
    with open(txt_path, "r") as f:
        for line in f:
            s = line.strip()
            if s and not s.startswith("#"):
                ids.add(s.split()[0])
    return ids

def scan_features(root):
    root = Path(root)
    mapping = {}
    if not root.exists(): return mapping
    for p in root.rglob("*.pt"):
        mapping[p.stem] = str(p)
    return mapping

def base_id(s):
    m = re.search(r'(P\d{2}_R\d{2})', str(s))
    return m.group(1) if m else None

def f2(x):
    # zero-pad to width 7 with two decimals, e.g., 26.48 -> "0026.48", 493.52 -> "0493.52"
    try:
        return f"{float(x):07.2f}"
    except Exception:
        return None


def main(args):
    outdir = Path(args.out_dir); outdir.mkdir(parents=True, exist_ok=True)

    # 1) official splits (no leakage)
    S_train = read_ids(args.train_ids)
    S_val   = read_ids(args.val_ids)

    # 2) read InHARD.csv (comma first, then ';' as fallback)
    try:
        meta = pd.read_csv(args.inhard_csv)
    except Exception:
        meta = pd.read_csv(args.inhard_csv, sep=';')

    # 3) resolve columns (low-level = Action_label; high-level = Meta_action_label)
    cols = {c.lower(): c for c in meta.columns}
    file_col  = cols.get("file") or cols.get("filename") or cols.get("file_name")
    low_col   = cols.get("action_label") or cols.get("action_low") or cols.get("action_low_label")
    high_col  = cols.get("meta_action_label") or cols.get("action_high") or cols.get("action_high_label")
    start_col = cols.get("action_start_rgb_sec") or cols.get("start_sec") or cols.get("start")
    end_col   = cols.get("action_end_rgb_sec")   or cols.get("end_sec")   or cols.get("end")
    if not (file_col and start_col and end_col and (low_col or high_col)):
        raise SystemExit(f"Column mismatch in {args.inhard_csv}. Got: {meta.columns.tolist()}")

    label_col = low_col if args.label_level == "low" and low_col else (high_col if high_col else low_col)

    # 4) filter & normalize
    meta = meta.dropna(subset=[file_col, start_col, end_col, label_col]).copy()
    meta["base_id"] = meta[file_col].apply(base_id)
    meta = meta[meta["base_id"].isin(S_train | S_val)].copy()
    meta["t0"] = meta[start_col].apply(f2)
    meta["t1"] = meta[end_col].apply(f2)
    meta = meta.dropna(subset=["t0","t1"])

    # 5) scan features you already extracted
    vjepa_map = scan_features(args.vjepa_dir)
    skel_map  = scan_features(args.skel_dir)

    # 6) build records (left/right/top views)
    recs = []
    views = ["left","right","top"]
    for _, r in meta.iterrows():
        split = "train" if r["base_id"] in S_train else "val"
        label = str(r[label_col])

        t0_padded = r["t0"]  # already padded via f2()
        t1_padded = r["t1"]
        # also keep non-padded fallback
        t0_plain = f"{float(r[start_col]):.2f}"
        t1_plain = f"{float(r[end_col]):.2f}"

        for v in ["left", "right", "top"]:
            # primary (padded) ID
            clip_id = f"{r['base_id']}_{t0_padded}_{t1_padded}_{v}"
            p_v = vjepa_map.get(clip_id, "")
            p_s = skel_map.get(clip_id, "")

            # fallback (non-padded) if missing
            if not p_v and not p_s:
                clip_id_fallback = f"{r['base_id']}_{t0_plain}_{t1_plain}_{v}"
                p_v = vjepa_map.get(clip_id_fallback, "")
                p_s = skel_map.get(clip_id_fallback, "")
                if p_v or p_s:
                    clip_id = clip_id_fallback  # record the one that actually exists

            if p_v or p_s:
                recs.append({
                    "clip": clip_id, "view": v, "label": label, "split": split,
                    "path_vjepa": p_v, "path_skel": p_s
                })

    df = pd.DataFrame.from_records(recs)
    if df.empty:
        raise SystemExit("No matching clips found; check time rounding/paths.")

    # 7) optional min duration filter (helps prune tiny BG snippets)
    if args.min_duration > 0:
        def dur(c):
            m = re.search(r'_(\d+\.\d+)_([\d]+\.\d+)_(left|right|top)$', c)
            if not m: return 0.0
            return float(m.group(2)) - float(m.group(1))
        df["__dur__"] = df["clip"].apply(dur)
        df = df[df["__dur__"] >= args.min_duration].drop(columns="__dur__")


    # 9) write indices:
    vid_train = df[(df["split"]=="train") & (df["path_vjepa"]!="")].copy()
    vid_val   = df[(df["split"]=="val")   & (df["path_vjepa"]!="")].copy()
    sk_train  = df[(df["split"]=="train") & (df["path_skel"]!="")].copy()
    sk_val    = df[(df["split"]=="val")   & (df["path_skel"]!="")].copy()

    out_vt = Path(args.out_dir) / "baseline_video_train_bg.csv"
    out_vv = Path(args.out_dir) / "baseline_video_val_bg.csv"
    out_st = Path(args.out_dir) / "baseline_skel_train_bg.csv"
    out_sv = Path(args.out_dir) / "baseline_skel_val_bg.csv"

    vid_train.to_csv(out_vt, index=False)
    vid_val.to_csv(out_vv, index=False)
    sk_train.to_csv(out_st, index=False)
    sk_val.to_csv(out_sv, index=False)

    # 10) print quick stats
    def stat(tag, d):
        print(f"[{tag}] rows={len(d)} classes={d['label'].nunique()}")
    stat("video_train", vid_train)
    stat("video_val",   vid_val)
    stat("skel_train",  sk_train)
    stat("skel_val",    sk_val)
    print("Wrote:\n ", out_vt, "\n ", out_vv, "\n ", out_st, "\n ", out_sv)
    vjepa_map = scan_features(args.vjepa_dir)
    skel_map  = scan_features(args.skel_dir)
    print(f"[scan] found V-JEPA feature files: {len(vjepa_map)}")
    print(f"[scan] found skeleton files:       {len(skel_map)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--inhard_csv", type=str, default="/workspace/datasets/inhard/InHARD/Segmented/InHARD.csv")
    ap.add_argument("--train_ids",  type=str, default="/workspace/datasets/inhard/splits/train_ids.txt")
    ap.add_argument("--val_ids",    type=str, default="/workspace/datasets/inhard/splits/val_ids.txt")
    ap.add_argument("--vjepa_dir",  type=str, default="/workspace/results/vjepa")
    ap.add_argument("--skel_dir",   type=str, default="/workspace/results/comotion")
    ap.add_argument("--out_dir",    type=str, default="/workspace/datasets/inhard/indices")
    ap.add_argument("--label_level", choices=["low","high"], default="low",
                    help="Use low-level Action_label or high-level Meta_action_label.")
    ap.add_argument("--min_duration", type=float, default=0.0,
                    help="Drop segments shorter than this many seconds.")

    args = ap.parse_args()
    main(args)
