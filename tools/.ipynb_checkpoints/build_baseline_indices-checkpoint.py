#!/usr/bin/env python3
import os, re, csv
from pathlib import Path
from decimal import Decimal
import pandas as pd

# ---- CONFIG ----
DS = Path("/workspace/datasets/inhard/indices")     # train_index.csv, val_index.csv
VJ = Path("/workspace/results/vjepa")
SK = Path("/workspace/results/comotion")
VIEWS = ("left","right","top")
TOL = Decimal("0.04")  # tolerance in seconds (adjust 0.02–0.04 depending on your crop fps)
# ----------------

# File name pattern: P07_R02_0138.04_0139.84_top.pt
RE = re.compile(r"^(P\d{2}_R\d{2})_(\d+\.\d+?)_(\d+\.\d+?)_(left|right|top)\.pt$")

def dec(x) -> Decimal:
    # robust decimal from csv float or str
    return Decimal(str(x))

def scan_stream(root: Path):
    """
    Build an index: {(file_id, view): [(s, e, basename, fullpath), ...]}
    where s/e are Decimal seconds parsed from filenames.
    """
    idx = {}
    count = 0
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            if not fn.endswith(".pt"):
                continue
            m = RE.match(fn)
            if not m:
                continue
            file_id, s, e, view = m.groups()
            sD, eD = Decimal(s), Decimal(e)
            full = os.path.join(dirpath, fn)
            idx.setdefault((file_id, view), []).append((sD, eD, fn, full))
            count += 1
    print(f"[scan] {root} -> {count} matched .pt files")
    # optional: sort by start time for easier nearest lookup
    for k in idx:
        idx[k].sort(key=lambda t: (t[0], t[1]))
    return idx

def nearest_basename(idx, file_id: str, view: str, s_csv: Decimal, e_csv: Decimal):
    """
    Return best-matching (basename, fullpath) if within tolerance; else ("","")
    """
    cand = idx.get((file_id, view), [])
    best = None
    best_err = None
    for s,e,fn,fp in cand:
        err = abs(s - s_csv) + abs(e - e_csv)
        if best_err is None or err < best_err:
            best, best_err = (fn, fp), err
    if best and best_err <= TOL:
        return best
    return ("","")

def rows_from_index(df: pd.DataFrame, split: str):
    label_col = "Meta_action_label" if "Meta_action_label" in df.columns else "Action_label"
    for _, r in df.iterrows():
        file_id = str(r["File"])                         # e.g., P01_R01
        s_csv, e_csv = dec(r["Action_start_rgb_sec"]), dec(r["Action_end_rgb_sec"])
        label = str(r[label_col])
        yield file_id, s_csv, e_csv, label, split

def write_csv(rows, out_csv, use, vj_idx, sk_idx):
    fields = ["clip","view","label","split","path_vjepa","path_skel"]
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    ok = miss = 0
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
        for file_id, s_csv, e_csv, label, split in rows:
            base_key = f"{file_id}_{s_csv:.2f}_{e_csv:.2f}"  # for readability only
            for view in VIEWS:
                bn_vj, path_vj = nearest_basename(vj_idx, file_id, view, s_csv, e_csv)
                bn_sk, path_sk = nearest_basename(sk_idx, file_id, view, s_csv, e_csv)
                need = path_vj if use=="video" else path_sk
                if not need:
                    miss += 1
                    continue
                # clip value without view (use matched basename minus _view.pt)
                clip = (bn_vj or bn_sk).rsplit(".",1)[0].rsplit("_",1)[0]
                w.writerow({
                    "clip": clip,
                    "view": view,
                    "label": label,
                    "split": split,
                    "path_vjepa": path_vj,
                    "path_skel":  path_sk,
                })
                ok += 1
    print(f"{out_csv} -> {ok} rows (miss={miss})")

def main():
    # scan once
    vj_idx = scan_stream(VJ)
    sk_idx = scan_stream(SK)

    tr = pd.read_csv(DS/"train_index.csv")
    va = pd.read_csv(DS/"val_index.csv")

    tr_rows = list(rows_from_index(tr, "train"))
    va_rows = list(rows_from_index(va, "val"))

    write_csv(tr_rows, DS/"baseline_video_train.csv", "video",    vj_idx, sk_idx)
    write_csv(va_rows, DS/"baseline_video_val.csv",   "video",    vj_idx, sk_idx)
    write_csv(tr_rows, DS/"baseline_skel_train.csv",  "skeleton", vj_idx, sk_idx)
    write_csv(va_rows, DS/"baseline_skel_val.csv",    "skeleton", vj_idx, sk_idx)

if __name__ == "__main__":
    main()
