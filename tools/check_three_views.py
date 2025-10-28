#!/usr/bin/env python3
import os, re, csv, sys
from pathlib import Path
from decimal import Decimal

# ---- CONFIG: update if your paths differ ----
RGB_SEG = Path("/workspace/datasets/inhard/InHARD/Segmented/RGBSegmented")
CROPPED = Path("/workspace/cropped")
VIEWS = ("left","right","top")
TOL = Decimal("0.06")   # tolerance seconds for tiny rounding differences (try 0.02–0.08)
# --------------------------------------------

# P01_R01_0124.48_0128.64.mp4
RE_ORIG = re.compile(r"^(P\d{2}_R\d{2})_(\d+\.\d+?)_(\d+\.\d+?)\.mp4$")
# P01_R01_0156.88_0159.16_right.mp4
RE_VIEW = re.compile(r"^(P\d{2}_R\d{2})_(\d+\.\d+?)_(\d+\.\d+?)_(left|right|top)\.mp4$")

def dec(x): return Decimal(str(x))

def scan_rgb_segmented(root: Path):
    """Return list of originals: dicts with file_id, s, e, label, path."""
    out = []
    for dirpath, _, files in os.walk(root):
        label = Path(dirpath).name
        for fn in files:
            if not fn.endswith(".mp4"): continue
            m = RE_ORIG.match(fn)
            if not m: continue
            fid, s, e = m.groups()
            out.append({
                "fid": fid,
                "s": dec(s), "e": dec(e),
                "label": label,
                "path": os.path.join(dirpath, fn),
            })
    return out

def scan_cropped(root: Path):
    """
    Return:
      idx[(fid, view)] -> list[(s,e, path)]
      all_entries -> list of (fid, s, e, view, path)
    """
    idx = {}
    all_entries = []
    count = 0
    for dirpath, _, files in os.walk(root):
        for fn in files:
            if not fn.endswith(".mp4"): continue
            m = RE_VIEW.match(fn)
            if not m: continue
            fid, s, e, view = m.groups()
            sD, eD = dec(s), dec(e)
            full = os.path.join(dirpath, fn)
            idx.setdefault((fid, view), []).append((sD, eD, full))
            all_entries.append((fid, sD, eD, view, full))
            count += 1
    # sort for stable "nearest" selection
    for k in idx:
        idx[k].sort(key=lambda t: (t[0], t[1]))
    return idx, all_entries

def nearest(idx, fid, view, s_csv, e_csv, used):
    """
    Choose the unmatched cropped file with smallest |s-s_csv|+|e-e_csv|
    if <= TOL. 'used' is a set of paths already matched to avoid double-use.
    """
    cand = idx.get((fid, view), [])
    best = None; best_err = None
    for s,e,fp in cand:
        if fp in used: 
            continue
        err = abs(s - s_csv) + abs(e - e_csv)
        if best_err is None or err < best_err:
            best, best_err = fp, err
    if best and best_err <= TOL:
        return best
    return ""

def main():
    print(f"Scanning original RGBSegmented under: {RGB_SEG}")
    orig = scan_rgb_segmented(RGB_SEG)
    print(f"  found {len(orig)} original segments")

    print(f"Scanning cropped outputs under: {CROPPED}")
    idx, all_cropped = scan_cropped(CROPPED)
    print(f"  found {len(all_cropped)} cropped view files")

    used = set()
    all3 = two = one = zero = 0
    missing_counts = {v: 0 for v in VIEWS}

    # CSV outputs
    out_dir = RGB_SEG.parent / "check_reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    coverage_csv = out_dir / "three_view_coverage.csv"
    missing_csv = out_dir / "missing_views.csv"
    extras_csv  = out_dir / "unmatched_cropped_views.csv"

    # check coverage per original
    with open(coverage_csv, "w", newline="") as f_cov, open(missing_csv, "w", newline="") as f_miss:
        cov_w = csv.DictWriter(f_cov, fieldnames=[
            "fid","start","end","label","orig_path","has_left","has_right","has_top",
            "path_left","path_right","path_top"
        ])
        miss_w = csv.DictWriter(f_miss, fieldnames=[
            "fid","start","end","label","missing_views","orig_path"
        ])
        cov_w.writeheader()
        miss_w.writeheader()

        for r in orig:
            fid, s, e = r["fid"], r["s"], r["e"]
            found = {}
            paths = {}
            for view in VIEWS:
                p = nearest(idx, fid, view, s, e, used)
                if p:
                    found[view] = True
                    paths[view] = p
                    used.add(p)
                else:
                    found[view] = False
                    missing_counts[view] += 1

            n = sum(found.values())
            if n == 3: all3 += 1
            elif n == 2: two += 1
            elif n == 1: one += 1
            else: zero += 1

            cov_w.writerow({
                "fid": fid, "start": float(s), "end": float(e), "label": r["label"], "orig_path": r["path"],
                "has_left": int(found["left"]), "has_right": int(found["right"]), "has_top": int(found["top"]),
                "path_left": paths.get("left",""), "path_right": paths.get("right",""), "path_top": paths.get("top",""),
            })
            if n < 3:
                miss_w.writerow({
                    "fid": fid, "start": float(s), "end": float(e), "label": r["label"], "orig_path": r["path"],
                    "missing_views": ",".join([v for v in VIEWS if not found[v]])
                })

    # any cropped views never matched to any original?
    unmatched = [t for t in all_cropped if t[4] not in used]
    with open(extras_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["fid","start","end","view","path"])
        w.writeheader()
        for fid, s, e, view, path in unmatched:
            w.writerow({"fid": fid, "start": float(s), "end": float(e), "view": view, "path": path})

    total = len(orig)
    print("\n=== THREE-VIEW COVERAGE REPORT ===")
    print(f"Original segments: {total}")
    print(f"  all 3 views: {all3} ({all3/total:.1%})")
    print(f"  exactly 2 : {two} ({two/total:.1%})")
    print(f"  exactly 1 : {one} ({one/total:.1%})")
    print(f"  none      : {zero} ({zero/total:.1%})")
    print("Missing counts by view:", missing_counts)
    print(f"\nWrote:")
    print(f"  {coverage_csv}")
    print(f"  {missing_csv}")
    print(f"  {extras_csv}")

if __name__ == "__main__":
    sys.exit(main())
