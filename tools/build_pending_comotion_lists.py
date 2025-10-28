#!/usr/bin/env python3
"""
Build lists of frames/clips that still need per-frame CoMotion (single-image) .pt files.

Compares:
  frames_root:  /workspace/frames/<label>/<clip>/*.jpg
  results_root: /workspace/results/comotion_frames/<label>/<clip>/*.pt

Outputs (default out_dir=/workspace/lists):
  - pending_summary.csv
  - pending_frames_all.list
  - pending_clips_all.list
  - pending_frames_part{1..K}.list    # NON-OVERLAPPING, ROUND-ROBIN SHARDS
  - pending_clips_part{1..K}.list     # NON-OVERLAPPING, GREEDY BALANCED

Run:
  python /workspace/tools/build_pending_comotion_lists.py \
    --frames-root /workspace/frames \
    --results-root /workspace/results/comotion_frames \
    --out-dir /workspace/lists \
    --workers 5
"""
import argparse
from pathlib import Path
import math
import pandas as pd
from typing import List, Tuple

def rel_to_workspace_fast(p: Path) -> str:
    s = str(p)
    pref = "/workspace/"
    return s[len(pref):] if s.startswith(pref) else s

def list_label_clip_dirs(frames_root: Path) -> List[Tuple[str, str, Path]]:
    out = []
    for label_dir in sorted(frames_root.iterdir()):
        if not label_dir.is_dir(): 
            continue
        for clip_dir in sorted(label_dir.iterdir()):
            if not clip_dir.is_dir():
                continue
            # fast existence check
            if next(clip_dir.glob("*.jpg"), None) is not None:
                out.append((label_dir.name, clip_dir.name, clip_dir))
    return out

def collect_pending_for_clip(label: str, clip: str, clip_dir: Path, results_root: Path):
    out_dir = results_root / label / clip
    jpgs = sorted(clip_dir.glob("*.jpg"))
    pending = []
    done = 0
    for jpg in jpgs:
        if (out_dir / f"{jpg.stem}.pt").exists():
            done += 1
        else:
            pending.append(jpg)
    return jpgs, pending, done

def greedy_balance_by_weight(items_with_weights, k: int):
    """
    Greedy bin packing by weight (largest-first).
    items_with_weights: list of (item, weight), weight >= 0
    Returns: list[b] -> list of (item, weight)
    """
    bins_items = [[] for _ in range(k)]
    bins_w = [0] * k
    items_with_weights = sorted(items_with_weights, key=lambda x: x[1], reverse=True)
    for item, w in items_with_weights:
        bi = min(range(k), key=lambda i: bins_w[i])
        bins_items[bi].append((item, w))
        bins_w[bi] += w
    return bins_items, bins_w

def round_robin_shards(seq: List[str], k: int) -> List[List[str]]:
    """
    Deterministic, non-overlapping sharding: shard i gets seq[i::k].
    Assumes seq is already deterministically ordered (we sort).
    """
    return [seq[i::k] for i in range(k)]

def verify_disjoint(parts: List[List[str]], universe: List[str], label: str):
    flat = [p for part in parts for p in part]
    if len(flat) != len(universe):
        raise SystemExit(f"[{label}] Size mismatch: union(parts)={len(flat)} vs universe={len(universe)}")
    if len(set(flat)) != len(flat):
        raise SystemExit(f"[{label}] Duplicate detected across parts.")
    # Disjointness by pairwise intersection (cheap check)
    seen = set()
    for i, part in enumerate(parts):
        inter = seen.intersection(part)
        if inter:
            raise SystemExit(f"[{label}] Overlap detected between shards: {len(inter)} items")
        seen.update(part)

def write_list(path: Path, lines: List[str]):
    with path.open("w") as f:
        for s in lines:
            f.write(s + "\n")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames-root", type=Path, default=Path("/workspace/frames"))
    ap.add_argument("--results-root", type=Path, default=Path("/workspace/results/comotion_frames"))
    ap.add_argument("--out-dir", type=Path, default=Path("/workspace/lists"))
    ap.add_argument("--workers", type=int, default=5)
    args = ap.parse_args()

    k = max(1, int(args.workers))
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print("🔎 Scanning label/clip directories…")
    label_clip_dirs = list_label_clip_dirs(args.frames_root)
    print(f"   found {len(label_clip_dirs)} clip folders with frames.")

    summary_rows = []
    pending_frames: List[str] = []
    clip_pending: List[Tuple[str, int]] = []

    for idx, (label, clip, clip_dir) in enumerate(label_clip_dirs, 1):
        if idx % 200 == 0:
            print(f"   …processed {idx}/{len(label_clip_dirs)} clips")

        jpgs, pend, done = collect_pending_for_clip(label, clip, clip_dir, args.results_root)
        n_total = len(jpgs)
        n_pending = len(pend)

        clip_dir_rel = rel_to_workspace_fast(clip_dir)
        out_dir_rel  = rel_to_workspace_fast(args.results_root / label / clip)

        if n_pending > 0:
            clip_pending.append((clip_dir_rel, n_pending))
            for pf in pend:
                pending_frames.append(rel_to_workspace_fast(pf))

        summary_rows.append({
            "label": label,
            "clip": clip,
            "clip_dir": clip_dir_rel,
            "out_dir": out_dir_rel,
            "n_frames_total": n_total,
            "n_done": done,
            "n_pending": n_pending,
            "pct_done": (100.0 * done / n_total) if n_total else 0.0
        })

    # Write summary & "all" lists
    print("📝 Writing summary & all-pending lists…")
    summary_df = pd.DataFrame(summary_rows).sort_values(
        ["n_pending","label","clip"], ascending=[False, True, True]
    )
    summary_csv = args.out_dir / "pending_summary.csv"
    summary_df.to_csv(summary_csv, index=False)
    print(f"✔ summary: {summary_csv} (clips={len(summary_df)})")

    # Important: sort frames deterministically before sharding
    pending_frames = sorted(set(pending_frames))
    frames_all = args.out_dir / "pending_frames_all.list"
    write_list(frames_all, pending_frames)
    print(f"✔ frames list: {frames_all} (pending frames={len(pending_frames)})")

    clips_all = args.out_dir / "pending_clips_all.list"
    write_list(clips_all, [c for (c, n) in clip_pending if n > 0])
    print(f"✔ clips list: {clips_all} (pending clips={sum(1 for _,n in clip_pending if n>0)})")

    # ---------- SHARD FRAMES: deterministic round-robin (guaranteed different) ----------
    print("📦 Sharding frames (round-robin)…")
    frame_parts = round_robin_shards(pending_frames, k)
    verify_disjoint(frame_parts, pending_frames, label="frames")
    for i, part in enumerate(frame_parts, 1):
        part_path = args.out_dir / f"pending_frames_part{i}.list"
        write_list(part_path, part)
        print(f"✔ frames part {i}: {part_path} (frames={len(part)})")

    # ---------- SHARD CLIPS: greedy by pending count (balanced) ----------
    print("📦 Sharding clips (greedy by pending count)…")
    clip_bins, clip_w = greedy_balance_by_weight(clip_pending, k)
    clip_parts = [[c for (c, _w) in bin_items] for bin_items in clip_bins]
    # verify disjointness against 'clips_all'
    clips_universe = [c for (c, n) in clip_pending if n > 0]
    verify_disjoint(clip_parts, clips_universe, label="clips")
    for i, (bin_items, total_w) in enumerate(zip(clip_bins, clip_w), 1):
        part_path = args.out_dir / f"pending_clips_part{i}.list"
        write_list(part_path, [c for (c, _w) in bin_items])
        print(f"✔ clips part {i}: {part_path} (clips={len(bin_items)}, pending_frames≈{int(total_w)})")

    print("✅ Done.")

if __name__ == "__main__":
    main()
