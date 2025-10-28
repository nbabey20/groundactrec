#!/usr/bin/env python3
import argparse, random
from pathlib import Path

def expected_out_pt(rel_line: str, in_dir: Path, results_root: Path) -> Path:
    """
    Map a list entry (usually 'datasets/.../file.avi') to the CoMotion output .pt:
      datasets/ucf101/UCF-101/<Action>/v_foo.avi
        -> results_root/ucf101/UCF-101/<Action>/v_foo.pt
      datasets/occlusion_extracted/<N>/vid0_0_7.avi
        -> results_root/occlusion_extracted/<N>/vid0_0_7.pt
    If the line is absolute, we first try to make it relative to in_dir.
    """
    p = Path(rel_line.strip())
    if p.is_absolute():
        try:
            rel = p.relative_to(in_dir)
        except ValueError:
            rel = p  # not under in_dir, just use as-is
    else:
        rel = p

    # strip the leading "datasets/" if present
    parts = rel.parts
    if len(parts) >= 2 and parts[0] == "datasets":
        sub = Path(*parts[1:])
    else:
        sub = rel

    return (results_root / sub).with_suffix(".pt")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", required=True, help="Path to ucf_all_videos.list (relative paths).")
    ap.add_argument("--out-dir", default=None, help="Where to write shards (default: alongside --list).")
    ap.add_argument("--num-shards", type=int, default=5)
    ap.add_argument("--shuffle", action="store_true")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--in-dir", default="/workspace", help="Root that list entries are relative to.")
    ap.add_argument("--results-root", default="/workspace/results/comotion_ucf",
                    help="Root where CoMotion writes outputs.")
    ap.add_argument("--require-txt", action="store_true",
                    help="Only consider a clip 'done' if BOTH .pt and .txt exist.")
    args = ap.parse_args()

    in_path = Path(args.list)
    out_dir = Path(args.out_dir) if args.out_dir else in_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    in_dir = Path(args.in_dir).resolve()
    results_root = Path(args.results_root).resolve()

    # read & clean
    with open(in_path, "r") as f:
        lines = [ln.strip() for ln in f]
    lines = [ln for ln in lines if ln and not ln.startswith("#")]

    # filter to remaining (not yet processed)
    remaining = []
    done = 0
    for ln in lines:
        out_pt = expected_out_pt(ln, in_dir, results_root)
        out_txt = out_pt.with_suffix(".txt")
        have_pt = out_pt.exists()
        have_txt = out_txt.exists()
        is_done = (have_pt and have_txt) if args.require_txt else have_pt
        if is_done:
            done += 1
        else:
            remaining.append(ln)

    if args.shuffle:
        random.seed(args.seed)
        random.shuffle(remaining)

    base = in_path.stem
    # write the remaining list
    rem_path = out_dir / f"{base}.remaining.list"
    with open(rem_path, "w") as w:
        for ln in remaining:
            w.write(ln + "\n")

    # balanced round-robin split of the remaining
    shard_paths = []
    for i in range(args.num_shards):
        shard_path = out_dir / f"{base}.remaining.shard{i+1}.list"
        shard_paths.append(shard_path)
        with open(shard_path, "w") as w:
            for j, ln in enumerate(remaining):
                if j % args.num_shards == i:
                    w.write(ln + "\n")

    print(f"Total in list: {len(lines)}")
    print(f"Already done : {done} (require_txt={args.require_txt})")
    print(f"Remaining    : {len(remaining)}")
    print(f"Wrote remaining list: {rem_path}")
    for p in shard_paths:
        print("wrote", p)

if __name__ == "__main__":
    main()
