#!/usr/bin/env python3
import argparse
from pathlib import Path

def load_lines(p: Path):
    return [ln.strip() for ln in p.read_text().splitlines() if ln.strip()]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ucf-root", default="/workspace/datasets/ucf101", help="root that contains UCF-101/ and ucfTrainTestlist/")
    ap.add_argument("--occ-root", default="/workspace/datasets/occlusion_extracted", help="root that contains occlusion subset organized as <id>/<file>.avi")
    ap.add_argument("--out-list", default="/workspace/lists/ucf_all_videos.list", help="output list (paths relative to /workspace)")
    ap.add_argument("--split-id", default="01", choices=["01","02","03"], help="which official split to use")
    args = ap.parse_args()

    ucf_root = Path(args.ucf_root)
    vid_root = ucf_root / "UCF-101"
    split_dir= ucf_root / "ucfTrainTestlist"
    train_txt = split_dir / f"trainlist{args.split_id}.txt"
    test_txt  = split_dir / f"testlist{args.split_id}.txt"

    assert vid_root.is_dir(), f"Missing {vid_root}"
    assert train_txt.is_file(), f"Missing {train_txt}"
    assert test_txt.is_file(), f"Missing {test_txt}"

    # Paths in the split files are like: ApplyEyeMakeup/v_ApplyEyeMakeup_g01_c01.avi
    def make_ucf_abs(rel_line: str) -> str:
        return f"datasets/ucf101/UCF-101/{rel_line}"

    train_rel = [ln.split()[0] for ln in load_lines(train_txt)]   # robust if label column exists
    test_rel  = [ln.split()[0] for ln in load_lines(test_txt)]

    ucf_items = [make_ucf_abs(p) for p in (train_rel + test_rel)]

    # Occlusion subset: /workspace/datasets/occlusion_extracted/<id>/<file>.avi
    occ_root = Path(args.occ_root)
    occ_items = []
    if occ_root.is_dir():
        for avi in occ_root.rglob("*.avi"):
            # produce path relative to /workspace
            s = str(avi)
            if s.startswith("/workspace/"):
                s = s[len("/workspace/"):]
            occ_items.append(s)

    # Deduplicate and sort (deterministic)
    all_items = sorted(set(ucf_items + occ_items))

    outp = Path(args.out_list)
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text("\n".join(all_items) + ("\n" if all_items else ""))
    print(f"✔ wrote {outp} (n={len(all_items)})")

if __name__ == "__main__":
    main()
