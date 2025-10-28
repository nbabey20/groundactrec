#!/usr/bin/env python
"""
extract_vjepa_frame_feats.py

 Sanity check (one clip, prints shapes only)
$ python extract_vjepa_frame_feats.py --src /workspace/datasets/inhard --test one

 Full sweep
$ python extract_vjepa_frame_feats.py --src /workspace/datasets/inhard \
      --dst /workspace/features/vjepa2_vitg_g/frame \
      --num-workers 8
"""
import argparse, gzip, json, signal
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch
from decord import VideoReader, cpu
from tqdm import tqdm
from contextlib import nullcontext
# ───────── helpers for path handling ─────────
from pathlib import Path
import time
from functools import partial


amp_ctx = torch.amp.autocast(device_type="cuda") if torch.cuda.is_available() else nullcontext()

# ─────────── backbone & pre-processor (frozen) ───────────
# ─────────── backbone & pre-processor (frozen) ───────────
device = "cuda" if torch.cuda.is_available() else "cpu"
processor = torch.hub.load("facebookresearch/vjepa2", "vjepa2_preprocessor")

# hub returns (model, cfg)  →  we only need model
backbone, _ = torch.hub.load("facebookresearch/vjepa2",
                             "vjepa2_vit_giant_384")
backbone = backbone.to(device).eval()

def dst_for(video_path: Path, subset_tag: str | None):
    rel  = video_path.relative_to(SRC_ROOT)            # <Action>/<clip>.mp4
    sub  = "_sample" if subset_tag else ""             # "" for full run
    head = DST_ROOT / sub / rel.parts[0]               # action folder
    head.mkdir(parents=True, exist_ok=True)
    return head / (video_path.stem + ".pt")


def sample_uniform(vr, num=64):
    idx = np.linspace(0, len(vr)-1, num=num, dtype=int)
    return vr.get_batch(idx).asnumpy()                 # (T, H, W, 3)

def encode(video_path: Path, subset_tag: str | None = None, overwrite: bool = False):
    dst = dst_for(video_path, subset_tag)

    # Skip if a valid file already exists (shape/dtype check)
    if not overwrite and dst.exists():
        try:
            t = torch.load(dst, map_location="cpu", weights_only=True)
            if isinstance(t, torch.Tensor) and t.dtype == torch.float16 and t.shape == (64, 1408):
                return {"id": video_path.stem, "feat_path": str(dst), "skipped": True}
        except Exception:
            # corrupt or old-format file → fall through and recompute
            pass

    vr   = VideoReader(str(video_path), ctx=cpu(0))
    imgs = sample_uniform(vr, 64)
    inp  = torch.stack(processor(list(imgs)), dim=0).to(device)

    with torch.no_grad(), amp_ctx:
        feats = backbone(inp)
        if isinstance(feats, dict):      x = feats["x"]
        elif isinstance(feats, (list, tuple)): x = feats[0]
        else:                            x = feats
        x = x.squeeze(0)                 # (8192, 1408)
        cls = x.view(64, 128, -1)[:, 0, :].half()   # (64, 1408)

    # atomic-ish write to avoid half-written files
    tmp = dst.with_suffix(".pt.tmp")
    torch.save(cls.cpu(), tmp)
    tmp.replace(dst)

    return {"id": video_path.stem, "feat_path": str(dst), "skipped": False}


                                  # used by --test paths


# ──────────── CLI / multiprocessing harness ────────────
def _worker(arg):
    src_p, dst_root = arg
    try:    return encode(src_p, dst_root)
    except Exception as e: return {"error": f"{src_p}: {e}"}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src",  required=True,
                    help="root folder with videos")
    ap.add_argument("--dst",  default="/workspace/results/vjepa",
                    help="root folder for result hierarchy")
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--sample", type=int,
                    help="extract N clips into _sample/ for inspection")
    ap.add_argument("--overwrite", action="store_true",
                help="recompute even if output .pt already exists")

    args = ap.parse_args()

    # make path vars visible to helper functions
    global SRC_ROOT, DST_ROOT
    SRC_ROOT = Path(args.src).expanduser()
    DST_ROOT = Path(args.dst).expanduser()
    DST_ROOT.mkdir(parents=True, exist_ok=True)

    clips = sorted(SRC_ROOT.rglob("*.mp4"))
    if not clips:
        raise FileNotFoundError(f"No .mp4 under {SRC_ROOT}")
    print(f"Found {len(clips)} clips.")

    # ---------- SAMPLE MODE ----------
    if args.sample:
        for vp in clips[:args.sample]:
            encode(vp, subset_tag="_sample")
        print(f"Sample clips written to {DST_ROOT / '_sample'}")
        return

    # ---------- FULL RUN ------------
    start = time.time()
    processed = skipped = 0

    if args.num_workers == 0:                       # single-process path
        for vp in tqdm(clips, unit="clip"):
            res = encode(vp, overwrite=args.overwrite)
            skipped   += int(res.get("skipped", False))
            processed += int(not res.get("skipped", False))
    else:                                           # multiprocessing path
        work = partial(encode, subset_tag=None, overwrite=args.overwrite)
        original_sigint = signal.signal(signal.SIGINT, signal.SIG_IGN)
        with Pool(args.num_workers) as pool:
            signal.signal(signal.SIGINT, original_sigint)
            for res in tqdm(pool.imap_unordered(work, clips),
                        total=len(clips), unit="clip"):
                skipped   += int(res.get("skipped", False))
                processed += int(not res.get("skipped", False))

    elapsed = time.time() - start
    print(f"\nFinished {len(clips)} clips in {elapsed/60:.1f} min "
          f"({elapsed/len(clips):.2f} s/clip)  "
          f"[processed={processed}, skipped={skipped}]")



if __name__ == "__main__":
    import time
    main()