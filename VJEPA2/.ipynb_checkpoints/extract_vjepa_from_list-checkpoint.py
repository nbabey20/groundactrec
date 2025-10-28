#!/usr/bin/env python3
"""
Extract V-JEPA features for every video path in a list file.

Usage:
  python extract_vjepa_from_list.py \
    --list /mnt/data/ucf_all_videos.list \
    --in-dir /workspace \
    --dst /workspace/results/ucf_vjepa \
    --num-workers 8

Notes:
- Supports paths like:
    datasets/occlusion_extracted/<N>/vid*.avi
    datasets/ucf101/UCF-101/<Action>/v_*.avi
- Destination layout:
    dst/occlusion_extracted/<N>/<stem>.pt
    dst/UCF-101/<Action>/<stem>.pt
- Each .pt contains a (64, 1408) float16 tensor (1 per sampled frame).
"""

import argparse, signal, time
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch
from decord import VideoReader, cpu
from contextlib import nullcontext
from tqdm import tqdm
import torch.multiprocessing as mp

torch.backends.cudnn.benchmark = True  # speed up convs for fixed input sizes

# Globals that will be (re)initialized per worker
_PROCESSOR = None
_BACKBONE  = None
_DEVICE    = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------- AMP / Device ----------------

def _init_worker():
    """Executed once in each worker process; safe CUDA init with spawn."""
    global _PROCESSOR, _BACKBONE, _DEVICE
    _DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _PROCESSOR = torch.hub.load("facebookresearch/vjepa2", "vjepa2_preprocessor")
    _BACKBONE, _ = torch.hub.load("facebookresearch/vjepa2", "vjepa2_vit_giant_384")
    _BACKBONE = _BACKBONE.to(_DEVICE).eval()


# ---------------- Path helpers ----------------
def parse_dst_path(src_abs: Path, src_root: Path, dst_root: Path) -> Path:
    """
    Map an absolute source path → destination PT path based on two known patterns:
      - .../datasets/occlusion_extracted/<N>/<file>.avi
      - .../datasets/ucf101/UCF-101/<Action>/<file>.avi
    Falls back to mirroring the tail under dst_root/misc/ if pattern is unknown.
    """
    try:
        rel = src_abs.relative_to(src_root)
    except ValueError:
        # If not under src_root, still try to parse from its parts
        rel = Path(*src_abs.parts)

    parts = rel.parts

    # Pattern A: occlusion_extracted/<N>/...
    if "occlusion_extracted" in parts:
        idx = parts.index("occlusion_extracted")
        subset = parts[idx + 1] if idx + 1 < len(parts) else "unknown"
        out_dir = dst_root / "occlusion_extracted" / subset
        out_dir.mkdir(parents=True, exist_ok=True)
        return out_dir / (src_abs.stem + ".pt")

    # Pattern B: ucf101/UCF-101/<Action>/...
    if "UCF-101" in parts:
        idx = parts.index("UCF-101")
        action = parts[idx + 1] if idx + 1 < len(parts) else "UnknownAction"
        out_dir = dst_root / "UCF-101" / action
        out_dir.mkdir(parents=True, exist_ok=True)
        return out_dir / (src_abs.stem + ".pt")

    # Fallback: keep a shallow mirror for anything else
    out_dir = dst_root / "misc"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / (src_abs.stem + ".pt")


def sample_uniform(vr: VideoReader, num: int = 64):
    total = len(vr)
    if total <= 0:
        raise RuntimeError("Empty video / unsupported codec.")
    idx = np.linspace(0, total - 1, num=num, dtype=int)
    return vr.get_batch(idx).asnumpy()  # (T, H, W, 3)


def encode_one(src_abs: Path, src_root: Path, dst_root: Path, overwrite: bool = False):
    global _PROCESSOR, _BACKBONE, _DEVICE
    dst = parse_dst_path(src_abs, src_root, dst_root)

    if dst.exists() and not overwrite:
        try:
            t = torch.load(dst, map_location="cpu", weights_only=True)
            if isinstance(t, torch.Tensor) and t.dtype == torch.float16 and t.shape == (64, 1408):
                return {"src": str(src_abs), "dst": str(dst), "skipped": True}
        except Exception:
            pass

    vr = VideoReader(str(src_abs), ctx=cpu(0))
    imgs = sample_uniform(vr, 64)  # (T, H, W, 3), numpy

    # Preprocessor returns a list of CHW tensors on CPU
    frames = _PROCESSOR(list(imgs))                    # list[(C,H,W)]
    inp = torch.stack(frames, dim=0).pin_memory()      # (T, C, H, W) pinned
    inp = inp.to(_DEVICE, non_blocking=True)

    with torch.inference_mode(), torch.amp.autocast('cuda' if _DEVICE.type == 'cuda' else 'cpu'):
        feats = _BACKBONE(inp)
        if isinstance(feats, dict):
            x = feats.get("x", next(iter(feats.values())))
        elif isinstance(feats, (list, tuple)):
            x = feats[0]
        else:
            x = feats
        x = x.squeeze(0).view(64, 128, -1)[:, 0, :].half()  # (64, 1408) fp16

    tmp = dst.with_suffix(".pt.tmp")
    torch.save(x.cpu(), tmp)
    tmp.replace(dst)
    return {"src": str(src_abs), "dst": str(dst), "skipped": False}


# ---------------- Worker ----------------
def _worker(arg):
    src_abs, src_root, dst_root, overwrite = arg
    try:
        return encode_one(src_abs, src_root, dst_root, overwrite=overwrite)
    except Exception as e:
        return {"src": str(src_abs), "error": str(e)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", required=True, help="Text file: one video path per line.")
    ap.add_argument("--in-dir", default="/workspace", help="Join relative paths to this root.")
    ap.add_argument("--dst", default="/workspace/results/ucf_vjepa", help="Output root.")
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--overwrite", action="store_true", help="Recompute even if output exists.")
    args = ap.parse_args()

    src_root = Path(args.in_dir).expanduser().resolve()
    dst_root = Path(args.dst).expanduser().resolve()
    dst_root.mkdir(parents=True, exist_ok=True)

    # Read list file; ignore blank/comment lines
    list_path = Path(args.list).expanduser().resolve()
    with open(list_path, "r") as f:
        raw = [ln.strip() for ln in f]
    rels = [ln for ln in raw if ln and not ln.startswith("#")]

    # Resolve to absolute paths
    src_paths = []
    for p in rels:
        pth = Path(p)
        if not pth.is_absolute():
            pth = (src_root / p).resolve()
        if pth.exists():
            src_paths.append(pth)
        else:
            print(f"⚠️  missing: {pth}")
    if not src_paths:
        raise FileNotFoundError("No valid paths from list file were found on disk.")

    print(f"Found {len(src_paths)} videos.")

    start = time.time()
    processed = skipped = 0

    if args.num_workers == 0:
        _init_worker()
        for vp in tqdm(src_paths, unit="clip"):
            res = _worker((vp, src_root, dst_root, args.overwrite))
            if "error" in res:
                print(f"❌ {res['src']}: {res['error']}")
            else:
                skipped += int(res.get("skipped", False))
                processed += int(not res.get("skipped", False))
    else:
        work = [(vp, src_root, dst_root, args.overwrite) for vp in src_paths]
        original_sigint = signal.signal(signal.SIGINT, signal.SIG_IGN)

        # Make CUDA multiprocessing safe
        try:
            mp.set_start_method("spawn", force=True)
        except RuntimeError:
            pass  # already set

        with Pool(args.num_workers, initializer=_init_worker) as pool:
            signal.signal(signal.SIGINT, original_sigint)
            for res in tqdm(pool.imap_unordered(_worker, work), total=len(work), unit="clip"):
                if "error" in res:
                    print(f"❌ {res['src']}: {res['error']}")
                else:
                    skipped += int(res.get("skipped", False))
                    processed += int(not res.get("skipped", False))

    dt = time.time() - start
    print(f"\nFinished {len(src_paths)} clips in {dt/60:.1f} min "
          f"({dt/len(src_paths):.2f} s/clip)  "
          f"[processed={processed}, skipped={skipped}]")

if __name__ == "__main__":
    main()
