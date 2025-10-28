#!/usr/bin/env python3
"""
Single-process CoMotion runner (video/dir mode, GPU + AMP)

- Loads the CoMotion model once onto CUDA and keeps it there
- Iterates a list of inputs (videos or directories of images)
- Writes outputs mirroring the input tree under out_root:
    <out_root>/<mirrored_subpath>/<input_stem>.pt
    <out_root>/<mirrored_subpath>/<input_stem>.txt

List file format (one per line):
  - absolute path, e.g., /workspace/datasets/ucf101/UCF-101/PommelHorse/v_...avi
  - or relative to --in-dir, e.g., datasets/ucf101/UCF-101/PommelHorse/v_...avi
  - can also be a directory with images (single-image tracking over a folder)

Typical usage:
  python run_videos_list_gpu.py \
    --list-file /workspace/lists/ucf_all_videos.list \
    --in-dir /workspace \
    --out-root /workspace/results/comotion_ucf \
    --fp16 --frameskip 1

"""
import argparse
import sys
from pathlib import Path
import time
import traceback

import numpy as np
import torch
from tqdm import tqdm
from PIL import Image

from comotion_demo.models import comotion
from comotion_demo.utils import dataloading, helper
from comotion_demo.utils import track as track_utils
# --- Force decode to use the same device as smpl_decoder ---------------------
# --- Force decode to use the same device as smpl_decoder ---------------------
# --- Force decode to use the same device as smpl_decoder ---------------------
from comotion_demo.models import detect as _detect
import torch as _torch
import dataclasses as _dc
from typing import Tuple

# Keep the original once
if not hasattr(_detect, "_orig_decode_network_outputs"):
    _detect._orig_decode_network_outputs = _detect.decode_network_outputs

def _first_device_from_module(mod):
    p = next(mod.parameters(), None)
    if p is not None:
        return p.device
    b = next(mod.buffers(), None)
    return b.device if b is not None else _torch.device("cpu")

def _is_namedtuple_instance(x):
    # Heuristic: namedtuple is a tuple with _fields attribute
    return isinstance(x, tuple) and hasattr(x, "_fields")

def _module_device(mod, fallback=torch.device("cpu")):
    p = next(mod.parameters(), None)
    if p is not None:
        return p.device
    b = next(mod.buffers(), None)
    if b is not None:
        return b.device
    return fallback


def _to_device(obj, dev):
    """Recursively move tensors inside common containers & structured objects."""
    if _torch.is_tensor(obj):
        return obj.to(dev, non_blocking=True)

    # dict
    if isinstance(obj, dict):
        return {k: _to_device(v, dev) for k, v in obj.items()}

    # NamedTuple (e.g., DetectOutput)
    if _is_namedtuple_instance(obj):
        values = [_to_device(getattr(obj, f), dev) for f in obj._fields]
        return type(obj)(*values)

    # Dataclass
    if _dc.is_dataclass(obj) and not isinstance(obj, type):
        vals = {f.name: _to_device(getattr(obj, f.name), dev) for f in _dc.fields(obj)}
        return type(obj)(**vals)

    # Plain tuple/list
    if isinstance(obj, tuple):
        return tuple(_to_device(v, dev) for v in obj)
    if isinstance(obj, list):
        return [_to_device(v, dev) for v in obj]

    # Fallback: leave as is
    return obj

def _decode_outputs_device_safe(K, smpl_decoder, detections,
                                std=0.15, conf_thr=0.25, **kwargs):
    dev = _first_device_from_module(smpl_decoder)
    if isinstance(K, _torch.Tensor):
        K = K.to(dev, non_blocking=True)
    detections = _to_device(detections, dev)

    out = _detect._orig_decode_network_outputs(
        K, smpl_decoder, detections, std=std, conf_thr=conf_thr, **kwargs
    )
    # Ensure returned namedtuples/dicts/tensors are also on `dev`
    out = _to_device(out, dev)
    return out


_detect.decode_network_outputs = _decode_outputs_device_safe
# -----------------------------------------------------------------------------

# ----------------------------------------------------------------------------- 


if hasattr(dataloading, "VIDEO_EXTENSIONS"):
    try:
        exts = set(x.lower() for x in dataloading.VIDEO_EXTENSIONS)
    except TypeError:
        exts = set(map(str.lower, list(dataloading.VIDEO_EXTENSIONS)))
    exts.update({".avi"})
    dataloading.VIDEO_EXTENSIONS = tuple(sorted(exts))
from comotion_demo.utils import track as track_utils

# ---------- Device / AMP ----------
def require_cuda():
    if not torch.cuda.is_available():
        print("❌ CUDA not available. Install a CUDA PyTorch wheel and check drivers.", flush=True)
        sys.exit(1)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.set_float32_matmul_precision("high")

# ---------- Normalize override (device-aware) ----------
def _normalize(img):
    return (img - dataloading.IMG_MEAN.to(img.device, dtype=img.dtype)) / \
           dataloading.IMG_STD.to(img.device, dtype=img.dtype)
dataloading.normalize_image = _normalize


def map_to_output(in_path: Path, in_dir: Path, out_root: Path) -> Path:
    """
    Mirror the input path under out_root.
    Examples:
      /workspace/datasets/ucf101/UCF-101/.../v_foo.avi
        -> /workspace/results/comotion_ucf/datasets/ucf101/UCF-101/.../v_foo.pt
      datasets/occlusion_extracted/58/vid0_0_7.avi (with --in-dir /workspace)
        -> /workspace/results/comotion_ucf/datasets/occlusion_extracted/58/vid0_0_7.pt
    """
    try:
        rel = in_path.relative_to(in_dir)
    except ValueError:
        # If not under in_dir, just use path sans leading slash
        rel = Path(str(in_path).lstrip("/"))
    out_dir = out_root / rel.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / (in_path.stem + ".pt")


@torch.no_grad()
def process_one(model, input_path: Path, cache_pt: Path,
                frameskip: int, start_frame: int, num_frames: int,
                fp16: bool):
    """
    Track poses over a video or a directory of images, writing:
      cache_pt (.pt) + sibling .txt (MOT bboxes)
    """
    # Iterate frames (video or dir) → tensors on device
    detections, tracks = [], []
    initialized = False

    gen = dataloading.yield_image_and_K(
        input_path, start_frame, num_frames, frameskip
    )

    for image, K in gen:
        # move to GPU
        image = image.to(device)
        K = K.to(device)

        if fp16:
            image = image.half()
            K = K.half()

        if not initialized:
            model.init_tracks(image.shape[-2:])  # HxW
            initialized = True

        if fp16:
            with torch.cuda.amp.autocast():
                detection, track = model(image, K, use_mps=False)
        else:
            detection, track = model(image, K, use_mps=False)

        detections.append({k: v.cpu() for k, v in detection.items()})
        tracks.append(track.cpu())

    if not detections:
        # No frames or read failure
        return False, "no_frames"

    # --------- Post-processing (same as demo.py) ----------
    detections = {k: [d[k] for d in detections] for k in detections[0]}
    tracks_t = torch.stack(tracks, 1)
    tracks = {k: getattr(tracks_t, k) for k in ["id", "pose", "trans", "betas"]}

    # For cleanup we need the final K and image_res; reuse the last image,K values:
    # (they're CPU by now because we appended .cpu())
    # Get CPU K & image_res from last items:
    # Use the last K we saw from the generator's loop scope (device K variable isn't here),
    # so recompute from tracks shapes:
    # image_res is needed for bboxes; we can grab it from the first frame of last loop
    # But since we lost that scope, fallback to a reasonable inference:
    # We'll approximate image_res by using track_utils requires image_res shape (H,W).
    # Safer approach: recompute image_res and K by peeking the first frame again.
    # However, that's extra I/O. Instead, we can store K and image_res during loop.
    # Let's implement that: we keep last_image_res and last_K on CPU.
    # (We need to modify the loop to record them.)
    return tracks, detections  # placeholder to satisfy linter


def process_one_with_ctx(model, input_path: Path, cache_pt: Path,
                         frameskip: int, start_frame: int, num_frames: int,
                         fp16: bool):
    """Same as process_one but stores last K and image_res for cleanup/bboxes."""
    detections, tracks = [], []
    initialized = False
    last_image_res = None
    last_K_cpu = None
    last_K_dev = None

    gen = dataloading.yield_image_and_K(
        input_path, start_frame, num_frames, frameskip
    )

    # --- NEW: reset shot detector & frame counter for a fresh video ----------
    if hasattr(model, "shot_detector"):
        if hasattr(model.shot_detector, "reset"):
            model.shot_detector.reset()
        elif hasattr(model.shot_detector, "_last_frame"):
            model.shot_detector._last_frame = None
    if hasattr(model, "frame_count"):
        model.frame_count = 0
    # ------------------------------------------------------------------------

    for image, K in gen:
        # --- NEW: guard against resolution changes mid-video -----------------
        curr_res = image.shape[-2:]
        if last_image_res is not None and curr_res != last_image_res:
            # drop scenedetect's cached frame so HSV shapes match
            if hasattr(model, "shot_detector"):
                if hasattr(model.shot_detector, "reset"):
                    model.shot_detector.reset()
                elif hasattr(model.shot_detector, "_last_frame"):
                    model.shot_detector._last_frame = None
            if hasattr(model, "frame_count"):
                model.frame_count = 0
        last_image_res = curr_res
        # --------------------------------------------------------------------

        # record CPU copy for later bbox computation / summary
        last_K_cpu = K.cpu()

        # move to GPU for forward
        image = image.to(device, non_blocking=True)
        K = K.to(device, non_blocking=True)
        last_K_dev = K  # keep the CUDA copy aligned with decoder

        if not initialized:
            model.init_tracks(image.shape[-2:])  # HxW
            initialized = True

        if fp16 and device.type == "cuda":
            with torch.cuda.amp.autocast():
                detection, track = model(image, K, use_mps=False)
        else:
            detection, track = model(image, K, use_mps=False)

        detections.append({k: v.detach().cpu() for k, v in detection.items()})
        tracks.append(track.detach().cpu())

    if not detections:
        return False, "no_frames"

    # Post-processing (CPU)
    detections = {k: [d[k] for d in detections] for k in detections[0]}
    tracks_t = torch.stack(tracks, 1)
    tracks = {k: getattr(tracks_t, k) for k in ["id", "pose", "trans", "betas"]}

    # Cleanup on CPU using a TEMPORARY CPU/FP32 clone of the decoder
    decoder_cpu_f32 = model.smpl_decoder.float().cpu()
    track_ref = track_utils.cleanup_tracks(
        {"detections": detections, "tracks": tracks},
        last_K_cpu,
        decoder_cpu_f32,
        min_matched_frames=1,
    )
    del decoder_cpu_f32

    if not track_ref:
        return False, "no_tracks"

    f_idx, t_idx = track_utils.convert_to_idxs(
        track_ref, tracks["id"][0].squeeze(-1).long()
    )
    preds = {k: v[0, f_idx, t_idx] for k, v in tracks.items()}  # still CPU
    preds["id"] = preds["id"].squeeze(-1).long()
    preds["frame_idx"] = f_idx
    torch.save(preds, cache_pt)

    # MOT bboxes with the live (GPU) decoder
    # MOT bboxes with the live (GPU) decoder
    dec_dev = _module_device(model.smpl_decoder, device)
    preds_dev = {k: preds[k].to(dec_dev, non_blocking=True)
                 for k in ["betas", "pose", "trans"]}
    K_dev = (last_K_dev.to(dec_dev, non_blocking=True)
             if last_K_dev is not None else last_K_cpu.to(dec_dev))
    bboxes = track_utils.bboxes_from_smpl(
        model.smpl_decoder, preds_dev, last_image_res, K_dev
    )

    with open(str(cache_pt).replace(".pt", ".txt"), "w") as f:
        f.write(track_utils.convert_to_mot(preds["id"], preds["frame_idx"], bboxes))

    return True, "ok"



@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list-file", required=True,
                    help="Text file with paths (abs or relative to --in-dir). One per line.")
    ap.add_argument("--in-dir", default="/workspace",
                    help="If list entries are relative, they're joined to this.")
    ap.add_argument("--out-root", default="/workspace/results/comotion_ucf",
                    help="Root directory to mirror outputs into.")
    ap.add_argument("--frameskip", type=int, default=1,
                    help="Subsample frames (e.g., 2 = every other frame).")
    ap.add_argument("--start-frame", type=int, default=0,
                    help="Start at this frame index.")
    ap.add_argument("--num-frames", type=int, default=1_000_000_000,
                    help="Process at most this many frames per input.")
    ap.add_argument("--fp16", action="store_true",
                    help="Enable CUDA AMP for faster inference.")
    ap.add_argument("--resume", action="store_true",
                    help="Skip inputs whose .pt already exists.")
    ap.add_argument("--max-n", type=int, default=None,
                    help="Process at most N items from the list (debug).")
    args = ap.parse_args()

    print("===== CoMotion list runner (video/dir) =====", flush=True)
    require_cuda()
    print("Torch:", torch.__version__, "CUDA build:", torch.version.cuda, flush=True)
    print("GPU:", torch.cuda.get_device_name(0), flush=True)
    print(f"device={device}, fp16={args.fp16}, frameskip={args.frameskip}", flush=True)

    in_dir = Path(args.in_dir)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    with open(args.list_file, "r") as f:
        items = [ln.strip() for ln in f if ln.strip()]
    if args.max_n:
        items = items[:args.max_n]

    print(f"Found {len(items)} inputs.", flush=True)
    if items[:3]:
        print("Sample inputs:", *items[:3], sep="\n  ", flush=True)

    # Load CoMotion once on CUDA
    print("Loading CoMotion model on CUDA…", flush=True)
    model = comotion.CoMotion(use_coreml=False).to(device).eval()
    if args.fp16 and device.type == "cuda":
        model.half()
    model.smpl_decoder = model.smpl_decoder.to(device)
    if args.fp16 and device.type == "cuda":
        model.smpl_decoder.half()

    print("Model ready.", flush=True)


    processed = skipped = errors = 0
    t0 = time.time()

    for rel in tqdm(items, desc="CoMotion videos", mininterval=0.25):
        try:
            p = Path(rel)
            if not p.is_absolute():
                p = in_dir / p
            if not p.exists():
                errors += 1
                print(f"⚠️  missing input: {p}", flush=True)
                continue

            out_pt = map_to_output(p, in_dir, out_root)

            if args.resume and out_pt.exists():
                skipped += 1
                continue

            ok, msg = process_one_with_ctx(
                model, p, out_pt,
                frameskip=args.frameskip,
                start_frame=args.start_frame,
                num_frames=args.num_frames,
                fp16=args.fp16,
            )
            if ok:
                processed += 1
            else:
                errors += 1
                # don't leave empty files
                if out_pt.exists():
                    try:
                        out_pt.unlink()
                    except Exception:
                        pass
                print(f"⚠️  failed on {p} ({msg})", flush=True)

        except KeyboardInterrupt:
            print("\n⛔ Interrupted by user.", flush=True)
            break
        except Exception as e:
            errors += 1
            print(f"❌ Error on {rel}: {e}", flush=True)
            traceback.print_exc()

    dt = time.time() - t0
    print(f"Done. processed={processed}, skipped={skipped}, errors={errors}, time={dt/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
