#!/usr/bin/env python3
import argparse, os, sys, time, traceback
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

# CoMotion imports
from comotion_demo.models import comotion
from comotion_demo.utils import dataloading

# ---------- Helpers ----------
def require_cuda():
    if not torch.cuda.is_available():
        print("❌ CUDA not available. Check torch build / drivers.", flush=True)
        sys.exit(1)

def _normalize(img):
    return (img - dataloading.IMG_MEAN.to(img.device, dtype=img.dtype)) / \
           dataloading.IMG_STD.to(img.device, dtype=img.dtype)
dataloading.normalize_image = _normalize

def map_rel_to_out(rel: str, in_prefix="frames/", out_root="/workspace/results/comotion_frames"):
    # rel: frames/<label>/<clip>/000123.jpg  (ideally)
    if rel.startswith(in_prefix):
        sub = rel[len(in_prefix):]
    else:
        # try to find "frames/" inside
        idx = rel.find(in_prefix)
        sub = rel[idx+len(in_prefix):] if idx >= 0 else Path(rel).name
    out_dir = Path(out_root) / Path(sub).parent
    out_pt  = out_dir / (Path(sub).stem + ".pt")
    return out_dir, out_pt

@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list-file", required=True,
                    help="Text file with relative frame paths like frames/<label>/<clip>/000123.jpg")
    ap.add_argument("--in-dir", default="/workspace", help="Prefix to prepend to list-file entries")
    ap.add_argument("--out-root", default="/workspace/results/comotion_frames",
                    help="Root for per-frame .pt outputs")
    ap.add_argument("--fp16", action="store_true", help="Enable AMP fp16 on GPU")
    ap.add_argument("--conf-thr", type=float, default=0.25, help="Confidence cutoff for decode")
    ap.add_argument("--std", type=float, default=0.15, help="Decode std (NMS-like sensitivity)")
    ap.add_argument("--max-n", type=int, default=None, help="Process at most N frames (debug)")
    args = ap.parse_args()

    print("===== CoMotion single-image GPU runner =====", flush=True)
    print(f"list-file: {args.list_file}", flush=True)
    print(f"in-dir:    {args.in_dir}", flush=True)
    print(f"out-root:  {args.out_root}", flush=True)
    print(f"fp16:      {args.fp16}", flush=True)

    require_cuda()
    device = torch.device("cuda:0")
    print("Torch:", torch.__version__, "CUDA build:", torch.version.cuda, flush=True)
    print("GPU:", torch.cuda.get_device_name(0), flush=True)
    torch.set_float32_matmul_precision("high")

    # Build model ONCE on GPU
    print("Loading CoMotion model on CUDA…", flush=True)
    model = comotion.CoMotion(use_coreml=False)
    model = model.to(device).eval()
    if args.fp16:
        model.half()
    print("Model ready.", flush=True)

    # Read list
    with open(args.list_file, "r") as f:
        items = [ln.strip() for ln in f if ln.strip()]
    if args.max_n:
        items = items[:args.max_n]
    print(f"Found {len(items)} frames to process.", flush=True)
    if items[:3]:
        print("Sample items:", *items[:3], sep="\n  ", flush=True)

    processed = 0
    skipped = 0
    err = 0
    t0 = time.time()

    for rel in tqdm(items, desc="CoMotion single-image", mininterval=0.5):
        try:
            img_path = Path(args.in_dir) / rel
            if not img_path.exists():
                err += 1
                if err < 5:
                    print(f"⚠️  missing image: {img_path}", flush=True)
                continue

            out_dir, out_pt = map_rel_to_out(rel, in_prefix="frames/", out_root=args.out_root)
            if out_pt.exists():
                skipped += 1
                continue
            out_dir.mkdir(parents=True, exist_ok=True)

            # --------- Load image ----------
            try:
                image_np = np.array(Image.open(img_path))
            except Exception as e:
                err += 1
                if err < 5:
                    print(f"⚠️  PIL open failed: {img_path} ({e})", flush=True)
                continue

            # --------- Prepare tensors ----------
            image_t = dataloading.convert_image_to_tensor(image_np)  # [C,H,W], float32 in 0..1
            K = dataloading.get_default_K(image_t)
            cropped_image, cropped_K = dataloading.prepare_network_inputs(image_t, K, device)

            # --------- Forward + decode ----------
            if args.fp16:
                with torch.cuda.amp.autocast():
                    det = model.detection_model(cropped_image, cropped_K)
                    det = comotion.detect.decode_network_outputs(
                        K.to(device), model.smpl_decoder, det, std=args.std, conf_thr=args.conf_thr
                    )
            else:
                det = model.detection_model(cropped_image, cropped_K)
                det = comotion.detect.decode_network_outputs(
                    K.to(device), model.smpl_decoder, det, std=args.std, conf_thr=args.conf_thr
                )

            det = {k: v[0].cpu() for k, v in det.items()}
            torch.save(det, out_pt)
            processed += 1

        except KeyboardInterrupt:
            print("\n⛔ Interrupted by user.", flush=True)
            break
        except Exception as e:
            err += 1
            print(f"❌ Error on {rel}: {e}", flush=True)
            if err < 3:
                traceback.print_exc()
            continue

    dt = time.time() - t0
    print(f"Done. processed={processed}, skipped(existed)={skipped}, errors={err}, time={dt:.1f}s", flush=True)

if __name__ == "__main__":
    main()
