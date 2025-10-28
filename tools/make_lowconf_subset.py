#!/usr/bin/env python3
import argparse, os, math
from pathlib import Path
from glob import glob
import numpy as np
import pandas as pd
import torch

# Optional deps for visualization
try:
    import cv2
except Exception:
    cv2 = None
try:
    import imageio.v3 as iio
except Exception:
    iio = None
try:
    from PIL import Image, ImageDraw, ImageFont
except Exception:
    Image = ImageDraw = ImageFont = None


# ---------------- Path resolvers (reused style) ----------------
def find_cropped_mp4s(cropped_root: str | Path, label: str, clip_id: str,
                      preferred_camera: str | None = None) -> dict:
    base = Path(cropped_root) / label
    cams = ("left", "right", "top")
    out = {}
    # common patterns
    for cam in cams:
        p = base / f"{clip_id}_{cam}.mp4"
        if p.exists(): out[cam] = str(p)
    # fallback without suffix
    p0 = base / f"{clip_id}.mp4"
    if p0.exists(): out["nosfx"] = str(p0)
    # primary
    primary = None
    if preferred_camera and preferred_camera in out:
        primary = out[preferred_camera]
    else:
        for k in ("left","right","top","nosfx"):
            if k in out: primary = out[k]; break
    out["primary"] = primary or ""
    return out


# ---------------- Confidence extraction ----------------
def _pick_track(d: dict) -> dict:
    """Select a track dict containing pose/conf; pick the longest by pose length."""
    if isinstance(d, dict) and all(k in d for k in ("pose","trans","betas")):
        return d
    cand = []
    if isinstance(d, (list, tuple)):
        for it in d:
            if isinstance(it, dict) and ("pose" in it or "kp_conf" in it or "keypoints2d" in it or "joints2d" in it):
                cand.append(it)
    elif isinstance(d, dict):
        tracks = d.get("tracks", [])
        if isinstance(tracks, (list, tuple)):
            for it in tracks:
                if isinstance(it, dict) and ("pose" in it or "kp_conf" in it or "keypoints2d" in it or "joints2d" in it):
                    cand.append(it)
    if not cand:
        return d if isinstance(d, dict) else {}
    def _lenpose(x):
        p = x.get("pose", None)
        if isinstance(p, torch.Tensor): return int(p.shape[0])
        if isinstance(p, np.ndarray):  return int(p.shape[0])
        return int(x.get("T", 0))
    cand.sort(key=_lenpose, reverse=True)
    return cand[0]

def _to_numpy(x):
    if x is None: return None
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    if isinstance(x, np.ndarray):
        return x
    return None

def extract_kp_conf(track: dict) -> np.ndarray | None:
    """
    Try multiple common layouts. Return float array [T, J] in [0,1] if found, else None.
    """
    # 1) explicit kp_conf
    for key in ["kp_conf", "keypoints_conf", "joints_conf", "conf"]:
        if key in track:
            arr = _to_numpy(track[key])
            if arr is not None:
                if arr.ndim == 2: return arr.astype(np.float32)
                if arr.ndim == 3 and arr.shape[-1] == 1:  # [T,J,1]
                    return arr[...,0].astype(np.float32)

    # 2) 2D keypoints with confidence in last channel
    for key in ["keypoints2d", "joints2d", "kp2d", "pose2d", "kps2d"]:
        if key in track:
            arr = _to_numpy(track[key])
            if arr is not None and arr.ndim == 3:
                if arr.shape[-1] >= 3:
                    conf = arr[..., 2]
                    return conf.astype(np.float32)

    # 3) sometimes 2D is under a nested dict
    for key in ["kpt2d", "pred_2d"]:
        if key in track and isinstance(track[key], dict):
            sub = track[key]
            for subk in ["conf", "confidence", "kp_conf"]:
                if subk in sub:
                    arr = _to_numpy(sub[subk])
                    if arr is not None:
                        if arr.ndim == 2: return arr.astype(np.float32)
                        if arr.ndim == 3 and arr.shape[-1] == 1:
                            return arr[...,0].astype(np.float32)
            for subk in ["keypoints", "joints"]:
                if subk in sub:
                    arr = _to_numpy(sub[subk])
                    if arr is not None and arr.ndim == 3 and arr.shape[-1] >= 3:
                        return arr[...,2].astype(np.float32)

    return None


# ---------------- Scoring ----------------
def score_clip_from_pt(skel_pt_path: str, conf_thresh: float, low_joints_k: int):
    try:
        d = torch.load(skel_pt_path, map_location="cpu")
    except Exception:
        return dict(frames=0, frames_low=0, occ_pct=0.0, low_j_mean=0.0, low_j_max=0, T_conf=0)

    track = _pick_track(d) if isinstance(d, (dict, list, tuple)) else {}
    conf = extract_kp_conf(track)
    if conf is None or not isinstance(conf, np.ndarray) or conf.size == 0:
        return dict(frames=0, frames_low=0, occ_pct=0.0, low_j_mean=0.0, low_j_max=0, T_conf=0)

    conf = np.clip(conf.astype(np.float32), 0.0, 1.0)
    T, J = conf.shape[0], conf.shape[1]
    low_mask = (conf < conf_thresh).astype(np.int32)   # [T,J]
    low_counts = low_mask.sum(axis=1)                  # [T]
    frames_low = int((low_counts >= int(low_joints_k)).sum())
    occ_pct = float(frames_low / max(T, 1))
    low_j_mean = float(low_counts.mean() / max(J, 1))
    low_j_max = int(low_counts.max(initial=0))
    return dict(frames=T, frames_low=frames_low, occ_pct=occ_pct,
                low_j_mean=low_j_mean, low_j_max=low_j_max, T_conf=T)


# ---------------- Visualization (same style) ----------------
def _available_video_backend():
    if cv2 is not None: return "cv2"
    if iio is not None: return "imageio"
    return None

def sample_video_frames(video_path, num_frames=12, backend=None):
    backend = backend or _available_video_backend()
    frames = []
    if backend == "cv2":
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened(): return frames
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        idxs = np.linspace(0, max(total-1,0), num=min(num_frames, max(total,1)), dtype=int).tolist() if total>0 else list(range(num_frames))
        for idx in idxs:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ok, frame = cap.read()
            if ok and frame is not None:
                frames.append(frame[:,:,::-1])  # BGR->RGB
        cap.release()
    elif backend == "imageio":
        try:
            meta = iio.immeta(video_path, plugin="pyav") if hasattr(iio, "immeta") else {}
            total = int(meta.get("n_frames", 0))
        except Exception:
            total = 0
        if total <= 0:
            try:
                it = iio.imiter(video_path, plugin="pyav")
                buf = []
                for f in it:
                    buf.append(np.asarray(f))
                    if len(buf) >= num_frames * 4: break
                if buf:
                    idxs = np.linspace(0, len(buf)-1, num=min(num_frames, len(buf)), dtype=int)
                    frames = [buf[i] for i in idxs]
            except Exception:
                pass
        else:
            idxs = np.linspace(0, max(total-1,0), num=min(num_frames, total), dtype=int)
            wanted = set(int(i) for i in idxs)
            for i, fr in enumerate(iio.imiter(video_path, plugin="pyav")):
                if i in wanted: frames.append(np.asarray(fr))
                if i > max(wanted, default=-1): break
    if Image is not None and isinstance(Image, type):
        pil_frames = []
        for f in frames:
            try: pil_frames.append(Image.fromarray(f))
            except Exception: pass
        frames = pil_frames if pil_frames else frames
    return frames

def make_contact_sheet(frames, cols=6, tile_h=240, pad=4, bg=(18,18,18)):
    if not frames or (Image is None) or (not isinstance(Image, type)): return None
    pil_frames = [Image.fromarray(f) if isinstance(f, np.ndarray) else f for f in frames]
    resized = []
    for im in pil_frames:
        w,h = im.size
        if h <= 0: continue
        new_w = int(round(w * (tile_h / h)))
        resized.append(im.resize((max(1,new_w), tile_h), Image.BILINEAR))
    if not resized: return None
    rows = int(math.ceil(len(resized)/cols))
    widths = [r.size[0] for r in resized]
    row_w = []
    for r in range(rows):
        seg = widths[r*cols:(r+1)*cols]
        row_w.append(sum(seg) + pad*(min(cols,len(seg))+1))
    canvas_w = max(row_w) if row_w else 0
    canvas_h = rows*tile_h + pad*(rows+1)
    sheet = Image.new("RGB", (canvas_w, canvas_h), bg)
    x=y=pad
    for i, im in enumerate(resized):
        if i>0 and (i%cols==0):
            y += tile_h + pad; x = pad
        sheet.paste(im, (x,y)); x += im.size[0] + pad
    return sheet

def add_header_bar(img, text, bar_h=36, fg=(255,255,255), bg=(32,32,32)):
    if (Image is None) or (ImageDraw is None): return img
    w,h = img.size
    canvas = Image.new("RGB", (w, h+bar_h), bg); canvas.paste(img, (0, bar_h))
    draw = ImageDraw.Draw(canvas)
    try: font = ImageFont.truetype("DejaVuSans.ttf", 18)
    except Exception: font = ImageFont.load_default()
    draw.text((8, (bar_h-18)//2), text, fill=fg, font=font)
    return canvas


# ---------------- Main ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index_csv", required=True, help="CSV with at least 'clip','label' and path to skeleton (.pt)")
    ap.add_argument("--out_dir", required=True, help="Output dir for scores + subset CSVs")
    ap.add_argument("--lowconf_thresh", type=float, default=0.50, help="Keypoint confidence threshold")
    ap.add_argument("--low_joints_k",  type=int,   default=3,     help="# low-conf joints to flag a frame")
    ap.add_argument("--high_pct",      type=float, default=0.20,  help="Top percentage for high-occlusion subset")
    ap.add_argument("--cropped_root",  type=str,   default="/workspace/cropped", help="Root of cropped videos by action folder")
    # Viz
    ap.add_argument("--viz_top_k",   type=int, default=0, help="Generate contact sheets for top-K clips")
    ap.add_argument("--viz_frames",  type=int, default=12)
    ap.add_argument("--viz_cols",    type=int, default=6)
    ap.add_argument("--viz_height",  type=int, default=240)
    args = ap.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.index_csv)
    col_skel = "path_skel" if "path_skel" in df.columns else ("path_skeleton" if "path_skeleton" in df.columns else None)
    if col_skel is None or not {"clip","label",col_skel}.issubset(df.columns):
        raise ValueError(f"index_csv must include 'clip','label' and 'path_skel' (or 'path_skeleton'). Got: {df.columns.tolist()}")

    rows = []
    found_conf = 0
    examples = []
    for _, r in df.iterrows():
        clip = str(r["clip"]); label = str(r["label"])
        skel_pt = str(r[col_skel])
        metrics = score_clip_from_pt(skel_pt, args.lowconf_thresh, args.low_joints_k)
        if metrics["T_conf"] > 0: found_conf += 1

        # try to locate cropped videos for viz
        crops = find_cropped_mp4s(args.cropped_root, label, clip, preferred_camera=None)

        rows.append({
            "clip": clip, "label": label, "path_skel": skel_pt,
            "frames": metrics["frames"], "frames_low": metrics["frames_low"],
            "occ_pct": metrics["occ_pct"], "low_j_mean": metrics["low_j_mean"],
            "low_j_max": metrics["low_j_max"],
            "cropped_primary": crops.get("primary",""),
            "cropped_left": crops.get("left",""),
            "cropped_right": crops.get("right",""),
            "cropped_top": crops.get("top",""),
        })
        if len(examples) < 3 and metrics["T_conf"] > 0:
            examples.append((clip, skel_pt))

    print(f"[info] Found usable per-joint confidence for {found_conf}/{len(df)} clips.")
    if examples:
        print("[info] Examples:")
        for c, p in examples:
            print(f"  {c} -> {p}")

    score_df = pd.DataFrame(rows).sort_values("occ_pct", ascending=False).reset_index(drop=True)

    # Save scores
    score_csv = out_dir / "occlusion_scores_conf.csv"
    score_df.to_csv(score_csv, index=False)
    print(f"[OK] Wrote scores -> {score_csv}")

    # Build subsets (preserve original columns)
    N = len(score_df); k = max(1, int(round(args.high_pct * N)))
    base_df = df.copy()
    merged = score_df.merge(base_df, on=["clip","label"], how="left", suffixes=("", "_orig"))

    high_csv = out_dir / "val_high_occ_conf.csv"
    low_csv  = out_dir / "val_low_occ_conf.csv"
    base_cols = list(base_df.columns)
    merged.iloc[:k][base_cols].to_csv(high_csv, index=False)
    merged.iloc[k:][base_cols].to_csv(low_csv, index=False)
    print(f"[OK] Wrote subsets -> {high_csv} (top {k}/{N}), {low_csv} (rest)")

    # Summary CSV for top-K
    summary_cols = [
        "clip","label","occ_pct","low_j_mean","low_j_max","frames","frames_low",
        "path_skel","cropped_primary","cropped_left","cropped_right","cropped_top"
    ]
    summary_csv = out_dir / "val_high_occ_conf_summary.csv"
    score_df.iloc[:k][summary_cols].to_csv(summary_csv, index=False)
    print(f"[OK] Wrote high-occlusion summary -> {summary_csv}")

    # Viz
    if args.viz_top_k > 0:
        backend = _available_video_backend()
        if backend is None or (Image is None) or (not isinstance(Image, type)):
            print("[viz] Missing cv2/imageio or PIL; skipping contact sheets.")
        else:
            viz_dir = out_dir / "viz_topk_conf"; viz_dir.mkdir(parents=True, exist_ok=True)
            topK = min(args.viz_top_k, len(score_df))
            print(f"[viz] Generating contact sheets for top-{topK} clips...")
            for _, row in score_df.iloc[:topK].iterrows():
                pth = str(row.get("cropped_primary","") or "")
                if not pth or not Path(pth).exists(): continue
                head = f"{row['clip']} | {row['label']} | occ_pct={float(row['occ_pct']):.2f} lowJmean={float(row['low_j_mean']):.2f} lowJmax={int(row['low_j_max'])}"
                frames = sample_video_frames(pth, num_frames=args.viz_frames, backend=backend)
                sheet = make_contact_sheet(frames, cols=args.viz_cols, tile_h=args.viz_height)
                if sheet is not None:
                    sheet = add_header_bar(sheet, head)
                    outp = viz_dir / f"{row['clip']}.png"
                    sheet.save(outp)
                    print(f"[viz] wrote {outp}")
            print(f"[viz] Done. See {viz_dir}")


if __name__ == "__main__":
    main()
