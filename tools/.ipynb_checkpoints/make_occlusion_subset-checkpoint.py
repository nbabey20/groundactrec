#!/usr/bin/env python3
import argparse, os, math
from pathlib import Path
from glob import glob
import numpy as np
import pandas as pd

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


# ---------------- Path resolvers ----------------
def find_mot_txt(mot_root: str | Path, label: str, clip_id: str,
                 camera_order=("left", "right", "top")) -> str | None:
    """
    Return the best-matching MOT .txt, preferring left->right->top.
    Expected layout: <mot_root>/<label>/<clip_id>_{left|right|top}.txt
    """
    base = Path(mot_root) / label
    # exact matches first
    for cam in camera_order:
        p = base / f"{clip_id}_{cam}.txt"
        if p.exists():
            return str(p)
    # fallback: any view with suffix
    m = sorted(glob(str(base / f"{clip_id}_*.txt")))
    if m:
        return m[0]
    # last resort: clip without suffix
    p = base / f"{clip_id}.txt"
    return str(p) if p.exists() else None


def find_cropped_mp4s(cropped_root: str | Path, label: str, clip_id: str,
                      preferred_camera: str | None) -> dict:
    """
    Return dict of available cropped mp4s by camera.
    Always tries the preferred camera first when selecting a 'primary'.
    """
    base = Path(cropped_root) / label
    cams = ("left", "right", "top")
    out = {}
    for cam in cams:
        p = base / f"{clip_id}_{cam}.mp4"
        if p.exists():
            out[cam] = str(p)

    # Also provide a recommended/primary path
    primary = None
    if preferred_camera and preferred_camera in out:
        primary = out[preferred_camera]
    else:
        for cam in cams:
            if cam in out:
                primary = out[cam]
                break
    out["primary"] = primary or ""
    return out


# ---------------- IoU ----------------
def iou_xyxy(a, b):
    # a,b: [x1,y1,x2,y2]
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    iw = max(0.0, x2 - x1)
    ih = max(0.0, y2 - y1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, (a[2] - a[0])) * max(0.0, (a[3] - a[1]))
    area_b = max(0.0, (b[2] - b[0])) * max(0.0, (b[3] - b[1]))
    denom = area_a + area_b - inter
    return float(inter / denom) if denom > 0 else 0.0


def max_pairwise_iou_xyxy(boxes):
    n = len(boxes)
    if n < 2:
        return 0.0
    m = 0.0
    for i in range(n):
        Ai = boxes[i]
        for j in range(i + 1, n):
            Bij = boxes[j]
            m = max(m, iou_xyxy(Ai, Bij))
            if m >= 0.999:
                return 1.0
    return m


# -------------- MOT parser --------------
def load_mot_txt(mot_path):
    """Returns dict[int_frame] -> list of boxes [x1,y1,x2,y2]"""
    frame_to_boxes = {}
    with open(mot_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(",")
            if len(parts) < 6:
                continue
            try:
                fr = int(float(parts[0]))
                x = float(parts[2]); y = float(parts[3])
                w = float(parts[4]); h = float(parts[5])
            except Exception:
                continue
            if not math.isfinite(x) or not math.isfinite(y) or not math.isfinite(w) or not math.isfinite(h):
                continue
            if w <= 0 or h <= 0:
                continue
            box = [x, y, x + w, y + h]
            frame_to_boxes.setdefault(fr, []).append(box)
    return frame_to_boxes


def clip_occlusion_scores(mot_path, thresh=0.30):
    ftb = load_mot_txt(mot_path)
    if not ftb:
        return dict(frames=0, frames_occ=0, occ_mean=0.0, occ_max=0.0, occ_pct=0.0)
    frames = sorted(ftb.keys())
    scores, occ_frames = [], 0
    for fr in frames:
        s = max_pairwise_iou_xyxy(ftb[fr])
        scores.append(s)
        if s >= thresh:
            occ_frames += 1
    scores = np.array(scores, dtype=float)
    return dict(
        frames=len(frames),
        frames_occ=int(occ_frames),
        occ_mean=float(scores.mean()) if len(scores) else 0.0,
        occ_max=float(scores.max()) if len(scores) else 0.0,
        occ_pct=float(occ_frames / max(len(frames), 1)),
    )


# ---------------- Visualization helpers ----------------
def _available_video_backend():
    if cv2 is not None:
        return "cv2"
    if iio is not None:
        return "imageio"
    return None


def sample_video_frames(video_path, num_frames=12, backend=None):
    """
    Returns list of RGB PIL Images (or numpy arrays if PIL unavailable).
    Evenly samples frames across the video length.
    """
    backend = backend or _available_video_backend()
    frames = []

    if backend == "cv2":
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return frames
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0:
            idxs = list(range(num_frames))
        else:
            idxs = np.linspace(0, max(total - 1, 0), num=min(num_frames, max(total, 1)), dtype=int).tolist()

        for idx in idxs:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            frame = frame[:, :, ::-1]  # BGR -> RGB
            frames.append(frame)
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
                    if len(buf) >= num_frames * 4:
                        break
                if buf:
                    idxs = np.linspace(0, len(buf) - 1, num=min(num_frames, len(buf)), dtype=int)
                    frames = [buf[i] for i in idxs]
            except Exception:
                pass
        else:
            idxs = np.linspace(0, max(total - 1, 0), num=min(num_frames, total), dtype=int)
            wanted = set(int(i) for i in idxs)
            for i, fr in enumerate(iio.imiter(video_path, plugin="pyav")):
                if i in wanted:
                    frames.append(np.asarray(fr))
                if i > max(wanted, default=-1):
                    break
    else:
        return frames

    # Convert to PIL if available
    if Image is not None:
        pil_frames = []
        for f in frames:
            try:
                pil_frames.append(Image.fromarray(f))
            except Exception:
                pass
        frames = pil_frames if pil_frames else frames

    return frames


def make_contact_sheet(frames, cols=6, tile_h=240, pad=4, bg=(18, 18, 18)):
    """
    frames: list of PIL Images or np arrays RGB
    Returns a PIL Image (or None on failure).
    """
    if not frames:
        return None
    if Image is None:
        return None
    pil_frames = []
    for im in frames:
        if isinstance(im, np.ndarray):
            im = Image.fromarray(im)
        pil_frames.append(im)

    # Resize to fixed height, keep aspect
    resized = []
    for im in pil_frames:
        w, h = im.size
        if h == 0:
            continue
        new_w = int(round(w * (tile_h / h)))
        resized.append(im.resize((max(1, new_w), tile_h), Image.BILINEAR))

    if not resized:
        return None

    rows = int(math.ceil(len(resized) / cols))
    widths = [r.size[0] for r in resized]
    row_w = []
    for r in range(rows):
        row_w.append(sum(widths[r * cols:(r + 1) * cols]) + pad * (min(cols, len(widths[r * cols:(r + 1) * cols])) + 1))
    canvas_w = max(row_w) if row_w else 0
    canvas_h = rows * tile_h + pad * (rows + 1)

    sheet = Image.new("RGB", (canvas_w, canvas_h), bg)
    x = y = pad
    for i, im in enumerate(resized):
        if i > 0 and (i % cols == 0):
            y += tile_h + pad
            x = pad
        sheet.paste(im, (x, y))
        x += im.size[0] + pad
    return sheet


def add_header_bar(img, text, bar_h=36, fg=(255, 255, 255), bg=(32, 32, 32)):
    if (Image is None) or (ImageDraw is None):
        return img
    w, h = img.size
    canvas = Image.new("RGB", (w, h + bar_h), bg)
    canvas.paste(img, (0, bar_h))
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 18)
    except Exception:
        font = ImageFont.load_default()
    draw.text((8, (bar_h - 18) // 2), text, fill=fg, font=font)
    return canvas


def save_contact_sheets_for_clip(row, viz_dir, viz_frames=12, viz_cols=6, viz_height=240):
    """
    Makes a contact sheet for each available camera (left/right/top).
    """
    if Image is None:
        print("[viz] PIL not available; skipping contact sheets.")
        return

    clip = row["clip"]
    label = row["label"]
    occ_pct = float(row.get("occ_pct", 0.0))
    occ_mean = float(row.get("occ_mean", 0.0))
    occ_max = float(row.get("occ_max", 0.0))
    head = f"{clip} | {label} | occ_pct={occ_pct:.2f}  occ_mean={occ_mean:.2f}  occ_max={occ_max:.2f}"

    # Gather available camera videos from row
    candidates = []
    for cam in ("left", "right", "top"):
        path_key = f"cropped_{cam}"
        p = str(row.get(path_key, "") or "")
        if p:
            candidates.append((cam, Path(p)))

    backend = _available_video_backend()
    if backend is None:
        print("[viz] No video backend (cv2/imageio) available; skipping.")
        return

    viz_dir.mkdir(parents=True, exist_ok=True)

    for side_tag, pth in candidates:
        if pth.exists():
            frames = sample_video_frames(pth, num_frames=viz_frames, backend=backend)
            sheet = make_contact_sheet(frames, cols=viz_cols, tile_h=viz_height)
            if sheet is not None:
                sheet = add_header_bar(sheet, f"{head} | {side_tag}")
                outp = viz_dir / f"{clip}_{side_tag}.png"
                sheet.save(outp)
                print(f"[viz] wrote {outp}")


# -------------- Main driver --------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index_csv", required=True, help="Validation (or full) index CSV used by your trainers (must have 'clip' and 'label')")
    ap.add_argument("--mot_root", required=True, help="Directory containing MOT .txt files from CoMotion")
    ap.add_argument("--thresh", type=float, default=0.30, help="IoU threshold classifying a frame as occluded")
    ap.add_argument("--high_pct", type=float, default=0.20, help="Top percentage (0..1) to mark as high-occlusion")
    ap.add_argument("--out_dir", type=str, required=True, help="Where to write outputs (scores + subset CSVs)")
    ap.add_argument("--cropped_root", type=str, default="workspace/cropped", help="Root of cropped videos by action folder")

    # Visualization flags
    ap.add_argument("--viz_top_k", type=int, default=0, help="If >0, save contact-sheet PNGs for the top-K occluded clips")
    ap.add_argument("--viz_frames", type=int, default=12, help="Number of frames per contact sheet")
    ap.add_argument("--viz_cols", type=int, default=6, help="Columns in the contact sheet grid")
    ap.add_argument("--viz_height", type=int, default=240, help="Tile height for each frame (keeps aspect)")

    args = ap.parse_args()

    # Normalize roots to absolute paths
    mot_root = Path(args.mot_root).resolve()
    cropped_root = Path(args.cropped_root).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.index_csv)
    if not {"clip", "label"}.issubset(df.columns):
        raise ValueError("index_csv must include 'clip' and 'label' columns")

    rows = []
    found_mot = 0
    examples = []

    for _, r in df.iterrows():
        clip_id = str(r["clip"])
        label = str(r["label"])

        mot_path_str = find_mot_txt(mot_root, label, clip_id)
        mot_path = Path(mot_path_str) if mot_path_str else None

        if mot_path and mot_path.exists():
            found_mot += 1
            # parse camera from filename suffix (e.g., ..._left.txt)
            stem = mot_path.stem
            parts = stem.split("_")
            camera = parts[-1] if parts and parts[-1] in {"left", "right", "top"} else ""
            # get cropped videos, preferring same camera
            crops = find_cropped_mp4s(cropped_root, label, clip_id, camera)
            metrics = clip_occlusion_scores(str(mot_path), thresh=args.thresh)

            rows.append({
                "clip": clip_id,
                "label": label,
                "camera": camera,
                "mot_path": str(mot_path),
                **metrics,
                "cropped_left": crops.get("left", ""),
                "cropped_right": crops.get("right", ""),
                "cropped_top": crops.get("top", ""),
                "cropped_primary": crops.get("primary", ""),
            })
            if len(examples) < 3:
                examples.append((clip_id, str(mot_path)))
        else:
            # no mot; still record row
            rows.append({
                "clip": clip_id, "label": label,
                "camera": "", "mot_path": "",
                "frames": 0, "frames_occ": 0,
                "occ_mean": 0.0, "occ_max": 0.0, "occ_pct": 0.0,
                "cropped_left": "", "cropped_right": "", "cropped_top": "", "cropped_primary": "",
            })

    print(f"[info] Resolved MOT files for {found_mot}/{len(df)} clips.")
    if examples:
        print("[info] Examples:")
        for c, p in examples:
            print(f"  {c} -> {p}")

    score_df = pd.DataFrame(rows)
    score_df["rank_key"] = score_df["occ_pct"]  # robust; switch to occ_mean/occ_max if desired
    score_df = score_df.sort_values("rank_key", ascending=False).reset_index(drop=True)

    # Save all scores
    score_csv = out_dir / "occlusion_scores.csv"
    score_df.to_csv(score_csv, index=False)
    print(f"[OK] Wrote scores -> {score_csv}")

    # Build high/low subsets
    N = len(score_df)
    k = max(1, int(round(args.high_pct * N)))
    base_df = pd.read_csv(args.index_csv)

    merged = score_df.merge(base_df, on=["clip", "label"], how="left", suffixes=("", "_orig"))
    high_csv = out_dir / "val_high_occ.csv"
    low_csv = out_dir / "val_low_occ.csv"

    base_cols = list(base_df.columns)
    merged.iloc[:k][base_cols].to_csv(high_csv, index=False)
    merged.iloc[k:][base_cols].to_csv(low_csv, index=False)
    print(f"[OK] Wrote subsets -> {high_csv} (top {k}/{N}), {low_csv} (rest)")

    # Summary of top-K by occlusion
    summary_csv = out_dir / "val_high_occ_summary.csv"
    high_summary_cols = [
        "clip", "label", "camera", "occ_pct", "occ_mean", "occ_max",
        "frames", "frames_occ", "mot_path",
        "cropped_primary", "cropped_left", "cropped_right", "cropped_top"
    ]
    score_df.iloc[:k][high_summary_cols].to_csv(summary_csv, index=False)
    print(f"[OK] Wrote high-occlusion summary -> {summary_csv}")

    # ---- Visualization for top-K ----
    if args.viz_top_k > 0:
        viz_dir = out_dir / "viz_topk"
        topK = min(args.viz_top_k, len(score_df))
        print(f"[viz] Generating contact sheets for top-{topK} clips...")
        for _, row in score_df.iloc[:topK].iterrows():
            save_contact_sheets_for_clip(
                row, viz_dir,
                viz_frames=args.viz_frames,
                viz_cols=args.viz_cols,
                viz_height=args.viz_height
            )
        print(f"[viz] Done. See {viz_dir}")


if __name__ == "__main__":
    main()
