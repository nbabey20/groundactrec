#!/usr/bin/env python3
import argparse, os, math, json
from pathlib import Path
from glob import glob
import numpy as np
import pandas as pd

# optional deps
try:
    import torch
except Exception:
    torch = None

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
    
# --- label normalization (strip leading [CODE] ) ---
import re
def normalize_label(lbl: str) -> str:
    s = str(lbl).strip()
    # remove things like "[OP012] " at the start
    s = re.sub(r'^\s*\[[^\]]+\]\s*', '', s)
    # collapse spaces
    s = re.sub(r'\s+', ' ', s).strip()
    return s

# ---------------- utils ----------------
def robust_z(x):
    x = np.asarray(x, dtype=float)
    if x.size == 0: return np.zeros(0, dtype=float)
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    if mad <= 1e-12:
        return np.zeros_like(x)
    return 0.67448975 * (x - med) / mad  # ~N(0,1) if Laplace-ish

def exists(p): return Path(p).exists()

# ---------------- MOT helpers ----------------
def load_mot_txt(path):
    """
    Parse MOT-like txt:
    frame,id,x,y,w,h,score,x3d,y3d,z3d
    Returns sorted frames and dict per-frame.
    """
    frames = []
    rows = {}
    with open(path, "r") as f:
        for ln in f:
            ln = ln.strip()
            if not ln or ln.startswith("#"): continue
            parts = ln.split(",")
            if len(parts) < 7:  # need at least score
                continue
            try:
                fr = int(float(parts[0]))
                x  = float(parts[2]); y   = float(parts[3])
                w  = float(parts[4]); h   = float(parts[5])
                sc = float(parts[6])
            except Exception:
                continue
            frames.append(fr)
            rows[fr] = dict(x=x, y=y, w=w, h=h, score=sc)
    frames = sorted(set(frames))
    return frames, rows

def bbox_proxy_signals(frames, rows):
    """
    Return per-frame bbox instability score using:
      - delta log-area (robust-z)
      - delta aspect ratio (robust-z)
      - center jitter (robust-z on scale-normalized dx,dy)
      - low detector score relative to its median (if it varies)
    """
    if not frames:
        zeros = np.zeros(0)
        return zeros, dict(area=zeros, aspect=zeros, score=zeros, center=zeros)

    T = len(frames)
    A = np.zeros(T); R = np.zeros(T); S = np.zeros(T)
    Cx = np.zeros(T); Cy = np.zeros(T)

    for i, fr in enumerate(frames):
        r = rows.get(fr)
        if r is None:
            continue
        w = max(1e-6, r["w"]); h = max(1e-6, r["h"])
        x = r["x"]; y = r["y"]
        A[i]  = w * h
        R[i]  = w / h
        S[i]  = r.get("score", 1.0)
        Cx[i] = x + 0.5 * w
        Cy[i] = y + 0.5 * h

    # deltas
    dlogA = np.diff(np.log(np.maximum(A, 1e-6)), prepend=np.log(max(A[0], 1e-6)))
    dR    = np.diff(R, prepend=R[0])

    # center motion, scale-normalized by sqrt(area) for invariance
    dx = np.diff(Cx, prepend=Cx[0])
    dy = np.diff(Cy, prepend=Cy[0])
    scale = np.sqrt(np.maximum(A, 1e-6))
    nx = dx / np.maximum(scale, 1e-6)
    ny = dy / np.maximum(scale, 1e-6)
    ncent = np.hypot(nx, ny)  # per-frame normalized move

    zA    = np.abs(robust_z(dlogA))
    zR    = np.abs(robust_z(dR))
    zCent = np.abs(robust_z(ncent))

    # detector score: penalize when low relative to clip median
    Srel = 1.0 - (S / (np.median(S) + 1e-6))
    Srel = np.clip(Srel, 0.0, 1.0)  # often all zeros if score is constant

    # combine (slightly upweight center jitter)
    bbox_instability = 0.4 * zA + 0.3 * zR + 0.5 * zCent + 0.3 * Srel
    return bbox_instability, dict(area=A, aspect=R, score=S, center=zCent)

# ---------------- SMPL decoding & proxies ----------------
SMPL_PARENTS_24 = np.array([
    -1, 0, 0, 0, 1, 2, 3, 4, 5, 6,
     7, 8, 9,12,12,12,13,14,16,17,18,19,20,21
], dtype=int)  # a common SMPL kinematic tree (24 joints)

def _pick_track_struct(d):
    # matches your FusionIndexDataset logic
    if isinstance(d, dict) and all(k in d for k in ("pose","trans","betas")):
        return d
    cand = []
    if isinstance(d, (list, tuple)):
        for it in d:
            if isinstance(it, dict) and "pose" in it:
                cand.append(it)
    elif isinstance(d, dict):
        tracks = d.get("tracks", [])
        if isinstance(tracks, (list, tuple)):
            for it in tracks:
                if isinstance(it, dict) and "pose" in it:
                    cand.append(it)
    if not cand:
        raise RuntimeError("Unexpected skeleton .pt structure.")
    cand.sort(key=lambda x: int(x["pose"].shape[0] if hasattr(x["pose"], "shape") else len(x["pose"])), reverse=True)
    return cand[0]

def decode_smpl_to_joints(track, smpl_model_dir, center_root=True):
    import smplx
    pose  = torch.as_tensor(track.get("pose"),  dtype=torch.float32)
    trans = torch.as_tensor(track.get("trans", torch.zeros((pose.shape[0],3))))
    betas = torch.as_tensor(track.get("betas", torch.zeros((pose.shape[0],10))))
    T = pose.shape[0]
    if betas.ndim == 1: betas = betas.unsqueeze(0).expand(T, -1)
    if trans.ndim == 1: trans = trans.unsqueeze(0).expand(T, -1)

    layer = smplx.create(model_path=str(Path(smpl_model_dir)), model_type='smpl',
                         gender='neutral', num_betas=10, use_pca=False, ext='pkl').eval()
    # chunked forward
    outJ = []
    BS = 512
    with torch.no_grad():
        for s in range(0, T, BS):
            e = min(T, s + BS)
            out = layer(betas=betas[s:e],
                        global_orient=pose[s:e, :3].float(),
                        body_pose=pose[s:e, 3:72].float(),
                        transl=trans[s:e].float())
            outJ.append(out.joints)  # [b, J, 3]
    J = torch.cat(outJ, dim=0)  # [T, J, 3]
    if center_root:
        J = J - J[:, [0], :]
    return J.cpu().numpy()  # [T,J,3]

def joints_jerk_signal(J):  # J: [T,J,3] numpy
    if J.size == 0: return np.zeros(0)
    # finite-diff along time
    V = np.diff(J, axis=0, prepend=J[[0]])
    A = np.diff(V, axis=0, prepend=V[[0]])
    K = np.diff(A, axis=0, prepend=A[[0]])
    # magnitude per frame (mean across joints)
    mag = np.linalg.norm(K, axis=2)  # [T,J]
    per_frame = np.mean(mag, axis=1)  # [T]
    # robust z
    return np.abs(robust_z(per_frame))

def bone_length_error(J):
    if J.size == 0: return np.zeros(0)
    parents = SMPL_PARENTS_24
    T, Jn, _ = J.shape
    # if SMPL outputs >24 joints, take first 24
    if Jn > len(parents):
        J = J[:, :len(parents), :]
        Jn = len(parents)
    # compute per-bone lengths
    L = []
    for c in range(Jn):
        p = parents[c]
        if p < 0:  # root
            L.append(np.zeros(T))
            continue
        vec = J[:, c, :] - J[:, p, :]
        L.append(np.linalg.norm(vec, axis=1))
    L = np.stack(L, axis=1)  # [T, J]
    med = np.median(L, axis=0, keepdims=True) + 1e-6
    rel_err = np.abs(L - med) / med  # [T,J]
    # aggregate per-frame
    per_frame = np.mean(rel_err, axis=1)  # mean relative error across bones
    return np.abs(robust_z(per_frame))

def skeleton_proxy_signals(skel_pt_path, smpl_model_dir):
    if (torch is None) or (not exists(skel_pt_path)):
        return np.zeros(0), np.zeros(0)
    d = torch.load(skel_pt_path, map_location="cpu")
    track = _pick_track_struct(d)
    J = decode_smpl_to_joints(track, smpl_model_dir=smpl_model_dir, center_root=True)  # [T,J,3]
    jerk = joints_jerk_signal(J)        # robust-z jerk
    blen = bone_length_error(J)         # robust-z bone-length error
    return jerk, blen

# ---------------- video helpers (size, truncation) ----------------
def get_video_size(video_path):
    """Return (W,H) quickly using cv2 if possible, else imageio fallback."""
    if not video_path or not Path(video_path).exists():
        return None, None
    if cv2 is not None:
        cap = cv2.VideoCapture(str(video_path))
        if cap.isOpened():
            W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()
            if W > 0 and H > 0:
                return W, H
    if iio is not None:
        try:
            meta = iio.immeta(video_path, plugin="pyav") if hasattr(iio, "immeta") else {}
            W = int(meta.get("width", 0)); H = int(meta.get("height", 0))
            if W > 0 and H > 0:
                return W, H
        except Exception:
            pass
    return None, None

def truncation_flags(frames, rows, W, H, eps=2.0):
    """Boolean per frame: bbox touches any image border within eps pixels."""
    if not frames or W is None or H is None:
        return np.zeros(0, dtype=bool)
    out = np.zeros(len(frames), dtype=bool)
    for i, fr in enumerate(frames):
        r = rows.get(fr)
        if r is None:
            continue
        x,y,w,h = r["x"], r["y"], r["w"], r["h"]
        touch = (x <= eps) or (y <= eps) or (x + w >= W - eps) or (y + h >= H - eps)
        out[i] = touch
    return out

# ---------------- crop video viz (same as your older script) ----------------
def _available_video_backend():
    if cv2 is not None:
        return "cv2"
    if iio is not None:
        return "imageio"
    return None

def sample_video_frames(video_path, num_frames=12, backend=None):
    backend = backend or _available_video_backend()
    frames = []
    if backend == "cv2":
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return frames
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        idxs = np.linspace(0, max(total - 1, 0), num=min(num_frames, max(total, 1)), dtype=int).tolist()
        for idx in idxs:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            frame = frame[:, :, ::-1]
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
    # to PIL
    if Image is not None and isinstance(Image, type):
        out = []
        for f in frames:
            try: out.append(Image.fromarray(f))
            except Exception: pass
        frames = out if out else frames
    return frames

def make_contact_sheet(frames, cols=6, tile_h=240, pad=4, bg=(18,18,18)):
    if not frames or (Image is None) or (not isinstance(Image, type)):
        return None
    pil_frames = [Image.fromarray(f) if not isinstance(f, Image.Image) else f for f in frames]
    resized = []
    for im in pil_frames:
        w,h = im.size
        new_w = int(round(w * (tile_h / max(h,1))))
        resized.append(im.resize((max(1,new_w), tile_h), Image.BILINEAR))
    rows = int(math.ceil(len(resized) / cols))
    widths = [r.size[0] for r in resized]
    row_w = []
    for r in range(rows):
        seg = widths[r*cols:(r+1)*cols]
        row_w.append(sum(seg) + pad * (min(cols, len(seg)) + 1))
    canvas_w = max(row_w) if row_w else 0
    canvas_h = rows * tile_h + pad * (rows + 1)
    sheet = Image.new("RGB", (canvas_w, canvas_h), bg)
    x = y = pad
    for i, im in enumerate(resized):
        if i>0 and (i % cols == 0):
            y += tile_h + pad
            x = pad
        sheet.paste(im, (x,y))
        x += im.size[0] + pad
    return sheet

def add_header_bar(img, text, bar_h=36, fg=(255,255,255), bg=(32,32,32)):
    if (Image is None) or (ImageDraw is None): return img
    w,h = img.size
    canvas = Image.new("RGB", (w, h+bar_h), bg)
    canvas.paste(img, (0,bar_h))
    draw = ImageDraw.Draw(canvas)
    try: font = ImageFont.truetype("DejaVuSans.ttf", 18)
    except Exception: font = ImageFont.load_default()
    draw.text((8, (bar_h-18)//2), text, fill=fg, font=font)
    return canvas

# ---------------- main scoring ----------------
def combine_frame_scores(jerk_z, blen_z, bbox_instab, w_jerk=1.0, w_blen=1.0, w_bbox=1.0):
    # align lengths
    T = max(len(jerk_z), len(blen_z), len(bbox_instab))
    def pad(a):
        a = np.asarray(a, dtype=float)
        if len(a) == T: return a
        if len(a) == 0: return np.zeros(T)
        return np.pad(a, (0, T-len(a)), mode="edge")
    j = pad(jerk_z)
    b = pad(blen_z)
    x = pad(bbox_instab)
    # weighted sum (all are >=0 robust z's), then robust-z again to normalize scale
    raw = w_jerk*j + w_blen*b + w_bbox*x
    # no second z if all zeros
    if np.allclose(raw, 0):
        return raw
    return np.maximum(0.0, robust_z(raw))  # non-negative normalized difficulty

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index_csv", required=True,
                    help="CSV with at least clip,label. If it also has path_skel/path_skeleton, we'll use skeleton proxies.")
    ap.add_argument("--mot_root", required=True, help="Directory with MOT .txt files in <label>/<clip>_{cam}.txt")
    ap.add_argument("--cropped_root", type=str, default="/workspace/cropped", help="Root for <label>/<clip>_{cam}.mp4 (for viz)")
    ap.add_argument("--out_dir", required=True)
    # SMPL decoding (optional)
    ap.add_argument("--decode_smpl", action="store_true", help="Enable skeleton-based proxies by decoding joints")
    ap.add_argument("--smpl_model_dir", type=str, default=None)
    # subset & viz
    ap.add_argument("--high_pct", type=float, default=0.20)
    ap.add_argument("--viz_top_k", type=int, default=24)
    ap.add_argument("--viz_frames", type=int, default=12)
    ap.add_argument("--viz_cols",   type=int, default=6)
    ap.add_argument("--viz_height", type=int, default=240)
    # thresholds and weights
    ap.add_argument("--hard_z", type=float, default=1.0, help="Z-threshold for a frame to count as 'hard'.")
    ap.add_argument("--w_jerk", type=float, default=1.0, help="Weight for SMPL jerk proxy.")
    ap.add_argument("--w_blen", type=float, default=1.0, help="Weight for SMPL bone-length proxy.")
    ap.add_argument("--w_bbox", type=float, default=1.0, help="Weight for bbox instability proxy.")
    args = ap.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    mot_root = Path(args.mot_root)
    crop_root = Path(args.cropped_root)

        # ...everything above here stays the same ...

    df = pd.read_csv(args.index_csv)
    col_skel = "path_skel" if "path_skel" in df.columns else ("path_skeleton" if "path_skeleton" in df.columns else None)
    have_skeletons = args.decode_smpl and (col_skel is not None)

    rows = []
    resolved = 0

    # counters for sanity
    vid_found = 0
    skel_ok = 0
    skel_fail = 0

    # (Keep all the code before this loop the same)

    for _, r in df.iterrows():
        clip = str(r["clip"])
        label = str(r["label"])

        # --- init signals (ALWAYS define these) ---
        jerk_z = np.zeros(0)
        blen_z = np.zeros(0)
        bbox_instab = np.zeros(0)
        frames = []
        trunc_frac = 0.0
        mot_path = None
        vid_path = None
        camera = ""
        label_fs_used = normalize_label(label) # Default label for output paths

        # ===== Find MOT file using a more robust glob search =====
        search_pattern = mot_root / "**" / f"{clip}_*.txt"
        found_files = sorted(glob(str(search_pattern), recursive=True))

        if found_files:
            # Prefer left -> right -> top if multiple camera views exist
            preferred_order = {"left": 0, "right": 1, "top": 2}
            found_files.sort(key=lambda p: preferred_order.get(Path(p).stem.split("_")[-1], 99))
        
            mot_path = Path(found_files[0])
            camera = mot_path.stem.split("_")[-1]
            label_fs_used = mot_path.parent.name # Use the actual directory name found
            resolved += 1

    # ===== Find the corresponding cropped video =====
        if camera: # Only search for a video if we found a MOT file and know the camera
            vid_search_pattern = crop_root / "**" / f"{clip}_{camera}.mp4"
            found_videos = glob(str(vid_search_pattern), recursive=True)
            if found_videos:
                vid_path = Path(found_videos[0])
                vid_found += 1
                # Update label_fs_used to the video's parent dir if it's different
                label_fs_used = vid_path.parent.name

    # ===== Read MOT + compute bbox/truncation signals =====
        if mot_path:
            frames, mot_rows = load_mot_txt(mot_path)
            bbox_instab, _ = bbox_proxy_signals(frames, mot_rows)

            # Truncation fraction (bbox touches image border)
            W = H = None
            if vid_path:
                W, H = get_video_size(str(vid_path))
            tflags = truncation_flags(frames, mot_rows, W, H, eps=2.0)
            trunc_frac = float(np.mean(tflags)) if len(tflags) else 0.0

    # ===== SMPL proxies (jerk/bone-length) with visible warnings =====
        if have_skeletons:
            skel_path = str(r[col_skel])
            if not Path(skel_path).exists():
                print(f"[warn] missing skeleton: {skel_path}")
            else:
                try:
                    jz, bz = skeleton_proxy_signals(skel_path, smpl_model_dir=args.smpl_model_dir)
                    jerk_z, blen_z = jz, bz
                    skel_ok += 1
                except Exception as e:
                    print(f"[warn] SMPL decode failed for {clip} ({label}) -> {e}")
                    skel_fail += 1

    # ===== Combine per-frame scores and aggregate to clip-level =====
        combo = combine_frame_scores(jerk_z, blen_z, bbox_instab, args.w_jerk, args.w_blen, args.w_bbox)
        T = int(max(len(jerk_z), len(blen_z), len(bbox_instab), len(frames)))
        hard = (combo > args.hard_z).astype(float)
        diff_pct = float(hard.mean()) if T > 0 else 0.0

        rows.append({
            "clip": clip, "label": label, "camera": camera,
            "mot_path": str(mot_path) if mot_path else "",
            "frames": T,
            "proxy_mean": float(np.mean(combo)) if T > 0 else 0.0,
            "proxy_max": float(np.max(combo)) if T > 0 else 0.0,
            "diff_pct": diff_pct,
            "trunc_frac": trunc_frac,
            "jerk_mean": float(np.mean(jerk_z)) if len(jerk_z) > 0 else 0.0,
            "blen_mean": float(np.mean(blen_z)) if len(blen_z) > 0 else 0.0,
            "bbox_mean": float(np.mean(bbox_instab)) if len(bbox_instab) > 0 else 0.0,
            # Construct viz paths based on the actual directory found
            "cropped_left": str(crop_root / label_fs_used / f"{clip}_left.mp4"),
            "cropped_right": str(crop_root / label_fs_used / f"{clip}_right.mp4"),
            "cropped_top": str(crop_root / label_fs_used / f"{clip}_top.mp4"),
        })

    # (The rest of the script from the 'print(f"[info]..." line onwards stays the same)


    # NEW: print what we actually found/decoded
    print(f"[info] Resolved MOT files for {resolved}/{len(df)} clips.")
    print(f"[info] Videos found {vid_found} | SMPL decoded {skel_ok} (fail {skel_fail})")

    score_df = pd.DataFrame(rows)
    # blended ranking: % hard frames + bonus for truncation rate
    score_df["rank_key"] = score_df["diff_pct"] + 0.5 * score_df.get("trunc_frac", 0.0)
    score_df = score_df.sort_values("rank_key", ascending=False).reset_index(drop=True)

    score_csv = out_dir / "occlusion_proxy_scores.csv"
    score_df.to_csv(score_csv, index=False)
    print(f"[OK] Wrote scores -> {score_csv}")

    # subsets (keep original index columns if present)
    N = len(score_df); k = max(1, int(round(args.high_pct * N)))
    base_cols = [c for c in df.columns]
    merged = score_df.merge(df, on=["clip","label"], how="left", suffixes=("", "_orig"))

    high_csv = out_dir / "val_high_occ.csv"
    low_csv  = out_dir / "val_low_occ.csv"
    merged.iloc[:k][base_cols].to_csv(high_csv, index=False)
    merged.iloc[k:][base_cols].to_csv(low_csv,  index=False)
    print(f"[OK] Wrote subsets -> {high_csv} (top {k}/{N}), {low_csv} (rest)")

    # summary for analysis
    summary_cols = ["clip","label","camera","frames","diff_pct","trunc_frac","proxy_mean","proxy_max",
                    "jerk_mean","blen_mean","bbox_mean","mot_path",
                    "cropped_left","cropped_right","cropped_top"]
    summary_csv = out_dir / "val_high_occ_summary.csv"
    score_df.iloc[:k][summary_cols].to_csv(summary_csv, index=False)
    print(f"[OK] Wrote high-occlusion summary -> {summary_csv}")

    # contact sheets (optional)
    if args.viz_top_k > 0:
        viz_dir = out_dir / "viz_topk"; viz_dir.mkdir(parents=True, exist_ok=True)
        backend = _available_video_backend()
        print(f"[viz] Generating contact sheets for top-{min(args.viz_top_k, len(score_df))} clips...")
        for _, row in score_df.iloc[:args.viz_top_k].iterrows():
            head = f"{row['clip']} | {row['label']} | diff_pct={row['diff_pct']:.2f}  mean={row['proxy_mean']:.2f}  max={row['proxy_max']:.2f}"
            for cam in ("left","right","top"):
                vp = Path(row.get(f"cropped_{cam}", ""))
                if not (vp and vp.exists()): continue
                frs = sample_video_frames(vp, num_frames=args.viz_frames, backend=backend)
                sheet = make_contact_sheet(frs, cols=args.viz_cols, tile_h=args.viz_height)
                if sheet is not None:
                    sheet = add_header_bar(sheet, f"{head} | {cam}")
                    outp = viz_dir / f"{row['clip']}_{cam}.png"
                    sheet.save(outp)
                    print(f"[viz] wrote {outp}")
        print(f"[viz] Done. See {viz_dir}")

if __name__ == "__main__":
    # lazy torch import so script still runs MOT-only without torch
    try:
        import torch
    except Exception:
        torch = None
    main()
