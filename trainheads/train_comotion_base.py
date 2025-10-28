#!/usr/bin/env python
import os, sys, io, json, math, argparse, random, warnings
from datetime import datetime
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import average_precision_score
from tqdm import tqdm
import math
# ---- Py3.11 shim for chumpy ----
import inspect
from collections import namedtuple
import time
from types import SimpleNamespace

from sklearn.metrics import (
    average_precision_score,
    precision_recall_fscore_support,
    confusion_matrix,
    top_k_accuracy_score,
)
# --- Speed knobs for Ampere/ADA ---
torch.backends.cuda.matmul.allow_tf32 = True   # speed up fp32 matmuls on A100
torch.backends.cudnn.allow_tf32 = True

# Prefer flash / memory-efficient SDPA kernels (PyTorch 2+)
try:
    from torch.backends.cuda import sdp_kernel
    sdp_kernel.enable_flash_sdp(True)
    sdp_kernel.enable_math_sdp(True)
    sdp_kernel.enable_mem_efficient_sdp(True)
except Exception:
    pass


# optional: save a heatmap if matplotlib is available
try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None

if not hasattr(inspect, "getargspec"):
    ArgSpec = namedtuple("ArgSpec", "args varargs keywords defaults")
    def _getargspec(func):
        fs = inspect.getfullargspec(func)
        return ArgSpec(args=fs.args, varargs=fs.varargs, keywords=fs.varkw, defaults=fs.defaults)
    inspect.getargspec = _getargspec
# --------------------------------
import numpy as np

# --- NumPy ≥1.24 compatibility shims for legacy deps (chumpy/smplx) ---
def _np_alias(name, target):
    if not hasattr(np, name):
        setattr(np, name, target)

_np_alias("bool",    np.bool_)              # removed alias
_np_alias("int",     int)                   # removed alias
_np_alias("float",   float)                 # removed alias
_np_alias("complex", np.complex128)         # removed alias
_np_alias("object",  object)                # removed alias
_np_alias("long",    int)                   # removed alias
_np_alias("str",     str)                   # removed alias
_np_alias("unicode", np.unicode_ if hasattr(np, "unicode_") else str)  # removed alias
# ----------------------------------------------------------------------




# =========================
# Logging to TXT (tee)
# =========================
class Tee(io.TextIOBase):
    def __init__(self, *streams):
        self.streams = streams
    def write(self, data):
        for s in self.streams:
            s.write(data)
            s.flush()
        return len(data)
    def flush(self):
        for s in self.streams:
            s.flush()

def start_text_logging(out_dir: str, prefix: str = "comotion_only", ts: str | None = None):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    ts = ts or datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = Path(out_dir) / f"{prefix}_{ts}.txt"
    fh = open(log_path, "a", buffering=1, encoding="utf-8")
    sys.stdout = Tee(sys.stdout, fh)
    sys.stderr = Tee(sys.stderr, fh)
    print("="*100)
    print(f"[RUN START] {ts} | log={log_path}")
    return str(log_path), fh, ts


# =========================
# Utils
# =========================
def count_trainable_params(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def topk_accuracies(probs: np.ndarray, y: np.ndarray, ks=(1, 5)):
    # probs: [N,C] (softmax probabilities), y: [N]
    res = {}
    order = np.argpartition(probs, -max(ks), axis=1)
    for k in ks:
        k = min(k, probs.shape[1])
        topk = order[:, -k:]
        correct = (topk == y[:, None]).any(axis=1).mean()
        res[f"top{k}"] = float(correct)
    return res

def compute_macro_prf(y_true_int: np.ndarray, y_pred_int: np.ndarray):
    P, R, F1, _ = precision_recall_fscore_support(
        y_true_int, y_pred_int, average="macro", zero_division=0
    )
    return float(P), float(R), float(F1)

def compute_per_class_tables(probs: np.ndarray, y: np.ndarray, class_names: list[str]):
    """Return per-class PRF + AP as a DataFrame-like dict (easy to write CSV)."""
    # PRF per class from hard predictions
    y_pred = probs.argmax(axis=1)
    P_c, R_c, F1_c, support = precision_recall_fscore_support(
        y, y_pred, labels=np.arange(len(class_names)), average=None, zero_division=0
    )
    # AP per class from scores (one-vs-rest)
    onehot = np.zeros((y.shape[0], len(class_names)), dtype=np.int32)
    onehot[np.arange(y.shape[0]), y] = 1
    AP_c = []
    for c in range(len(class_names)):
        # AP is undefined (and sklearn will warn) if class has zero positives
        if onehot[:, c].sum() == 0:
            AP_c.append(np.nan)
        else:
            AP_c.append(average_precision_score(onehot[:, c], probs[:, c]))
    table = {
        "class": class_names,
        "support": support.astype(int).tolist(),
        "precision": [float(x) for x in P_c],
        "recall":    [float(x) for x in R_c],
        "f1":        [float(x) for x in F1_c],
        "AP":        [float(x) if x == x else None for x in AP_c],  # None for NaN
    }
    return table

def measure_inference_fps(model, loader, device, amp: bool = True):
    """Throughput on the full validation loader (frames/sec)."""
    model.eval()
    total_frames = 0
    start = None
    with torch.no_grad():
        ctx = (torch.amp.autocast if torch.cuda.is_available() else torch.cpu.amp.autocast)
        # short warmup (don’t time)
        warmup = 5
        for i, (feats, _) in enumerate(loader):
            feats = feats.to(device, non_blocking=True)
            if i < warmup:
                with ctx(device_type="cuda", enabled=amp and torch.cuda.is_available()):
                    _ = model(feats)
                continue
            if start is None:
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                start = time.time()
            with ctx(device_type="cuda", enabled=amp and torch.cuda.is_available()):
                _ = model(feats)
            T = feats.shape[1]  # frames per clip
            B = feats.shape[0]
            total_frames += B * T
        if start is None:  # loader had <= warmup batches
            return 0.0
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.time() - start
    return float(total_frames / max(elapsed, 1e-9))

def save_confusion_artifacts(y_true: np.ndarray, y_pred: np.ndarray, class_names: list[str], out_dir: Path):
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(len(class_names)), normalize='true')
    # CSV
    cm_csv = out_dir / "confusion_matrix_normalized.csv"
    with open(cm_csv, "w") as f:
        f.write(",".join(["class"] + class_names) + "\n")
        for i, row in enumerate(cm):
            f.write(",".join([class_names[i]] + [f"{x:.6f}" for x in row]) + "\n")
    print(f"[FINAL] Saved normalized confusion matrix CSV -> {cm_csv}")
    # Optional PNG heatmap
    if plt is not None and len(class_names) <= 100:  # keep huge matrices as CSV only
        h = max(6, min(18, int(len(class_names) * 0.20)))
        w = h
        fig = plt.figure(figsize=(w, h), dpi=150)
        ax = plt.gca()
        im = ax.imshow(cm, interpolation='nearest', cmap='viridis')
        plt.colorbar(im, fraction=0.046, pad=0.04)
        ax.set_xticks(np.arange(len(class_names)))
        ax.set_yticks(np.arange(len(class_names)))
        ax.set_xticklabels(class_names, rotation=90, fontsize=6)
        ax.set_yticklabels(class_names, fontsize=6)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title("Normalized Confusion Matrix")
        plt.tight_layout()
        png_path = out_dir / "confusion_matrix_normalized.png"
        plt.savefig(png_path)
        plt.close(fig)
        print(f"[FINAL] Saved confusion matrix heatmap -> {png_path}")

def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def exists(p): return os.path.exists(p)

def make_label_map(labels):
    uniq = sorted(set(labels))
    return {lbl: i for i, lbl in enumerate(uniq)}

def load_label_map(path):
    with open(path, "r") as f:
        return json.load(f)

def save_json(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)

def uniform_to_T(x: torch.Tensor, T: int) -> torch.Tensor:
    """Uniformly sample (or pad) sequence to exactly T frames. x: [Tin, D]."""
    Tin = x.shape[0]
    if Tin == T:
        return x
    if Tin > T:
        idx = torch.linspace(0, Tin - 1, T).round().long()
        return x.index_select(0, idx)
    # pad
    pad = torch.zeros(T - Tin, x.shape[1], dtype=x.dtype)
    return torch.cat([x, pad], dim=0)

# ---- TSN (Temporal Segment Sampling) helpers ----

def tsn_snippet_indices(T_in: int, N: int, k: int, jitter: bool) -> torch.Tensor:
    """Split [0, T_in) into N segments; pick a k-length contiguous window per segment.
    Returns concatenated indices of length N*k. If a segment is shorter than k,
    repeat the last valid index to fill."""
    if T_in <= 0:
        return torch.zeros(0, dtype=torch.long)

    bounds = torch.linspace(0, T_in, steps=N + 1)
    idxs = []
    for i in range(N):
        s = int(math.floor(bounds[i].item()))
        e = int(math.ceil(bounds[i + 1].item()))  # exclusive
        seg_len = max(1, e - s)
        win = min(k, seg_len)

        # valid start so that start+win-1 < e
        start_lo = s
        start_hi = max(s, e - win)
        start = (random.randint(start_lo, start_hi) if jitter else (start_lo + start_hi) // 2)

        cont = torch.arange(start, start + win)
        cont = torch.clamp(cont, max=e - 1)

        if win < k:
            pad = cont[-1].repeat(k - win)
            cont = torch.cat([cont, pad], dim=0)

        idxs.append(cont)
    return torch.cat(idxs, dim=0)  # [N*k]

def tsn_sample(x: torch.Tensor, N: int, k: int, train_mode: bool) -> torch.Tensor:
    """x: [Tin, D] -> [N*k, D] via TSN snippet sampling."""
    Tin = x.shape[0]
    if Tin == 0:
        return torch.zeros(N * k, x.shape[1], dtype=x.dtype)
    idx = tsn_snippet_indices(Tin, N, k, jitter=train_mode)
    return x.index_select(0, idx)
# -----------------------------------------------


# =========================
# Dataset
# =========================
class CoMotionIndexDataset(Dataset):
    """
    Reads CSV with columns: clip, label, path_skel (from your indices builder).
    Loads CoMotion .pt dicts: keys expected: pose (T,72), trans (T,3), betas (T,10), id (T,), frame_idx (T,)
    Optionally decodes to 3D joints via SMPL (requires smplx + model files).
    """
    def __init__(self,
                 csv_path: str,
                 label2id: dict,
                 T: int = 64,
                 decode_smpl: bool = False,
                 smpl_model_dir: str | None = None,
                 center_root: bool = True,
                 include_trans: bool = False,
                 include_betas: bool = False,
                 cache_dir: str | None = None,
                 drop_missing: bool = True,
                 # --- NEW (TSN) ---
                 tsn_segments: int = 0,
                 tsn_snippet: int = 0,
                 jitter: bool = False):
        self.df = pd.read_csv(csv_path)
        # allow "path_skel" or "path_skeleton"
        col_skel = "path_skel" if "path_skel" in self.df.columns else ("path_skeleton" if "path_skeleton" in self.df.columns else None)
        needed = {"clip", "label", col_skel}
        if col_skel is None or not needed.issubset(set(self.df.columns)):
            raise ValueError(f"CSV must have columns including 'clip','label','path_skel'. Got: {self.df.columns.tolist()}")

        rows = []
        for _, r in self.df.iterrows():
            p = str(r[col_skel])
            if drop_missing and (not p or not exists(p)):
                continue
            rows.append({"clip": str(r["clip"]), "label": str(r["label"]), "path_skel": p})
        if len(rows) == 0:
            raise RuntimeError(f"No valid rows in {csv_path}. Check paths.")

        self.rows = rows
        self.label2id = label2id
        self.T = T
        self.decode_smpl = decode_smpl
        self.center_root = center_root
        self.include_trans = include_trans
        self.include_betas = include_betas
        self.tsn_segments = int(tsn_segments)
        self.tsn_snippet  = int(tsn_snippet)
        self.jitter       = bool(jitter)
        self._use_tsn     = (self.tsn_segments > 0 and self.tsn_snippet > 0)

        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        # SMPL layer (optional)
        self.smpl_layer = None
        if self.decode_smpl:
            try:
                import smplx
            except ImportError as e:
                raise ImportError("smplx not installed. `pip install smplx`") from e
            if not smpl_model_dir or not Path(smpl_model_dir).exists():
                raise FileNotFoundError("Provide --smpl_model_dir pointing to SMPL/SMPLX models (e.g., SMPL_NEUTRAL.pkl).")
            self.smpl_layer = smplx.create(
                model_path=str(Path(smpl_model_dir)),  # e.g., .../comotion_demo/data
                model_type='smpl',
                gender='neutral',
                num_betas=10,
                use_pca=False,
                ext='pkl',   # <-- for SMPL_NEUTRAL.pkl
            ).eval()

    def __len__(self): return len(self.rows)

    def _pick_track(self, d: dict) -> dict:
        """
        CoMotion saves one tracked person per .pt in many pipelines.
        If multiple, pick the ID with most frames. Expect keys: 'pose','trans','betas','id','frame_idx'.
        """
        # Standard single-track dict
        if all(k in d for k in ("pose", "trans", "betas")):
            return d

        # Fallback: list/tuple of dicts or nested structure
        cand = []
        if isinstance(d, (list, tuple)):
            for it in d:
                if isinstance(it, dict) and "pose" in it:
                    cand.append(it)
        elif isinstance(d, dict):
            # sometimes stored under 'tracks'
            tracks = d.get("tracks", [])
            if isinstance(tracks, (list, tuple)):
                for it in tracks:
                    if isinstance(it, dict) and "pose" in it:
                        cand.append(it)
        if not cand:
            raise RuntimeError("Unexpected .pt structure; expected dict with 'pose','trans','betas' or list thereof.")
        # choose longest
        cand.sort(key=lambda x: int(x["pose"].shape[0] if isinstance(x["pose"], torch.Tensor) else len(x["pose"])), reverse=True)
        return cand[0]

    def _decode_smpl_to_joints(self, pose: torch.Tensor, trans: torch.Tensor, betas: torch.Tensor) -> torch.Tensor:
        """
        pose: [T,72] axis-angle, trans: [T,3], betas: [T,10] or [10]
        returns joints [T, J, 3]
        """
        T = pose.shape[0]
        global_orient = pose[:, :3]
        body_pose     = pose[:, 3:72]  # 69

        if betas.ndim == 1:
            betas = betas.unsqueeze(0).expand(T, -1)  # [T,10]
        if trans.ndim == 1:
            trans = trans.unsqueeze(0).expand(T, -1)  # [T,3]

        # smplx expects float32 tensors
        global_orient = global_orient.float()
        body_pose     = body_pose.float()
        betas         = betas.float()
        trans         = trans.float()

        # Run in chunks if very long (avoid OOM)
        chunks = []
        BS = 512  # adjust if needed
        with torch.no_grad():
            for s in range(0, T, BS):
                e = min(T, s + BS)
                out = self.smpl_layer(
                    betas=betas[s:e],
                    global_orient=global_orient[s:e],
                    body_pose=body_pose[s:e],
                    transl=trans[s:e],
                )
                j = out.joints  # [b, J, 3]
                chunks.append(j)
        joints = torch.cat(chunks, dim=0)  # [T, J, 3]

        if self.center_root:
            joints = joints - joints[:, [0], :]  # root-centered
        return joints

    def _load_feat(self, path_skel: str) -> torch.Tensor:
        """
        Load .pt and produce per-frame features:
          - if decode_smpl: joints [T, J, 3] -> flatten to [T, J*3]
          - else: raw params concat -> [T, 72 + (3?) + (10?)]
        """
        # Disk cache (optional)
        if self.cache_dir:
            cache_name = f"{Path(path_skel).stem}__{'joints' if self.decode_smpl else 'params'}.pt"
            cache_path = self.cache_dir / cache_name
            if cache_path.exists():
                return torch.load(cache_path, map_location="cpu")

        d = torch.load(path_skel, map_location="cpu")
        d = self._pick_track(d)

        pose  = torch.as_tensor(d.get("pose"),  dtype=torch.float32)   # [T,72]
        trans = torch.as_tensor(d.get("trans", torch.zeros((pose.shape[0],3)))), 
        betas = torch.as_tensor(d.get("betas", torch.zeros((pose.shape[0],10))))

        # fix tuple unpack mistake (Python quirk when trailing comma)
        if isinstance(trans, tuple):  trans = trans[0]
        if isinstance(betas, tuple):  betas = betas[0]

        if self.decode_smpl:
            joints = self._decode_smpl_to_joints(pose, trans, betas)   # [T,J,3]
            feat = joints.reshape(joints.shape[0], -1)                 # [T,J*3]
        else:
            parts = [pose]
            if self.include_trans:
                parts.append(trans if trans.ndim == 2 else trans.unsqueeze(0).expand(pose.shape[0], -1))
            if self.include_betas:
                if betas.ndim == 1:
                    betas = betas.unsqueeze(0).expand(pose.shape[0], -1)
                parts.append(betas)
            feat = torch.cat(parts, dim=1)                             # [T,D]

        if self.cache_dir:
            try:
                torch.save(feat, cache_path)
            except Exception:
                pass
        return feat

    def __getitem__(self, idx):
        rec = self.rows[idx]
        path = rec["path_skel"]
        if not exists(path):
            raise FileNotFoundError(path)

        feat = self._load_feat(path)  # [Tin, D]
        if self._use_tsn:
            feat = tsn_sample(feat, N=self.tsn_segments, k=self.tsn_snippet, train_mode=self.jitter)
        else:
            feat = uniform_to_T(feat, self.T)
        feat = feat.to(torch.float32)  # [T,D] where T == self.T if TSN is enabled (enforced below)

        y = self.label2id[str(rec["label"])]
        return feat, y

# =========================
# Model: Attentive probe
# =========================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 2000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe)

    def forward(self, x):  # [B,T,D]
        T = x.size(1)
        return x + self.pe[:T, :].unsqueeze(0)

class SkelOnlyAttnProbe(nn.Module):
    def __init__(self, d_in=72, d_model=256, n_heads=8, depth=2, ff_mult=4, num_classes=13, dropout=0.1, T=64):
        super().__init__()
        self.proj = nn.Identity() if d_in == d_model else nn.Linear(d_in, d_model)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_enc = PositionalEncoding(d_model, max_len=T+1)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model*ff_mult,
            dropout=dropout, batch_first=True, activation="gelu", norm_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=depth)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, num_classes)
        nn.init.trunc_normal_(self.head.weight, std=0.02); nn.init.zeros_(self.head.bias)

    def forward(self, x):  # [B,T,d_in]
        x = self.proj(x)
        B, T, D = x.shape
        cls = self.cls_token.expand(B, 1, D)
        x = torch.cat([cls, x], dim=1)
        x = self.pos_enc(x)
        x = self.encoder(x)
        cls_out = self.norm(x[:, 0])
        return self.head(cls_out)

# =========================
# Eval
# =========================
@torch.inference_mode()
def final_evaluation_and_report(model, loader, device, num_classes, label2id, out_dir, amp=False):
    model.eval()

    # Forward pass over the entire val set, collect predictions
    all_logits, all_targets = [], []
    ctx = torch.amp.autocast if torch.cuda.is_available() else torch.cpu.amp.autocast
    with ctx(device_type="cuda", enabled=amp and torch.cuda.is_available()):
        for feats, y in loader:
            feats = feats.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            logits = model(feats)        
            all_logits.append(logits.detach().cpu())
            all_targets.append(y.detach().cpu())

    logits = torch.cat(all_logits, dim=0)               # [N, C]  (no grad)
    targets = torch.cat(all_targets, dim=0).numpy().astype(int)  # [N]
    probs = torch.softmax(logits, dim=-1).numpy()       # [N, C]
    preds = probs.argmax(axis=1)                        # [N]

    # Top-k accuracy
    top1 = (preds == targets).mean().item()
    try:
        top5 = top_k_accuracy_score(targets, probs, k=5, labels=list(range(num_classes)))
    except Exception:
        # If C < 5, fall back to min(C,5)
        k_eff = min(5, num_classes)
        top5 = top_k_accuracy_score(targets, probs, k=k_eff, labels=list(range(num_classes)))

    # mAP (all) and mAP (valid classes only)
    y_true_ovr = np.zeros((targets.shape[0], num_classes), dtype=np.int32)
    y_true_ovr[np.arange(targets.shape[0]), targets] = 1

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        mAP_all = average_precision_score(y_true_ovr, probs, average="macro")

    support = y_true_ovr.sum(axis=0) > 0
    if support.any():
        mAP_valid = average_precision_score(y_true_ovr[:, support], probs[:, support], average="macro")
    else:
        mAP_valid = float("nan")

    # Macro-averaged Precision/Recall/F1
    macro_P, macro_R, macro_F1, _ = precision_recall_fscore_support(
        targets, preds, average="macro", zero_division=0
    )

    # Per-class PRF and AP
    P_c, R_c, F1_c, Supp_c = precision_recall_fscore_support(
        targets, preds, average=None, labels=np.arange(num_classes), zero_division=0
    )
    # Per-class AP (one-vs-rest)
    AP_c = np.zeros(num_classes, dtype=np.float64)
    for c in range(num_classes):
        if support[c]:
            AP_c[c] = average_precision_score(y_true_ovr[:, c], probs[:, c])
        else:
            AP_c[c] = np.nan

    # Confusion matrix (row-normalized)
    cm = confusion_matrix(targets, preds, labels=np.arange(num_classes)).astype(np.float64)
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(cm, np.maximum(row_sums, 1.0), where=row_sums > 0)  # safe divide
    cm_norm = np.nan_to_num(cm_norm)

    # Pretty print headline metrics
    print("\n================ FINAL EVALUATION (best checkpoint) ================")
    print(f"Top-1 Acc: {top1:.4f}")
    print(f"Top-5 Acc: {top5:.4f}")
    print(f"mAP(all): {mAP_all:.4f}")
    print(f"mAP(valid): {mAP_valid:.4f}")
    print(f"Macro Precision: {macro_P:.4f}  Macro Recall: {macro_R:.4f}  Macro F1: {macro_F1:.4f}")

    # Per-class table
    id2label = {v: k for k, v in label2id.items()}
    class_names = [id2label.get(i, str(i)) for i in range(num_classes)]
    save_confusion_artifacts(targets, preds, class_names, Path(out_dir))
    print("\nPer-class metrics (P / R / F1 / AP / support):")
    for c in range(num_classes):
        name = id2label.get(c, str(c))
        ap   = AP_c[c]
        ap_s = f"{ap:.3f}" if np.isfinite(ap) else "nan"
        print(f"[{c:3d}] {name:40s}  P={P_c[c]:.3f}  R={R_c[c]:.3f}  F1={F1_c[c]:.3f}  AP={ap_s:>5}  N={Supp_c[c]}")

    # Save artifacts
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "confusion_matrix_norm.npy", cm_norm)
    np.save(out_dir / "confusion_matrix_counts.npy", cm)
    pd.DataFrame({
        "class_id": np.arange(num_classes),
        "class_name": [id2label.get(i, str(i)) for i in range(num_classes)],
        "precision": P_c, "recall": R_c, "f1": F1_c, "AP": AP_c, "support": Supp_c
    }).to_csv(out_dir / "per_class_metrics.csv", index=False)

    # Also return a dict if you want to consume programmatically
    return {
        "top1": float(top1), "top5": float(top5),
        "mAP_all": float(mAP_all), "mAP_valid": float(mAP_valid),
        "macroP": float(macro_P), "macroR": float(macro_R), "macroF1": float(macro_F1)
    }


@torch.inference_mode()
def evaluate(model, loader, device, num_classes, amp=False, label2id=None, bg_name='[OP000] No action'):
    model.eval()
    all_logits, all_targets = [], []

    ctx = torch.amp.autocast if torch.cuda.is_available() else torch.cpu.amp.autocast
    with ctx(device_type="cuda", enabled=amp and torch.cuda.is_available()):
        for feats, y in loader:
            feats = feats.to(device, non_blocking=True)
            y     = y.to(device, non_blocking=True)
            logits = model(feats)
            all_logits.append(logits.detach().cpu())
            all_targets.append(y.detach().cpu())

    logits  = torch.cat(all_logits, dim=0)
    probs   = torch.softmax(logits, dim=-1).numpy()     # [N, C]
    targets = torch.cat(all_targets, dim=0).numpy().astype(int)  # [N]
    preds   = probs.argmax(axis=1)

    # Top-k accuracy
    top1 = float((preds == targets).mean())
    try:
        k_eff = min(5, num_classes)
        top5 = float(top_k_accuracy_score(targets, probs, k=k_eff, labels=list(range(num_classes))))
    except Exception:
        top5 = float('nan')

    # mAP (all) and mAP (valid-only)
    y_true = np.zeros((targets.shape[0], num_classes), dtype=np.int32)
    y_true[np.arange(targets.shape[0]), targets] = 1
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        mAP_all = float(average_precision_score(y_true, probs, average="macro"))
    support = y_true.sum(axis=0) > 0
    if support.any():
        mAP_valid = float(average_precision_score(y_true[:, support], probs[:, support], average="macro"))
    else:
        mAP_valid = float('nan')

    # Macro P/R/F1 from hard preds
    from sklearn.metrics import precision_recall_fscore_support
    macroP, macroR, macroF1, _ = precision_recall_fscore_support(
        targets, preds, average="macro", zero_division=0
    )
    macroP = float(macroP); macroR = float(macroR); macroF1 = float(macroF1)

    # Optional: mAP with background removed
    mAP_noBG = None
    if label2id and (bg_name in label2id):
        bg_id = label2id[bg_name]
        valid_no_bg = support.copy()
        if 0 <= bg_id < len(valid_no_bg):
            valid_no_bg[bg_id] = False
        if valid_no_bg.any():
            mAP_noBG = float(average_precision_score(y_true[:, valid_no_bg], probs[:, valid_no_bg], average="macro"))

    return {
        "top1": top1,
        "top5": top5,
        "mAP_all": mAP_all,
        "mAP_valid": mAP_valid,
        "mAP_noBG": mAP_noBG,
        "macroP": macroP,
        "macroR": macroR,
        "macroF1": macroF1,
    }



# =========================
# Train
# =========================
def train(args):
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Label map
    train_df = pd.read_csv(args.train_csv)
    train_labels = [str(x) for x in train_df["label"].tolist()]
    if args.labelmap and exists(args.labelmap):
        label2id = load_label_map(args.labelmap)
    else:
        label2id = make_label_map(train_labels)
        if args.labelmap:
            save_json(label2id, args.labelmap)
    num_classes = len(label2id)
    print(f"Classes: {num_classes} -> {label2id}")
    # ---- TSN normalization: if enabled, ensure T = N * k ----
    # ---- TSN normalization: if enabled, ensure T = N * k ----
    if getattr(args, "tsn_segments", 0) and args.tsn_segments > 0:
        if not getattr(args, "tsn_snippet", 0) or args.tsn_snippet <= 0:
            # If k not provided, derive it from T and N (floor at 1)
            args.tsn_snippet = max(1, args.T // args.tsn_segments)
        T_eff = args.tsn_segments * args.tsn_snippet
        if T_eff != args.T:
            print(f"[TSN] Adjusting T from {args.T} -> {T_eff} to match N*k.")
            args.T = T_eff
        print(f"[TSN] Using N={args.tsn_segments}, k={args.tsn_snippet}, T={args.T} "
              f"(train=jitter, val=center)")
        assert args.T == args.tsn_segments * args.tsn_snippet, "T must equal N*k when TSN is enabled."



        # Datasets
    train_ds = CoMotionIndexDataset(
        args.train_csv, label2id,
        T=args.T,
        decode_smpl=args.decode_smpl,
        smpl_model_dir=args.smpl_model_dir,
        center_root=not args.no_center_root,
        include_trans=args.include_trans,
        include_betas=args.include_betas,
        cache_dir=args.cache_dir,
        drop_missing=True,
        # --- TSN ---
        tsn_segments=getattr(args, "tsn_segments", 0),
        tsn_snippet=getattr(args, "tsn_snippet", 0),
        jitter=True  # random snippet for training
    )
    val_ds = CoMotionIndexDataset(
        args.val_csv, label2id,
        T=args.T,
        decode_smpl=args.decode_smpl,
        smpl_model_dir=args.smpl_model_dir,
        center_root=not args.no_center_root,
        include_trans=args.include_trans,
        include_betas=args.include_betas,
        cache_dir=args.cache_dir,
        drop_missing=True,
        # --- TSN ---
        tsn_segments=getattr(args, "tsn_segments", 0),
        tsn_snippet=getattr(args, "tsn_snippet", 0),
        jitter=False  # center (deterministic) snippet for eval
    )


    # Infer d_in from one sample
    sample_feat, _ = train_ds[0]
    d_in = sample_feat.shape[1]
    print(f"Inferred skeleton feature dim d_in={d_in}")


    # ---- DataLoaders (safe defaults + eval batch override) ----
    bs_train = max(1, int(args.batch_size))
    eval_bs_arg = getattr(args, "eval_batch_size", None)  # allow None or 0 to mean "use train_bs"
    bs_val = (max(1, int(eval_bs_arg))
              if (eval_bs_arg is not None and int(eval_bs_arg) > 0)
              else bs_train)

    nw = max(0, int(getattr(args, "num_workers", 0)))
    prefetch_factor = int(getattr(args, "prefetch_factor", 2))

    loader_kwargs = dict(num_workers=nw, pin_memory=True)
    if nw > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = prefetch_factor  # only valid when workers > 0

    train_loader = DataLoader(
        train_ds, batch_size=bs_train, shuffle=True, drop_last=True, **loader_kwargs
    )
    val_loader = DataLoader(
        val_ds, batch_size=bs_val, shuffle=False, **loader_kwargs
    )

    print(f"[Loader] train_bs={bs_train}  val_bs={bs_val}  "
          f"workers={nw}  prefetch={loader_kwargs.get('prefetch_factor', 'n/a')}")
    

    model = SkelOnlyAttnProbe(
        d_in=d_in, d_model=args.d_model, n_heads=args.n_heads, depth=args.depth,
        ff_mult=args.ff_mult, num_classes=num_classes, dropout=args.dropout, T=args.T
    ).to(device)

    params_m = count_trainable_params(model) / 1e6
    print(f"Trainable parameters: {params_m:.3f}M")


    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Cosine LR with warmup
    total_steps = args.epochs * max(1, math.ceil(len(train_loader)))
    warmup_steps = int(0.05 * total_steps)
    def lr_schedule(step):
        if step < warmup_steps: return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scaler = torch.amp.GradScaler('cuda', enabled=args.amp and torch.cuda.is_available())
    criterion = nn.CrossEntropyLoss()

    # Logging / ckpts
    out_dir = Path(args.run_dir); out_dir.mkdir(parents=True, exist_ok=True)
    best_map = -1.0
    global_step = 0

    # Train
    for epoch in range(1, args.epochs+1):
        model.train()
        epoch_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", ncols=100)
        for feats, y in pbar:
            feats = feats.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            # LR schedule
            for g in optim.param_groups:
                g["lr"] = args.lr * lr_schedule(global_step)

            optim.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=args.amp and torch.cuda.is_available()):
                logits = model(feats)
                loss = criterion(logits, y)
            scaler.scale(loss).backward()
            if args.grad_clip is not None:
                scaler.unscale_(optim)
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optim)
            scaler.update()

            global_step += 1
            epoch_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}", "step": global_step})

        # Eval
        metrics = evaluate(model, val_loader, device, num_classes, amp=args.amp, label2id=label2id)
        avg_loss = epoch_loss / max(len(train_loader), 1)
        # ---- pretty/robust printing ----
        no_bg = metrics.get("mAP_noBG", None)
        no_bg_str = f"{no_bg:.4f}" if (isinstance(no_bg, float) and np.isfinite(no_bg)) else "n/a"

        top5 = metrics.get("top5", float("nan"))
        top5_str = f"{top5:.4f}" if (isinstance(top5, float) and np.isfinite(top5)) else "n/a"

        print(
            f"[Epoch {epoch}] "
            f"train_loss={avg_loss:.4f}  "
            f"val_top1={metrics['top1']:.4f}  val_top5={top5_str}  "
            f"val_mAP_all={metrics['mAP_all']:.4f}  val_mAP_valid={metrics['mAP_valid']:.4f}  "
            f"val_macroP={metrics['macroP']:.4f}  val_macroR={metrics['macroR']:.4f}  val_macroF1={metrics['macroF1']:.4f}  "
            f"val_mAP_noBG={no_bg_str}"
        )
# --------------------------------

        # Save best by val_mAP_valid
        current_score = metrics["mAP_valid"]

        if current_score > best_map:
            best_map = current_score
            ckpt = {
                "model": model.state_dict(),
                "args": vars(args),
                "label2id": label2id,
                "val_top1": metrics["top1"],
                "val_top5": metrics["top5"],
                "val_mAP_all": metrics["mAP_all"],
                "val_mAP_valid": metrics["mAP_valid"],
                "val_mAP_noBG": metrics["mAP_noBG"],
                "val_macroP": metrics["macroP"],
                "val_macroR": metrics["macroR"],
                "val_macroF1": metrics["macroF1"],
                "d_in": d_in,
                "params_m": params_m,
            }

            torch.save(ckpt, out_dir / "best_skel_only.ckpt")
            print(f"Saved best checkpoint to {out_dir/'best_skel_only.ckpt'} (mAP_valid={best_map:.4f})")
    # After the training loop
    best_path = Path(args.run_dir) / "best_skel_only.ckpt"
    if best_path.exists():
        print(f"\n[FINAL] Loading best checkpoint: {best_path}")
        ckpt = torch.load(best_path, map_location="cpu")
        model.load_state_dict(ckpt["model"])
        model.to(device)

        final_evaluation_and_report(
            model=model,
            loader=val_loader,
            device=device,
            num_classes=num_classes,
            label2id=label2id,
            out_dir=args.run_dir,   # saves confusion/per-class CSV here
            amp=args.amp
        )
    else:
        print("[FINAL] Best checkpoint not found; skipping final evaluation.")

# =========================
# Occlusion study (no training, no attn maps)

@torch.inference_mode()
def _eval_on_csv_comotion(model, csv_path, label2id, args):
    """Build a loader from a CSV and run evaluate()."""
    ds = CoMotionIndexDataset(
        csv_path, label2id,
        T=args.T,
        decode_smpl=args.decode_smpl,
        smpl_model_dir=args.smpl_model_dir,
        center_root=not args.no_center_root,
        include_trans=args.include_trans,
        include_betas=args.include_betas,
        cache_dir=args.cache_dir,
        drop_missing=True,
        tsn_segments=getattr(args, "tsn_segments", 0),
        tsn_snippet=getattr(args, "tsn_snippet", 0),
        jitter=False,
    )

    # ---- robust fallbacks so None never hits int() ----
    bs = getattr(args, "eval_batch_size", None)
    if not bs:
        bs = getattr(args, "batch_size", None)
    if not bs:
        bs = 64  # sensible default for eval
    try:
        bs = int(bs)
    except Exception:
        bs = 64
    bs = max(1, bs)

    nw = getattr(args, "num_workers", 0)
    try:
        nw = int(nw)
    except Exception:
        nw = 0
    nw = max(0, nw)

    loader = DataLoader(
        ds,
        batch_size=bs,
        shuffle=False,
        num_workers=nw,
        pin_memory=True,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    return evaluate(model, loader, device, num_classes=len(label2id), amp=args.amp, label2id=label2id)

def run_occlusion_study(args):
    """
    Evaluate a trained CoMotion (skeleton-only) checkpoint on low/high occlusion CSVs.
    Prints a small Δ table and writes JSON/MD artifacts next to the checkpoint.
    """
    ckpt_path = Path(args.ckpt) if args.ckpt else (Path(args.log_dir) / (args.run_name or "") / "best_skel_only.ckpt")
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location="cpu")

    # label map
    label2id = ckpt.get("label2id", None)
    if label2id is None:
        if not (args.labelmap and Path(args.labelmap).exists()):
            raise RuntimeError("label2id missing in ckpt; please pass --labelmap pointing to label2id.json")
        label2id = load_label_map(args.labelmap)

    # rebuild model from checkpoint args (fallback to CLI defaults)
    margs = ckpt.get("args", {}) if isinstance(ckpt.get("args", {}), dict) else {}
    d_model = margs.get("d_model", args.d_model)
    n_heads = margs.get("n_heads", args.n_heads)
    depth   = margs.get("depth",   args.depth)
    ff_mult = margs.get("ff_mult", args.ff_mult)
    dropout = margs.get("dropout", args.dropout)
    T_eff   = margs.get("T",       args.T)

    # d_in  — use what we saved in the ckpt if present; else infer from one sample
    d_in = ckpt.get("d_in", None)
    if d_in is None:
        tmp_ds = CoMotionIndexDataset(
            args.occ_low_csv, label2id,
            T=T_eff,
            decode_smpl=args.decode_smpl,
            smpl_model_dir=args.smpl_model_dir,
            center_root=not args.no_center_root,
            include_trans=args.include_trans,
            include_betas=args.include_betas,
            cache_dir=args.cache_dir,
            drop_missing=True,
            tsn_segments=getattr(args, "tsn_segments", 0),
            tsn_snippet=getattr(args, "tsn_snippet", 0),
            jitter=False
        )
        sample_feat, _ = tmp_ds[0]
        d_in = int(sample_feat.shape[1])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SkelOnlyAttnProbe(
        d_in=d_in, d_model=d_model, n_heads=n_heads, depth=depth,
        ff_mult=ff_mult, num_classes=len(label2id), dropout=dropout, T=T_eff
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    # evaluate both splits
    low_metrics  = _eval_on_csv_comotion(model, args.occ_low_csv,  label2id, args)
    high_metrics = _eval_on_csv_comotion(model, args.occ_high_csv, label2id, args)

    low_top1  = 100.0 * float(low_metrics["top1"])
    high_top1 = 100.0 * float(high_metrics["top1"])
    delta_pp  = high_top1 - low_top1

    print("\n=== Occlusion Study (CoMotion skeleton-only) ===")
    print(f"Low  ({Path(args.occ_low_csv).name}):  Top-1 = {low_top1:.1f}%")
    print(f"High ({Path(args.occ_high_csv).name}): Top-1 = {high_top1:.1f}%")
    print(f"Δ (High - Low): {delta_pp:+.1f} pp\n")

    # write artifacts next to the checkpoint
    out_dir = ckpt_path.parent
    payload = {
        "model": "CoMotion (skeleton-only)",
        "low_csv":  str(args.occ_low_csv),
        "high_csv": str(args.occ_high_csv),
        "low_top1":  low_top1,
        "high_top1": high_top1,
        "delta_pp":  delta_pp,
        "low_metrics":  low_metrics,
        "high_metrics": high_metrics,
        "ckpt": str(ckpt_path)
    }
    with open(out_dir / "occ_eval_comotion.json", "w") as f:
        json.dump(payload, f, indent=2)

    with open(out_dir / "occ_eval_comotion.md", "w") as f:
        f.write("| Model | Top-1 (Low) | Top-1 (High) | Δ (pp) |\n")
        f.write("|---|---:|---:|---:|\n")
        f.write(f"| CoMotion (skeleton-only) | {low_top1:.1f}% | {high_top1:.1f}% | {delta_pp:+.1f} |\n")

    return payload



# =========================
# Main
# =========================
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_csv", type=str, required=False)
    ap.add_argument("--val_csv",   type=str, required=False)

    # logging
    ap.add_argument("--log_dir",     type=str, default="/workspace/har-fusion/runs/skel_only")
    ap.add_argument("--txt_log_dir", type=str, default="/workspace/trainheads/runs/comotion_only",
                    help="Directory for timestamped .txt logs (mirrors stdout/stderr).")
    ap.add_argument("--labelmap",    type=str, default="/workspace/har-fusion/runs/skel_only/label2id.json")

    # data/modeling
    ap.add_argument("--T",           type=int, default=64)
    ap.add_argument("--d_model",     type=int, default=256)
    ap.add_argument("--n_heads",     type=int, default=8)
    ap.add_argument("--depth",       type=int, default=2)
    ap.add_argument("--ff_mult",     type=int, default=4)
    ap.add_argument("--dropout",     type=float, default=0.1)

    # CoMotion feature options
    ap.add_argument("--decode_smpl", action="store_true",
                    help="If set, decode pose/trans/betas to 3D joints with SMPL (requires --smpl_model_dir).")
    ap.add_argument("--smpl_model_dir", type=str, default=None,
                    help="Folder containing SMPL model files (e.g., SMPL_NEUTRAL.pkl).")
    ap.add_argument("--no_center_root", action="store_true", help="Do not root-center joints when decoding SMPL.")
    ap.add_argument("--include_trans", action="store_true", help="If not decoding SMPL, include trans in raw features.")
    ap.add_argument("--include_betas", action="store_true", help="If not decoding SMPL, include betas in raw features.")
    ap.add_argument("--cache_dir", type=str, default=None, help="Optional disk cache for per-clip features.")

    # training
    ap.add_argument("--epochs",      type=int, default=30)
    ap.add_argument("--batch_size",  type=int, default=128)
    ap.add_argument("--num_workers", type=int, default=6)
    ap.add_argument("--lr",          type=float, default=3e-4)
    ap.add_argument("--weight_decay",type=float, default=0.05)
    ap.add_argument("--grad_clip",   type=float, default=1.0)
    ap.add_argument("--amp",         action="store_true")
    ap.add_argument("--seed",        type=int, default=42)
    # TSN sampling
    ap.add_argument("--tsn_segments", type=int, default=0,
                    help="If >0, enable TSN with N segments (use together with --tsn_snippet).")
    ap.add_argument("--tsn_snippet", type=int, default=0,
                    help="Snippet length k per segment. If 0, it is derived so that T=N*k.")
    ap.add_argument("--eval_batch_size", type=int, default=None, help="If set, overrides batch size for evaluation; otherwise uses --batch_size.")
    ap.add_argument("--prefetch_factor", type=int, default=4)
    ap.add_argument("--run_name", type=str, default=None, help="If set, use this subfolder under --log_dir; otherwise a timestamp is used.")
        # -------- Occlusion study mode --------
    ap.add_argument("--occ_eval", action="store_true",
                    help="Run occlusion ablation (no training).")
    ap.add_argument("--occ_low_csv",  type=str, default=None,
                    help="CSV for low-occlusion split.")
    ap.add_argument("--occ_high_csv", type=str, default=None,
                    help="CSV for high-occlusion split.")
    ap.add_argument("--ckpt",         type=str, default=None,
                    help="Path to trained CoMotion checkpoint (.ckpt).")




    args = ap.parse_args()
        # ---- Occlusion ablation short-circuit ----
    if args.occ_eval:
        missing = [k for k in ("occ_low_csv", "occ_high_csv", "ckpt") if not getattr(args, k, None)]
        if missing:
            raise SystemExit(f"--occ_eval requires: {', '.join('--'+m for m in missing)}")

        # Optional: TSN sanity so your T matches N*k if needed
        if getattr(args, "tsn_segments", 0) and args.tsn_segments > 0:
            if not getattr(args, "tsn_snippet", 0) or args.tsn_snippet <= 0:
                args.tsn_snippet = max(1, args.T // args.tsn_segments)
            T_eff = args.tsn_segments * args.tsn_snippet
            if T_eff != args.T:
                print(f"[TSN] Adjusting T from {args.T} -> {T_eff} to match N*k.")
                args.T = T_eff

        run_occlusion_study(args)
        sys.exit(0)

    _run_id = args.run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
    # start TXT logging immediately
    log_path, log_fh, _ = start_text_logging(args.txt_log_dir, ts=_run_id)
    args.run_dir = str(Path(args.log_dir) / _run_id)
    print(f"[RUN DIR] {args.run_dir}")
    try:
        print("Args:", json.dumps(vars(args), indent=2))
        print(f"PyTorch: {torch.__version__} | CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA device count: {torch.cuda.device_count()} | current device: {torch.cuda.current_device()}")
            print(f"Device name: {torch.cuda.get_device_name(torch.cuda.current_device())}")
        train(args)
    finally:
        print(f"[RUN END] log saved to: {log_path}")
        try: log_fh.close()
        except Exception: pass
