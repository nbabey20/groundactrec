#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
train_gated_fusion_v2.py  —  GRU-based gated recurrent fusion baseline
Parity with cross-attention trainer: CLI, data, metrics, artifacts, reports.

This trainer:
- Takes identical inputs/flags as train_fusion.py (TSN, SMPL decode, etc.)
- Computes identical metrics (Top-1/Top-5/mAP_all/mAP_valid/mAP_noBG/Macro P/R/F1)
- Writes identical artifacts (epoch_metrics.csv, per_class_metrics.csv,
  confusion_matrix_normalized.csv/png, final_eval.json/txt) and saves best_fusion.ckpt
- Supports --occ_eval with low/high CSVs for occlusion ablation.

Note: --dump_attention/--attn_layers are accepted but ignored (non-attentional model).
"""

import os, sys, io, json, math, argparse, random, warnings, time
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm

from sklearn.metrics import (
    average_precision_score,
    precision_recall_fscore_support,
    confusion_matrix,
    top_k_accuracy_score,
)

# --- Py3.11 / NumPy 2.0 compatibility shims (match cross-attention script) ---
import numpy as _np
if not hasattr(_np, "bool"):    _np.bool = _np.bool_
if not hasattr(_np, "int"):     _np.int = int
if not hasattr(_np, "float"):   _np.float = float
if not hasattr(_np, "complex"): _np.complex = complex
if not hasattr(_np, "object"):  _np.object = object
if not hasattr(_np, "unicode"): _np.unicode = str
if not hasattr(_np, "str"):     _np.str = str

import inspect as _inspect
if not hasattr(_inspect, "getargspec"):
    from collections import namedtuple as _namedtuple
    def _getargspec(func):
        fs = _inspect.getfullargspec(func)
        ArgSpec = _namedtuple("ArgSpec", "args varargs keywords defaults")
        return ArgSpec(fs.args, fs.varargs, fs.varkw, fs.defaults)
    _inspect.getargspec = _getargspec
# ---------------------------------------------------------------------------

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None

# ----------------------------
# Logging (same as cross-attention)
# ----------------------------
class Tee(io.TextIOBase):
    def __init__(self, *streams): self.streams = streams
    def write(self, data):
        for s in self.streams:
            try: s.write(data); s.flush()
            except Exception: pass
        return len(data)
    def flush(self):
        for s in self.streams:
            try: s.flush()
            except Exception: pass
    def isatty(self):
        for s in self.streams:
            try:
                if s.isatty(): return True
            except Exception: continue
        return False
    def fileno(self):
        for s in self.streams:
            try: return s.fileno()
            except Exception: continue
        raise OSError("No fileno available")
    def __getattr__(self, name):
        if self.streams: return getattr(self.streams[0], name)
        raise AttributeError(name)

LOG_FH = None
def start_text_logging(out_dir: str, prefix: str = "fusion", ts: str | None = None):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    ts = ts or datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = Path(out_dir) / f"{prefix}_{ts}.txt"
    fh = open(log_path, "a", buffering=1, encoding="utf-8")
    sys.stdout = Tee(sys.stdout, fh)
    sys.stderr = Tee(sys.stderr, fh)
    print("="*100)
    print(f"[RUN START] {ts} | log={log_path}")
    return str(log_path), fh, ts

import atexit
@atexit.register
def _close_log_file():
    global LOG_FH
    if LOG_FH is not None:
        try: LOG_FH.flush(); LOG_FH.close()
        except Exception: pass

# ----------------------------
# Utils (identical helpers)
# ----------------------------
def set_seed(seed: int = 42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def exists(p): return os.path.exists(p)

def make_label_map(labels):
    uniq = sorted(set(labels)); return {lbl: i for i, lbl in enumerate(uniq)}

def load_label_map(path):
    with open(path, "r") as f: return json.load(f)

def save_json(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f: json.dump(obj, f, indent=2)

def count_trainable_params(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def uniform_to_T(x: torch.Tensor, T: int) -> torch.Tensor:
    Tin = x.shape[0]
    if Tin == T: return x
    if Tin > T:
        idx = torch.linspace(0, Tin - 1, T).round().long()
        return x.index_select(0, idx)
    pad = torch.zeros(T - Tin, x.shape[1], dtype=x.dtype)
    return torch.cat([x, pad], dim=0)

# ---- TSN helpers (identical) ----
def tsn_snippet_indices(T_in: int, N: int, k: int, jitter: bool) -> torch.Tensor:
    if T_in <= 0: return torch.zeros(0, dtype=torch.long)
    bounds = torch.linspace(0, T_in, steps=N + 1)
    idxs = []
    for i in range(N):
        s = int(math.floor(bounds[i].item())); e = int(math.ceil(bounds[i+1].item()))
        seg_len = max(1, e - s); win = min(k, seg_len)
        start_lo = s; start_hi = max(s, e - win)
        start = (np.random.randint(start_lo, start_hi + 1) if jitter else (start_lo + start_hi) // 2)
        cont = torch.arange(start, start + win); cont = torch.clamp(cont, max=e - 1)
        if win < k: cont = torch.cat([cont, cont[-1].repeat(k - win)], dim=0)
        idxs.append(cont)
    return torch.cat(idxs, dim=0)

def tsn_sample(x: torch.Tensor, N: int, k: int, train_mode: bool) -> torch.Tensor:
    Tin = x.shape[0]
    if Tin == 0: return torch.zeros(N * k, x.shape[1], dtype=x.dtype)
    idx = tsn_snippet_indices(Tin, N, k, jitter=train_mode)
    return x.index_select(0, idx)

# ----------------------------
# Dataset (copied to ensure exact parity)
# ----------------------------
class FusionIndexDataset(Dataset):
    """
    Reads CSV rows and loads both modalities, aligned by TSN.
    Required columns: clip, label, path_vjepa, and path_skel/path_skeleton.
    """
    def __init__(self, csv_path: str, label2id: dict, T: int = 64,
                 decode_smpl: bool = False, smpl_model_dir: str | None = None,
                 center_root: bool = True, include_trans: bool = False, include_betas: bool = False,
                 cache_dir: str | None = None, drop_missing: bool = True,
                 tsn_segments: int = 0, tsn_snippet: int = 0, jitter: bool = False,
                 d_video_in: int = 1408):
        self.df = pd.read_csv(csv_path)
        col_skel = "path_skel" if "path_skel" in self.df.columns else ("path_skeleton" if "path_skeleton" in self.df.columns else None)
        needed = {"clip", "label", "path_vjepa", col_skel}
        if col_skel is None or not needed.issubset(set(self.df.columns)):
            raise ValueError(f"CSV must have columns including 'clip','label','path_vjepa','path_skel'. Got: {self.df.columns.tolist()}")
        rows = []
        for _, r in self.df.iterrows():
            pv, ps = str(r["path_vjepa"]), str(r[col_skel])
            if drop_missing and (not exists(pv) or not exists(ps) or pv == "" or ps == ""):
                continue
            rows.append({"clip": str(r["clip"]), "label": str(r["label"]), "path_vjepa": pv, "path_skel": ps})
        if len(rows) == 0:
            raise RuntimeError(f"No valid rows in {csv_path}. Check paths.")

        self.rows, self.label2id, self.T = rows, label2id, T
        self.tsn_segments, self.tsn_snippet, self.jitter = int(tsn_segments), int(tsn_snippet), bool(jitter)
        self._use_tsn = (self.tsn_segments > 0 and self.tsn_snippet > 0)
        self.decode_smpl, self.center_root = decode_smpl, center_root
        self.include_trans, self.include_betas = include_trans, include_betas
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir: self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.d_video_in = d_video_in

        self.smpl_layer = None
        if self.decode_smpl:
            try:
                import smplx
            except ImportError as e:
                raise ImportError("smplx not installed. `pip install smplx`.") from e
            if not smpl_model_dir or not Path(smpl_model_dir).exists():
                raise FileNotFoundError("Provide --smpl_model_dir pointing to SMPL model dir.")
            self.smpl_layer = smplx.create(
                model_path=str(Path(smpl_model_dir)),
                model_type='smpl', gender='neutral', num_betas=10, use_pca=False, ext='pkl'
            ).eval()

    def __len__(self): return len(self.rows)

    def _pick_track(self, d: dict) -> dict:
        if all(k in d for k in ("pose","trans","betas")): return d
        cand = []
        if isinstance(d, (list, tuple)):
            for it in d:
                if isinstance(it, dict) and "pose" in it: cand.append(it)
        elif isinstance(d, dict):
            tracks = d.get("tracks", [])
            if isinstance(tracks, (list, tuple)):
                for it in tracks:
                    if isinstance(it, dict) and "pose" in it: cand.append(it)
        if not cand: raise RuntimeError("Unexpected .pt structure for skeleton file.")
        cand.sort(key=lambda x: int(x["pose"].shape[0] if isinstance(x["pose"], torch.Tensor) else len(x["pose"])), reverse=True)
        return cand[0]

    def _decode_smpl_to_joints(self, pose: torch.Tensor, trans: torch.Tensor, betas: torch.Tensor) -> torch.Tensor:
        T = pose.shape[0]
        global_orient = pose[:, :3]; body_pose = pose[:, 3:72]
        if betas.ndim == 1: betas = betas.unsqueeze(0).expand(T, -1)
        if trans.ndim == 1: trans = trans.unsqueeze(0).expand(T, -1)
        global_orient = global_orient.float(); body_pose = body_pose.float()
        betas = betas.float(); trans = trans.float()
        chunks = []; BS = 512
        with torch.no_grad():
            for s in range(0, T, BS):
                e = min(T, s + BS)
                out = self.smpl_layer(betas=betas[s:e], global_orient=global_orient[s:e], body_pose=body_pose[s:e], transl=trans[s:e])
                chunks.append(out.joints)  # [b,J,3]
        joints = torch.cat(chunks, dim=0)
        if self.center_root: joints = joints - joints[:, [0], :]
        return joints

    def _load_skel_feat(self, path_skel: str) -> torch.Tensor:
        if self.cache_dir:
            key = f"{Path(path_skel).stem}__{'joints' if self.decode_smpl else 'params'}.pt"
            cache_path = self.cache_dir / key
            if cache_path.exists(): return torch.load(cache_path, map_location="cpu")
        d = torch.load(path_skel, map_location="cpu")
        d = self._pick_track(d)
        pose = torch.as_tensor(d.get("pose"), dtype=torch.float32)
        trans = torch.as_tensor(d.get("trans", torch.zeros((pose.shape[0],3))))
        betas = torch.as_tensor(d.get("betas", torch.zeros((pose.shape[0],10))))
        if self.decode_smpl:
            joints = self._decode_smpl_to_joints(pose, trans, betas)   # [T,J,3]
            feat = joints.reshape(joints.shape[0], -1)                 # [T,J*3]
        else:
            parts = [pose]
            if self.include_trans:
                parts.append(trans if trans.ndim == 2 else trans.unsqueeze(0).expand(pose.shape[0], -1))
            if self.include_betas:
                if betas.ndim == 1: betas = betas.unsqueeze(0).expand(pose.shape[0], -1)
                parts.append(betas)
            feat = torch.cat(parts, dim=1)
        if self.cache_dir:
            try: torch.save(feat, cache_path)
            except Exception: pass
        return feat

    def __getitem__(self, idx):
        rec = self.rows[idx]
        pv, ps = rec["path_vjepa"], rec["path_skel"]
        if not exists(pv): raise FileNotFoundError(pv)
        if not exists(ps): raise FileNotFoundError(ps)

        V = torch.load(pv, map_location="cpu")   # [Tv, Dv]
        if V.ndim != 2: raise RuntimeError(f"Expected [T,D], got {V.shape} for {pv}")
        S = self._load_skel_feat(ps)            # [Ts, Ds]

        if self._use_tsn:
            V = tsn_sample(V, N=self.tsn_segments, k=self.tsn_snippet, train_mode=self.jitter)
            S = tsn_sample(S, N=self.tsn_segments, k=self.tsn_snippet, train_mode=self.jitter)
        else:
            V = uniform_to_T(V, self.T); S = uniform_to_T(S, self.T)
        V = V.to(torch.float32); S = S.to(torch.float32)
        y = self.label2id[str(rec["label"])]
        return V, S, y

# ----------------------------
# Model: Gated Recurrent Fusion (baseline)
# ----------------------------
class GatedRecurrentFusionModel(nn.Module):
    def __init__(self, d_video_in=1408, d_skel_in=85, hidden_dim=512, num_layers=4, dropout=0.1, num_classes=14):
        super().__init__()
        self.v_proj = nn.Linear(d_video_in, hidden_dim)
        self.s_proj = nn.Linear(d_skel_in,  hidden_dim)
        self.v_gru = nn.GRU(hidden_dim, hidden_dim, num_layers, batch_first=True,
                            dropout=dropout if num_layers > 1 else 0.0)
        self.s_gru = nn.GRU(hidden_dim, hidden_dim, num_layers, batch_first=True,
                            dropout=dropout if num_layers > 1 else 0.0)
        # gating over concatenated streams
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.Sigmoid()
        )
        self.fusion_gru = nn.GRU(hidden_dim, hidden_dim, num_layers, batch_first=True,
                                 dropout=dropout if num_layers > 1 else 0.0)
        self.classifier = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, num_classes))

        # init
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None: nn.init.zeros_(m.bias)

    def forward(self, V, S):
        # V,S: [B,T,Dv/Ds]
        v = self.v_proj(V); s = self.s_proj(S)
        v_out, _ = self.v_gru(v)  # [B,T,H]
        s_out, _ = self.s_gru(s)  # [B,T,H]
        z = self.gate(torch.cat([v_out, s_out], dim=-1))  # [B,T,H] in [0,1]
        fused = z * v_out + (1.0 - z) * s_out             # [B,T,H]
        _, hT = self.fusion_gru(fused)                    # [num_layers,B,H]
        logits = self.classifier(hT[-1])                  # [B,C]
        return logits

# ----------------------------
# Eval helpers (byte-for-byte behavior with fusion trainer)
# ----------------------------
@torch.inference_mode()
def evaluate(model, loader, device, num_classes, amp=False, label2id=None, bg_name='[OP000] No action'):
    model.eval()
    all_logits, all_targets = [], []
    ctx = torch.amp.autocast if torch.cuda.is_available() else torch.cpu.amp.autocast
    with ctx(device_type="cuda", enabled=amp and torch.cuda.is_available()):
        for V, S, y in loader:
            V = V.to(device, non_blocking=True); S = S.to(device, non_blocking=True); y = y.to(device, non_blocking=True)
            all_logits.append(model(V, S).detach().cpu()); all_targets.append(y.detach().cpu())
    logits = torch.cat(all_logits, dim=0)
    probs = torch.softmax(logits, dim=-1).numpy()
    targets = torch.cat(all_targets, dim=0).numpy().astype(int)
    preds = probs.argmax(axis=1)
    top1 = float((preds == targets).mean())
    k_eff = min(5, probs.shape[1])
    top5 = float(top_k_accuracy_score(targets, probs, k=k_eff, labels=list(range(probs.shape[1]))))
    y_true = np.zeros((targets.shape[0], probs.shape[1]), dtype=np.int32)
    y_true[np.arange(targets.shape[0]), targets] = 1
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        mAP_all = float(average_precision_score(y_true, probs, average="macro"))
    support = y_true.sum(axis=0) > 0
    mAP_valid = float(average_precision_score(y_true[:, support], probs[:, support], average="macro")) if support.any() else float("nan")
    macroP, macroR, macroF1, _ = precision_recall_fscore_support(targets, preds, average="macro", zero_division=0)
    mAP_noBG = None
    if label2id and (bg_name in label2id):
        bg_id = label2id[bg_name]
        mask = support.copy()
        if 0 <= bg_id < len(mask): mask[bg_id] = False
        if mask.any():
            mAP_noBG = float(average_precision_score(y_true[:, mask], probs[:, mask], average="macro"))
    return {
        "top1": top1, "top5": top5, "mAP_all": mAP_all, "mAP_valid": mAP_valid,
        "mAP_noBG": mAP_noBG, "macroP": float(macroP), "macroR": float(macroR), "macroF1": float(macroF1)
    }

def _save_confusion_csv_png(y_true, y_pred, class_names, out_dir: Path):
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(len(class_names)), normalize='true')
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "confusion_matrix_normalized.csv"
    with open(csv_path, "w") as f:
        f.write(",".join(["class"] + class_names) + "\n")
        for i, row in enumerate(cm):
            f.write(",".join([class_names[i]] + [f"{x:.6f}" for x in row]) + "\n")
    print(f"[FINAL] Saved confusion matrix CSV -> {csv_path}")
    if plt is not None and len(class_names) <= 100:
        h = max(6, min(18, int(len(class_names) * 0.20)))
        fig = plt.figure(figsize=(h, h), dpi=150); ax = plt.gca()
        im = ax.imshow(cm, interpolation='nearest', cmap='viridis'); plt.colorbar(im, fraction=0.046, pad=0.04)
        ax.set_xticks(np.arange(len(class_names))); ax.set_yticks(np.arange(len(class_names)))
        ax.set_xticklabels(class_names, rotation=90, fontsize=6); ax.set_yticklabels(class_names, fontsize=6)
        ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title("Normalized Confusion Matrix")
        plt.tight_layout(); png_path = out_dir / "confusion_matrix_normalized.png"; plt.savefig(png_path); plt.close(fig)
        print(f"[FINAL] Saved confusion matrix PNG -> {png_path}")

@torch.inference_mode()
def final_evaluation_and_report(model, loader, device, num_classes, label2id, out_dir, amp=False):
    model.eval()
    all_logits, all_targets = [], []
    ctx = torch.amp.autocast if torch.cuda.is_available() else torch.cpu.amp.autocast
    with ctx(device_type="cuda", enabled=amp and torch.cuda.is_available()):
        for V, S, y in loader:
            V = V.to(device, non_blocking=True); S = S.to(device, non_blocking=True); y = y.to(device, non_blocking=True)
            all_logits.append(model(V, S).detach().cpu()); all_targets.append(y.detach().cpu())
    logits = torch.cat(all_logits, dim=0); probs = torch.softmax(logits, dim=-1).numpy()
    targets = torch.cat(all_targets, dim=0).numpy().astype(int); preds = probs.argmax(axis=1)
    top1 = float((preds == targets).mean()); k_eff = min(5, probs.shape[1])
    top5 = float(top_k_accuracy_score(targets, probs, k=k_eff, labels=list(range(probs.shape[1]))))
    y_true = np.zeros_like(probs, dtype=np.int32); y_true[np.arange(targets.shape[0]), targets] = 1
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        mAP_all = float(average_precision_score(y_true, probs, average="macro"))
    support = (y_true.sum(axis=0) > 0)
    mAP_valid = float(average_precision_score(y_true[:, support], probs[:, support], average="macro")) if support.any() else float("nan")
    macroP, macroR, macroF1, _ = precision_recall_fscore_support(targets, preds, average="macro", zero_division=0)
    P_c, R_c, F1_c, Supp_c = precision_recall_fscore_support(targets, preds, average=None, labels=np.arange(probs.shape[1]), zero_division=0)
    AP_c = []
    for c in range(probs.shape[1]):
        if support[c]: AP_c.append(float(average_precision_score(y_true[:, c], probs[:, c])))
        else: AP_c.append(float("nan"))
    id2label = {v: k for k, v in label2id.items()}; class_names = [id2label.get(i, str(i)) for i in range(probs.shape[1])]
    per_class_df = pd.DataFrame({ "class_id": np.arange(probs.shape[1]), "class_name": class_names,
                                  "precision": P_c, "recall": R_c, "f1": F1_c, "AP": AP_c, "support": Supp_c })
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    per_class_csv = out_dir / "per_class_metrics.csv"; per_class_df.to_csv(per_class_csv, index=False)
    print(f"[FINAL] Saved per-class metrics -> {per_class_csv}")
    _save_confusion_csv_png(targets, preds, class_names, out_dir)
    print("\n================ FINAL EVALUATION (best checkpoint) ================")
    print(f"Top-1 Acc: {top1:.4f}  |  Top-5 Acc: {top5:.4f}")
    print(f"Macro mAP (all):   {mAP_all:.4f}")
    print(f"Macro mAP (valid): {mAP_valid:.4f}")
    print(f"Macro Precision:   {macroP:.4f}")
    print(f"Macro Recall:      {macroR:.4f}")
    print(f"Macro F1-Score:    {macroF1:.4f}")
    return {"top1": top1, "top5": top5, "mAP_all": mAP_all, "mAP_valid": mAP_valid, "macroP": float(macroP), "macroR": float(macroR), "macroF1": float(macroF1)}

@torch.no_grad()
def measure_inference_fps(model, loader, device, amp: bool = True):
    model.eval(); total_frames = 0; start = None
    ctx = torch.amp.autocast if torch.cuda.is_available() else torch.cpu.amp.autocast
    warmup = 5
    for i, (V, S, _) in enumerate(loader):
        V = V.to(device, non_blocking=True); S = S.to(device, non_blocking=True)
        if i < warmup:
            with ctx(device_type="cuda", enabled=amp and torch.cuda.is_available()):
                _ = model(V, S)
            continue
        if start is None:
            if torch.cuda.is_available(): torch.cuda.synchronize()
            start = time.time()
        with ctx(device_type="cuda", enabled=amp and torch.cuda.is_available()):
            _ = model(V, S)
        B, T = V.shape[0], V.shape[1]; total_frames += B * T
    if start is None: return 0.0
    if torch.cuda.is_available(): torch.cuda.synchronize()
    return float(total_frames / max(time.time() - start, 1e-9))

# ----------------------------
# Occlusion study (mirrors fusion trainer; no attention export)
# ----------------------------
@torch.inference_mode()
def _eval_on_csv(model, csv_path, label2id, args):
    ds = FusionIndexDataset(
        csv_path, label2id,
        T=args.T,
        decode_smpl=args.decode_smpl, smpl_model_dir=args.smpl_model_dir,
        center_root=not args.no_center_root,
        include_trans=args.include_trans, include_betas=args.include_betas,
        cache_dir=args.cache_dir, drop_missing=True,
        tsn_segments=getattr(args, "tsn_segments", 0),
        tsn_snippet=getattr(args, "tsn_snippet", 0),
        d_video_in=args.d_video_in
    )
    loader = DataLoader(ds, batch_size=max(1, args.batch_size*2), shuffle=False,
                        num_workers=max(0, int(args.num_workers)), pin_memory=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return evaluate(model, loader, device, num_classes=len(label2id),
                    amp=args.amp, label2id=label2id)

def run_occlusion_study(args):
    ckpt_path = Path(args.ckpt) if args.ckpt else Path(args.log_dir) / (args.run_name or "") / "best_fusion.ckpt"
    if not ckpt_path.exists(): raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    label2id = ckpt.get("label2id", None)
    if label2id is None:
        if not (args.labelmap and Path(args.labelmap).exists()):
            raise RuntimeError("label2id missing in ckpt; please pass --labelmap")
        label2id = load_label_map(args.labelmap)

    margs = ckpt.get("args", {})
    d_model = margs.get("d_model", args.d_model)
    depth   = margs.get("depth",   args.depth)
    dropout = margs.get("dropout", args.dropout)
    T       = margs.get("T",       args.T)
    d_video_in = margs.get("d_video_in", args.d_video_in)

    tmp_ds = FusionIndexDataset(args.occ_low_csv, label2id, T=T,
                                decode_smpl=args.decode_smpl, smpl_model_dir=args.smpl_model_dir,
                                center_root=not args.no_center_root,
                                include_trans=args.include_trans, include_betas=args.include_betas,
                                cache_dir=args.cache_dir, drop_missing=True,
                                tsn_segments=getattr(args, "tsn_segments", 0),
                                tsn_snippet=getattr(args, "tsn_snippet", 0),
                                d_video_in=d_video_in)
    _, Si, _ = tmp_ds[0]
    d_skel_in = int(Si.shape[1])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = GatedRecurrentFusionModel(
        d_video_in=d_video_in, d_skel_in=d_skel_in,
        hidden_dim=d_model, num_layers=depth, dropout=dropout, num_classes=len(label2id)
    ).to(device)
    model.load_state_dict(ckpt["model"]); model.eval()

    low_metrics  = _eval_on_csv(model, args.occ_low_csv,  label2id, args)
    high_metrics = _eval_on_csv(model, args.occ_high_csv, label2id, args)

    low_top1  = 100.0 * float(low_metrics["top1"])
    high_top1 = 100.0 * float(high_metrics["top1"])
    delta_pp  = high_top1 - low_top1

    print("\n=== Occlusion Study (Gated GRU Fusion) ===")
    print(f"Low  ({Path(args.occ_low_csv).name}):  Top-1 = {low_top1:.1f}%")
    print(f"High ({Path(args.occ_high_csv).name}): Top-1 = {high_top1:.1f}%")
    print(f"Δ (High - Low): {delta_pp:+.1f} pp\n")

    out_dir = ckpt_path.parent
    payload = {
        "model": "Fusion (Gated GRU)",
        "low_csv":  str(args.occ_low_csv),
        "high_csv": str(args.occ_high_csv),
        "low_top1":  low_top1,
        "high_top1": high_top1,
        "delta_pp":  delta_pp,
        "low_metrics":  low_metrics,
        "high_metrics": high_metrics,
        "ckpt": str(ckpt_path)
    }
    with open(out_dir / "occ_eval_fusion.json", "w") as f:
        json.dump(payload, f, indent=2)
    with open(out_dir / "occ_eval_fusion.md", "w") as f:
        f.write("| Model | Top-1 (Low) | Top-1 (High) | Δ (pp) |\n|---|---:|---:|---:|\n")
        f.write(f"| Fusion (Gated GRU) | {low_top1:.1f}% | {high_top1:.1f}% | {delta_pp:+.1f} |\n")
    return payload

# ----------------------------
# Training (mirrors cross-attention flow)
# ----------------------------
def train(args):
    for p, flag in [(args.train_csv, "--train_csv"), (args.val_csv, "--val_csv")]:
        if not exists(p): raise FileNotFoundError(f"{flag} not found: {p}")
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Speed knobs
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision('high')
        try:
            from torch.backends.cuda import sdp_kernel
            sdp_kernel(enable_flash=True, enable_mem_efficient=True, enable_math=False)
            print("[SDPA] flash=True mem_efficient=True math=False")
        except Exception:
            pass

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = args.run_name or ts
    args.run_dir = str(Path(args.log_dir) / run_id)
    Path(args.run_dir).mkdir(parents=True, exist_ok=True)
    txt_dir = args.txt_log_dir or (Path(args.log_dir).parent / "runs" / "fusion")
    global LOG_FH
    _, LOG_FH, _ = start_text_logging(txt_dir, prefix="fusion", ts=run_id)

    # Label map
    train_df = pd.read_csv(args.train_csv)
    train_labels = [str(x) for x in train_df["label"].tolist()]
    if args.labelmap and exists(args.labelmap):
        label2id = load_label_map(args.labelmap)
    else:
        label2id = make_label_map(train_labels)
        if args.labelmap: save_json(label2id, args.labelmap)
    num_classes = len(label2id)
    print(f"Classes: {num_classes}")

    # TSN normalization (normalize T to N*k when TSN is used)
    if getattr(args, "tsn_segments", 0) and args.tsn_segments > 0:
        if not getattr(args, "tsn_snippet", 0) or args.tsn_snippet <= 0:
            args.tsn_snippet = max(1, args.T // args.tsn_segments)
        T_eff = args.tsn_segments * args.tsn_snippet
        if T_eff != args.T:
            print(f"[TSN] Adjusting T from {args.T} -> {T_eff} to match N*k.")
            args.T = T_eff
        print(f"[TSN] Using N={args.tsn_segments}, k={args.tsn_snippet}, T={args.T} (train=jitter, val=center)")

    # Datasets / loaders
    base_ds_kwargs = dict(
        T=args.T, decode_smpl=args.decode_smpl, smpl_model_dir=args.smpl_model_dir,
        center_root=not args.no_center_root, include_trans=args.include_trans, include_betas=args.include_betas,
        cache_dir=args.cache_dir, drop_missing=True,
        tsn_segments=getattr(args, "tsn_segments", 0), tsn_snippet=getattr(args, "tsn_snippet", 0),
        d_video_in=args.d_video_in
    )
    train_ds = FusionIndexDataset(args.train_csv, label2id, jitter=True,  **base_ds_kwargs)
    val_ds   = FusionIndexDataset(args.val_csv,   label2id, jitter=False, **base_ds_kwargs)

    Vi, Si, _ = train_ds[0]; d_skel_in = int(Si.shape[1])
    print(f"Inferred d_skel_in={d_skel_in} | d_video_in={args.d_video_in}")

    num_workers = max(0, int(args.num_workers)); prefetch_factor = int(getattr(args, "prefetch_factor", 2))
    loader_kwargs = dict(num_workers=num_workers, pin_memory=True)
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True; loader_kwargs["prefetch_factor"] = prefetch_factor
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  drop_last=True, **loader_kwargs)
    val_loader   = DataLoader(val_ds,   batch_size=max(1, args.batch_size*2), shuffle=False, **loader_kwargs)

    # Model (map transformer params to GRU params)
    model = GatedRecurrentFusionModel(
        d_video_in=args.d_video_in, d_skel_in=d_skel_in,
        hidden_dim=args.d_model, num_layers=args.depth, dropout=args.dropout, num_classes=num_classes
    ).to(device)
    params_m = count_trainable_params(model) / 1e6
    print(f"Trainable parameters: {params_m:.3f}M")

    # Optim & schedule
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, fused=torch.cuda.is_available())
    total_steps = args.epochs * max(1, math.ceil(len(train_loader)))
    warmup_steps = int(0.05 * total_steps)
    def lr_schedule(step):
        if step < warmup_steps: return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scaler = torch.amp.GradScaler('cuda', enabled=args.amp and torch.cuda.is_available())
    criterion = nn.CrossEntropyLoss()

    out_dir = Path(args.run_dir); out_dir.mkdir(parents=True, exist_ok=True)
    best_map = -1.0; global_step = 0; epoch_rows = []

    # Train loop
    for epoch in range(1, args.epochs+1):
        model.train(); epoch_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", ncols=100, dynamic_ncols=True, leave=True, file=sys.stdout, mininterval=0.1)
        for V, S, y in pbar:
            V = V.to(device, non_blocking=True); S = S.to(device, non_blocking=True); y = y.to(device, non_blocking=True)
            for g in optim.param_groups: g["lr"] = args.lr * lr_schedule(global_step)
            optim.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=args.amp and torch.cuda.is_available()):
                logits = model(V, S); loss = criterion(logits, y)
            scaler.scale(loss).backward()
            if args.grad_clip is not None:
                scaler.unscale_(optim); nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optim); scaler.update()
            global_step += 1; epoch_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}", "step": global_step})

        metrics = evaluate(model, val_loader, device, num_classes, amp=args.amp, label2id=label2id)
        avg_loss = epoch_loss / max(len(train_loader), 1)
        no_bg = metrics.get("mAP_noBG", None)
        no_bg_str = f"{no_bg:.4f}" if (isinstance(no_bg, float) and np.isfinite(no_bg)) else "n/a"
        top5 = metrics.get("top5", float("nan")); top5_str = f"{top5:.4f}" if (isinstance(top5, float) and np.isfinite(top5)) else "n/a"
        print(f"[Epoch {epoch}] train_loss={avg_loss:.4f}  "
              f"val_top1={metrics['top1']:.4f}  val_top5={top5_str}  "
              f"val_mAP_all={metrics['mAP_all']:.4f}  val_mAP_valid={metrics['mAP_valid']:.4f}  "
              f"val_macroP={metrics['macroP']:.4f}  val_macroR={metrics['macroR']:.4f}  val_macroF1={metrics['macroF1']:.4f}  "
              f"val_mAP_noBG={no_bg_str}")

        row = {
            "epoch": epoch, "train_loss": avg_loss,
            "val_top1": metrics["top1"], "val_top5": metrics.get("top5", float("nan")),
            "val_mAP_all": metrics["mAP_all"], "val_mAP_valid": metrics["mAP_valid"],
            "val_macroP": metrics["macroP"], "val_macroR": metrics["macroR"], "val_macroF1": metrics["macroF1"],
            "val_mAP_noBG": metrics.get("mAP_noBG", None),
        }
        epoch_rows.append(row)
        pd.DataFrame(epoch_rows).to_csv(Path(args.run_dir) / "epoch_metrics.csv", index=False)

        if metrics["mAP_valid"] > best_map:
            best_map = metrics["mAP_valid"]
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
                "params_m": params_m,
            }
            torch.save(ckpt, out_dir / "best_fusion.ckpt")  # same filename as cross-attention
            print(f"Saved best checkpoint -> {out_dir/'best_fusion.ckpt'} (mAP_valid {best_map:.4f})")

    # Final evaluation
    best_path = out_dir / "best_fusion.ckpt"
    if best_path.exists():
        print(f"\n[FINAL] Loading best checkpoint: {best_path}")
        ckpt = torch.load(best_path, map_location="cpu")
        model.load_state_dict(ckpt["model"]); model.to(device)

        final_metrics = final_evaluation_and_report(model, val_loader, device, num_classes, label2id, out_dir, amp=args.amp)
        fps = measure_inference_fps(model, val_loader, device, amp=args.amp)
        print(f"[FINAL] Inference throughput: {fps:.2f} FPS")

        run_dir = Path(args.run_dir)
        final_txt  = run_dir / "final_eval.txt"
        final_json = run_dir / "final_eval.json"
        headline = (
            f"GatedFusion (GRU) | Top-1 {final_metrics['top1']*100:5.2f}% | "
            f"Top-5 {final_metrics['top5']*100:5.2f}% | Macro mAP {final_metrics['mAP_valid']*100:5.2f}% | "
            f"Macro F1 {final_metrics['macroF1']*100:5.2f}% | Params {params_m:.3f}M | FPS {fps:.2f}"
        )
        with open(final_txt, "w") as f:
            f.write("================ FINAL EVALUATION (best checkpoint) ================\n")
            f.write(f"Top-1 Acc: {final_metrics['top1']:.4f}  |  Top-5 Acc: {final_metrics['top5']:.4f}\n")
            f.write(f"Macro mAP (all):   {final_metrics['mAP_all']:.4f}\n")
            f.write(f"Macro mAP (valid): {final_metrics['mAP_valid']:.4f}\n")
            f.write(f"Macro Precision:   {final_metrics.get('macroP', float('nan')):.4f}\n")
            f.write(f"Macro Recall:      {final_metrics.get('macroR', float('nan')):.4f}\n")
            f.write(f"Macro F1-Score:    {final_metrics['macroF1']:.4f}\n")
            f.write(f"[FINAL] Inference throughput: {fps:.2f} FPS\n\n")
            f.write("=== Headline Row (copy into paper table) ===\n")
            f.write(headline + "\n")
        with open(final_json, "w") as jf:
            json.dump({**final_metrics, "fps": float(fps), "params_m": float(params_m),
                       "run_dir": str(run_dir), "best_ckpt": str(best_path)}, jf, indent=2)
        print("\n=== Headline Row (copy into paper table) ==="); print(headline)
        if args.dump_attention and args.dump_attention > 0:
            print("[NOTE] --dump_attention requested, but gated GRU model has no attention maps; skipping export.")
    else:
        print("[FINAL] Best checkpoint not found; skipping final evaluation.")

# ----------------------------
# Main / CLI (mirrors cross-attention)
# ----------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    # training CSVs
    ap.add_argument("--train_csv", type=str, default=None)
    ap.add_argument("--val_csv",   type=str, default=None)

    # logging / runs
    ap.add_argument("--log_dir",     type=str, default="/workspace/har-fusion/runs/fusion")
    ap.add_argument("--txt_log_dir", type=str, default="/workspace/trainheads/runs/fusion")
    ap.add_argument("--run_name",    type=str, default=None)
    ap.add_argument("--labelmap",    type=str, default="/workspace/har-fusion/runs/fusion/label2id.json")

    # data/model hyperparams (same names; map internally where needed)
    ap.add_argument("--T",           type=int,   default=64)
    ap.add_argument("--d_video_in",  type=int,   default=1408)
    ap.add_argument("--d_model",     type=int,   default=512)  # -> GRU hidden_dim
    ap.add_argument("--n_heads",     type=int,   default=8)    # accepted, unused
    ap.add_argument("--depth",       type=int,   default=4)    # -> GRU num_layers
    ap.add_argument("--ff_mult",     type=int,   default=4)    # accepted, unused
    ap.add_argument("--dropout",     type=float, default=0.1)

    # skeleton decoding flags
    ap.add_argument("--decode_smpl",     action="store_true")
    ap.add_argument("--smpl_model_dir",  type=str, default=None)
    ap.add_argument("--no_center_root",  action="store_true")
    ap.add_argument("--include_trans",   action="store_true")
    ap.add_argument("--include_betas",   action="store_true")
    ap.add_argument("--cache_dir",       type=str, default=None)

    # TSN
    ap.add_argument("--tsn_segments", type=int, default=8)
    ap.add_argument("--tsn_snippet",  type=int, default=8)

    # train
    ap.add_argument("--epochs",       type=int,   default=30)
    ap.add_argument("--batch_size",   type=int,   default=128)
    ap.add_argument("--num_workers",  type=int,   default=16)
    ap.add_argument("--prefetch_factor", type=int, default=4)
    ap.add_argument("--lr",           type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=0.05)
    ap.add_argument("--grad_clip",    type=float, default=1.0)
    ap.add_argument("--amp",          action="store_true")
    ap.add_argument("--seed",         type=int,   default=42)

    # attention export (accepted, ignored)
    ap.add_argument("--dump_attention", type=int, default=0)
    ap.add_argument("--attn_layers",   type=str, default="")
    ap.add_argument("--disable_self_attn", action="store_true")  # accepted, ignored

    # Occlusion study
    ap.add_argument("--occ_eval", action="store_true")
    ap.add_argument("--occ_low_csv",  type=str, default=None)
    ap.add_argument("--occ_high_csv", type=str, default=None)
    ap.add_argument("--ckpt",         type=str, default=None)

    args = ap.parse_args()

    if args.occ_eval:
        missing = [k for k in ("occ_low_csv", "occ_high_csv", "ckpt") if not getattr(args, k, None)]
        if missing:
            raise SystemExit(f"--occ_eval requires: {', '.join('--'+m for m in missing)}")
        run_occlusion_study(args)
    else:
        if not args.train_csv or not args.val_csv:
            raise SystemExit("Missing --train_csv and/or --val_csv (or pass --occ_eval to run the ablation).")
        train(args)
