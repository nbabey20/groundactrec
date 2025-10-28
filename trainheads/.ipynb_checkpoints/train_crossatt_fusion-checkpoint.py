
#!/usr/bin/env python
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
# --- Py3.11 / NumPy 2.0 compatibility shims for chumpy/SMPL pickles ---

# 1) restore deprecated numpy aliases used by chumpy (NumPy >= 2.0 removed them)
import numpy as _np
if not hasattr(_np, "bool"):
    _np.bool = _np.bool_          # dtype alias
if not hasattr(_np, "int"):
    _np.int = int
if not hasattr(_np, "float"):
    _np.float = float
if not hasattr(_np, "complex"):
    _np.complex = complex
if not hasattr(_np, "object"):
    _np.object = object
if not hasattr(_np, "unicode"):
    _np.unicode = str
if not hasattr(_np, "str"):
    _np.str = str

# 2) restore inspect.getargspec expected by chumpy (removed in Py3.11)
import inspect as _inspect
if not hasattr(_inspect, "getargspec"):
    from collections import namedtuple as _namedtuple
    def _getargspec(func):
        fs = _inspect.getfullargspec(func)
        ArgSpec = _namedtuple("ArgSpec", "args varargs keywords defaults")
        return ArgSpec(fs.args, fs.varargs, fs.varkw, fs.defaults)
    _inspect.getargspec = _getargspec

# ----------------------------------------------------------------------

# === Occlusion-study helpers (add anywhere above main) =====================

@torch.inference_mode()
def _eval_on_csv(model, csv_path, label2id, args):
    """Build a val loader from a CSV and run evaluate()."""
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
    """
    Evaluate a trained Fusion checkpoint on low- and high-occlusion CSVs,
    print a small table, and write JSON/MD artifacts.
    """
    # --- load checkpoint ---
    ckpt_path = Path(args.ckpt) if args.ckpt else Path(args.log_dir) / (args.run_name or "") / "best_fusion.ckpt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location="cpu")
    label2id = ckpt.get("label2id", None)
    if label2id is None:
        # fall back to labelmap path if provided
        if not (args.labelmap and Path(args.labelmap).exists()):
            raise RuntimeError("label2id missing in ckpt; please pass --labelmap pointing to label2id.json")
        label2id = load_label_map(args.labelmap)

    # --- rebuild model with the ckpt args ---
    margs = ckpt.get("args", {})
    d_model = margs.get("d_model", args.d_model)
    n_heads = margs.get("n_heads", args.n_heads)
    depth   = margs.get("depth",   args.depth)
    ff_mult = margs.get("ff_mult", args.ff_mult)
    dropout = margs.get("dropout", args.dropout)
    T       = margs.get("T", args.T)
    d_video_in = margs.get("d_video_in", args.d_video_in)
    enable_self = not margs.get("disable_self_attn", False)

    # we need d_skel_in: infer quickly from the low CSV
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
    model = FusionXAttnModel(
        d_video_in=d_video_in, d_skel_in=d_skel_in, d_model=d_model, n_heads=n_heads,
        depth=depth, ff_mult=ff_mult, dropout=dropout, num_classes=len(label2id),
        T=T, enable_self_attn=enable_self
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    # --- evaluate on both splits ---
    low_metrics  = _eval_on_csv(model, args.occ_low_csv,  label2id, args)
    high_metrics = _eval_on_csv(model, args.occ_high_csv, label2id, args)

    low_top1  = 100.0 * float(low_metrics["top1"])
    high_top1 = 100.0 * float(high_metrics["top1"])
    delta_pp  = high_top1 - low_top1  # absolute Δ in percentage points (High - Low)

    # ✅ CHANGE START: Extract the new metrics
    low_map   = 100.0 * float(low_metrics["mAP_valid"])
    high_map  = 100.0 * float(high_metrics["mAP_valid"])
    low_f1    = 100.0 * float(low_metrics["macroF1"])
    high_f1   = 100.0 * float(high_metrics["macroF1"])
    # ✅ CHANGE END

    # --- pretty print table row ---
    # ✅ CHANGE START: Update console output
    print("\n=== Occlusion Study (Fusion) ===")
    print(f"Low  ({Path(args.occ_low_csv).name}):  Top-1={low_top1:.1f}% | Macro mAP={low_map:.1f}% | Macro F1={low_f1:.1f}%")
    print(f"High ({Path(args.occ_high_csv).name}): Top-1={high_top1:.1f}% | Macro mAP={high_map:.1f}% | Macro F1={high_f1:.1f}%")
    print(f"Δ (High - Low Top-1): {delta_pp:+.1f} pp\n")
    # ✅ CHANGE END


        # --- build datasets we can index for attention export ---
    base_ds_kwargs = dict(
        T=T,
        decode_smpl=args.decode_smpl, smpl_model_dir=args.smpl_model_dir,
        center_root=not args.no_center_root,
        include_trans=args.include_trans, include_betas=args.include_betas,
        cache_dir=args.cache_dir, drop_missing=True,
        tsn_segments=getattr(args, "tsn_segments", 0),
        tsn_snippet=getattr(args, "tsn_snippet", 0),
        d_video_in=d_video_in,
    )
    ds_low  = FusionIndexDataset(args.occ_low_csv,  label2id, jitter=False, **base_ds_kwargs)
    ds_high = FusionIndexDataset(args.occ_high_csv, label2id, jitter=False, **base_ds_kwargs)

    # --- Optional: export attention maps for both splits ---
    if getattr(args, "dump_attention", 0) and args.dump_attention > 0:

        out_root = Path(args.ckpt).parent / f"occ_eval_attn_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        out_high = out_root / "high"
        out_low  = out_root / "low"
        out_high.mkdir(parents=True, exist_ok=True)
        out_low.mkdir(parents=True, exist_ok=True)

        print(f"[OCC/ATTN] Exporting {args.dump_attention} samples per split. Layers={args.attn_layers or 'all'}")
        export_attention_maps(model, ds_high, device, out_high, num_samples=int(args.dump_attention), layer_ids=args.attn_layers)
        export_attention_maps(model, ds_low, device, out_low, num_samples=int(args.dump_attention), layer_ids=args.attn_layers)
        print(f"[OCC/ATTN] Done. See: {out_root}")


    # --- save artifacts next to the checkpoint directory ---
    out_dir = ckpt_path.parent
    # ✅ CHANGE START: Update the JSON payload
    payload = {
        "model": "Fusion (X-Attn)",
        "low_csv":  str(args.occ_low_csv),
        "high_csv": str(args.occ_high_csv),
        "low_top1":  low_top1,
        "high_top1": high_top1,
        "delta_pp":  delta_pp,
        "low_map": low_map,
        "high_map": high_map,
        "low_f1": low_f1,
        "high_f1": high_f1,
        "low_metrics":  low_metrics,
        "high_metrics": high_metrics,
        "ckpt": str(ckpt_path)
    }
    # ✅ CHANGE END
    with open(out_dir / "occ_eval_fusion.json", "w") as f:
        json.dump(payload, f, indent=2)

    # ✅ CHANGE START: Update the Markdown table
    with open(out_dir / "occ_eval_fusion.md", "w") as f:
        f.write("| Model | Top-1 (Low) | Macro mAP (Low) | Macro F1 (Low) | Top-1 (High) | Macro mAP (High) | Macro F1 (High) | Δ (pp) |\n")
        f.write("|---|---:|---:|---:|---:|---:|---:|---:|\n")
        f.write(f"| Fusion (X-Attn) | {low_top1:.1f}% | {low_map:.1f}% | {low_f1:.1f}% | {high_top1:.1f}% | {high_map:.1f}% | {high_f1:.1f}% | {delta_pp:+.1f} |\n")
    # ✅ CHANGE END

    return payload


try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None

# ----------------------------
# Logging (tee to TXT)
# ----------------------------
class Tee(io.TextIOBase):
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            try:
                s.write(data)
                s.flush()
            except Exception:
                pass
        return len(data)

    def flush(self):
        for s in self.streams:
            try:
                s.flush()
            except Exception:
                pass

    # report TTY capability if any underlying stream is a TTY
    def isatty(self):
        for s in self.streams:
            try:
                if s.isatty():
                    return True
            except Exception:
                continue
        return False

    # let tqdm (and others) grab a real fd if possible
    def fileno(self):
        for s in self.streams:
            try:
                return s.fileno()
            except Exception:
                continue
        raise OSError("No fileno available")

    # delegate other attributes (e.g., encoding) to the first stream
    def __getattr__(self, name):
        if self.streams:
            return getattr(self.streams[0], name)
        raise AttributeError(name)


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

# ✅ Add this near the top (after imports / start_text_logging definition)
LOG_FH = None

import atexit
@atexit.register
def _close_log_file():
    global LOG_FH
    if LOG_FH is not None:
        try:
            LOG_FH.flush()
            LOG_FH.close()
        except Exception:
            pass

# ----------------------------
# Utils
# ----------------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
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

# ---- TSN helpers ----
def tsn_snippet_indices(T_in: int, N: int, k: int, jitter: bool) -> torch.Tensor:
    if T_in <= 0:
        return torch.zeros(0, dtype=torch.long)
    bounds = torch.linspace(0, T_in, steps=N + 1)
    idxs = []
    for i in range(N):
        s = int(math.floor(bounds[i].item()))
        e = int(math.ceil(bounds[i + 1].item()))
        seg_len = max(1, e - s)
        win = min(k, seg_len)
        start_lo = s; start_hi = max(s, e - win)
        start = (np.random.randint(start_lo, start_hi + 1) if jitter else (start_lo + start_hi) // 2)
        cont = torch.arange(start, start + win)
        cont = torch.clamp(cont, max=e - 1)
        if win < k:
            cont = torch.cat([cont, cont[-1].repeat(k - win)], dim=0)
        idxs.append(cont)
    return torch.cat(idxs, dim=0)

def tsn_sample(x: torch.Tensor, N: int, k: int, train_mode: bool) -> torch.Tensor:
    Tin = x.shape[0]
    if Tin == 0:
        return torch.zeros(N * k, x.shape[1], dtype=x.dtype)
    idx = tsn_snippet_indices(Tin, N, k, jitter=train_mode)
    return x.index_select(0, idx)

# ----------------------------
# Dataset
# ----------------------------
class FusionIndexDataset(Dataset):
    """
    Reads a CSV row and loads both modalities, aligned by TSN.
    Required CSV columns: clip, label, path_vjepa, and path_skel (or path_skeleton).
    Skeleton decoding:
      - raw params: concat pose(72) + optional trans(3) + optional betas(10)
      - decoded joints (if --decode_smpl): run SMPL to get joints [T,J,3] and flatten to [T,J*3]
    """
    def __init__(self, csv_path: str, label2id: dict, T: int = 64,
                 # skeleton decoding
                 decode_smpl: bool = False, smpl_model_dir: str | None = None,
                 center_root: bool = True, include_trans: bool = False, include_betas: bool = False,
                 cache_dir: str | None = None, drop_missing: bool = True,
                 # tsn
                 tsn_segments: int = 0, tsn_snippet: int = 0, jitter: bool = False,
                 d_video_in: int = 1408):
        self.df = pd.read_csv(csv_path)
        col_skel = "path_skel" if "path_skel" in self.df.columns else ("path_skeleton" if "path_skeleton" in self.df.columns else None)
        needed = {"clip", "label", "path_vjepa", col_skel}
        if col_skel is None or not needed.issubset(set(self.df.columns)):
            raise ValueError(f"CSV must have columns including 'clip','label','path_vjepa','path_skel'. Got: {self.df.columns.tolist()}" )
        # rows
        rows = []
        for _, r in self.df.iterrows():
            pv = str(r["path_vjepa"])
            ps = str(r[col_skel])
            if drop_missing and (not exists(pv) or not exists(ps) or pv == "" or ps == ""):
                continue
            rows.append({"clip": str(r["clip"]), "label": str(r["label"]), "path_vjepa": pv, "path_skel": ps})
        if len(rows) == 0:
            raise RuntimeError(f"No valid rows in {csv_path}. Check paths.")

        self.rows = rows
        self.label2id = label2id
        self.T = T
        self.tsn_segments = int(tsn_segments); self.tsn_snippet = int(tsn_snippet); self.jitter = bool(jitter)
        self._use_tsn = (self.tsn_segments > 0 and self.tsn_snippet > 0)
        self.decode_smpl = decode_smpl
        self.center_root = center_root
        self.include_trans = include_trans
        self.include_betas = include_betas
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir: self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.d_video_in = d_video_in

        # Optional: SMPL layer
        self.smpl_layer = None
        if self.decode_smpl:
            try:
                import smplx
            except ImportError as e:
                raise ImportError("smplx not installed. `pip install smplx`.") from e
            if not smpl_model_dir or not Path(smpl_model_dir).exists():
                raise FileNotFoundError("Provide --smpl_model_dir pointing to SMPL model dir (contains SMPL_NEUTRAL.pkl)." )
            self.smpl_layer = smplx.create(
                model_path=str(Path(smpl_model_dir)),
                model_type='smpl', gender='neutral', num_betas=10, use_pca=False, ext='pkl'
            ).eval()

    def __len__(self): return len(self.rows)

    def _pick_track(self, d: dict) -> dict:
        if all(k in d for k in ("pose","trans","betas")):
            return d
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
        global_orient = pose[:, :3]
        body_pose     = pose[:, 3:72]
        if betas.ndim == 1: betas = betas.unsqueeze(0).expand(T, -1)
        if trans.ndim == 1: trans = trans.unsqueeze(0).expand(T, -1)
        global_orient = global_orient.float(); body_pose = body_pose.float(); betas = betas.float(); trans = trans.float()
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

        # TSN or uniform
        if self._use_tsn:
            V = tsn_sample(V, N=self.tsn_segments, k=self.tsn_snippet, train_mode=self.jitter)
            S = tsn_sample(S, N=self.tsn_segments, k=self.tsn_snippet, train_mode=self.jitter)
        else:
            V = uniform_to_T(V, self.T); S = uniform_to_T(S, self.T)
            
        V = V.to(torch.float32)
        S = S.to(torch.float32)
        y = self.label2id[str(rec["label"])]
        return V, S, y

# ----------------------------
# Model: Fusion (cross-attn) + attention map support
# ----------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 2000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div); pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe)
    def forward(self, x):  # [B,T,D]
        T = x.size(1)
        return x + self.pe[:T, :].to(dtype=x.dtype, device=x.device).unsqueeze(0)
class FeedForward(nn.Module):
    def __init__(self, d_model: int, ff_mult: int = 4, dropout: float = 0.1):
        super().__init__()
        hidden = d_model * ff_mult
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, d_model), nn.Dropout(dropout)
        )
    def forward(self, x): return self.net(x)

class SelfAttnBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1, ff_mult: int = 4):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ln1 = nn.LayerNorm(d_model); self.ffn = FeedForward(d_model, ff_mult, dropout); self.ln2 = nn.LayerNorm(d_model)
    def forward(self, x, need_weights: bool = False):
        a,w = self.attn(x, x, x, need_weights=need_weights, average_attn_weights=False)
        x = self.ln1(x + a); f = self.ffn(x); x = self.ln2(x + f)
        return (x, w) if need_weights else x

class CrossAttnBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1, ff_mult: int = 4):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ln1 = nn.LayerNorm(d_model); self.ffn = FeedForward(d_model, ff_mult, dropout); self.ln2 = nn.LayerNorm(d_model)
    def forward(self, q, kv, need_weights: bool = False):
        a,w = self.attn(q, kv, kv, need_weights=need_weights, average_attn_weights=False)
        x = self.ln1(q + a); f = self.ffn(x); x = self.ln2(x + f)
        return (x, w) if need_weights else x

class FusionLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1, ff_mult: int = 4, enable_self_attn: bool = True):
        super().__init__()
        self.s_from_v = CrossAttnBlock(d_model, n_heads, dropout, ff_mult)
        self.v_from_s = CrossAttnBlock(d_model, n_heads, dropout, ff_mult)
        self.enable_self = enable_self_attn
        if enable_self_attn:
            self.s_self = SelfAttnBlock(d_model, n_heads, dropout, ff_mult)
            self.v_self = SelfAttnBlock(d_model, n_heads, dropout, ff_mult)
    def forward(self, s, v, need_weights: bool = False):
        if need_weights:
            s, w_sv = self.s_from_v(s, v, need_weights=True)   # (B,H,T+1,T+1)
            v, w_vs = self.v_from_s(v, s, need_weights=True)
            if self.enable_self:
                s, w_ss = self.s_self(s, need_weights=True)
                v, w_vv = self.v_self(v, need_weights=True)
            else:
                w_ss = w_vv = None
            return (s, v, w_sv, w_vs, w_ss, w_vv)
        else:
            s = self.s_from_v(s, v)
            v = self.v_from_s(v, s)
            if self.enable_self:
                s = self.s_self(s); v = self.v_self(v)
            return s, v

class FusionXAttnModel(nn.Module):
    def __init__(self, d_video_in=1408, d_skel_in=85, d_model=512, n_heads=8, depth=4, ff_mult=4, dropout=0.1, num_classes=14, T=64, enable_self_attn=True):
        super().__init__()
        self.v_proj = nn.Linear(d_video_in, d_model)
        self.s_proj = nn.Linear(d_skel_in,  d_model)
        self.cls_v = nn.Parameter(torch.zeros(1,1,d_model))
        self.cls_s = nn.Parameter(torch.zeros(1,1,d_model))
        self.pos = PositionalEncoding(d_model, max_len=T+1)
        self.layers = nn.ModuleList([FusionLayer(d_model, n_heads, dropout, ff_mult, enable_self_attn) for _ in range(depth)])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(nn.LayerNorm(2*d_model), nn.Linear(2*d_model, num_classes))
        nn.init.normal_(self.cls_v, std=0.02); nn.init.normal_(self.cls_s, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight); 
                if m.bias is not None: nn.init.zeros_(m.bias)

    def forward(self, V, S):
        v = self.v_proj(V); s = self.s_proj(S)
        B,T,D = v.shape
        v = torch.cat([self.cls_v.expand(B,1,D), v], dim=1); v = self.pos(v)
        s = torch.cat([self.cls_s.expand(B,1,D), s], dim=1); s = self.pos(s)
        for layer in self.layers:
            s, v = layer(s, v)
        cls_v = self.norm(v[:,0]); cls_s = self.norm(s[:,0])
        fused = torch.cat([cls_v, cls_s], dim=-1)
        return self.head(fused)

    @torch.inference_mode()
    def forward_with_attn(self, V, S):
        """Return logits plus attention maps (per layer, per direction)."""
        v = self.v_proj(V); s = self.s_proj(S)
        B,T,D = v.shape
        v = torch.cat([self.cls_v.expand(B,1,D), v], dim=1); v = self.pos(v)
        s = torch.cat([self.cls_s.expand(B,1,D), s], dim=1); s = self.pos(s)
        attn_sv, attn_vs = [], []
        for layer in self.layers:
            s, v, w_sv, w_vs, _, _ = layer(s, v, need_weights=True)
            attn_sv.append(w_sv)  # (B,H,Ts+1,Tv+1)
            attn_vs.append(w_vs)  # (B,H,Tv+1,Ts+1)
        cls_v = self.norm(v[:,0]); cls_s = self.norm(s[:,0])
        fused = torch.cat([cls_v, cls_s], dim=-1)
        logits = self.head(fused)
        return logits, attn_sv, attn_vs

# ----------------------------
# Eval helpers
# ----------------------------
@torch.inference_mode()
def evaluate(model, loader, device, num_classes, amp=False, label2id=None, bg_name='[OP000] No action'):
    model.eval()
    all_logits, all_targets = [], []
    ctx = torch.amp.autocast if torch.cuda.is_available() else torch.cpu.amp.autocast
    with ctx(device_type="cuda", enabled=amp and torch.cuda.is_available()):
        for V, S, y in loader:
            V = V.to(device, non_blocking=True); S = S.to(device, non_blocking=True); y = y.to(device, non_blocking=True)
            logits = model(V, S)
            all_logits.append(logits.detach().cpu()); all_targets.append(y.detach().cpu())
    logits = torch.cat(all_logits, dim=0); probs = torch.softmax(logits, dim=-1).numpy()
    targets = torch.cat(all_targets, dim=0).numpy().astype(int); preds = probs.argmax(axis=1)
    top1 = float((preds == targets).mean())
    k_eff = min(5, probs.shape[1]); top5 = float(top_k_accuracy_score(targets, probs, k=k_eff, labels=list(range(probs.shape[1]))))
    y_true = np.zeros((targets.shape[0], probs.shape[1]), dtype=np.int32); y_true[np.arange(targets.shape[0]), targets] = 1
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
# Attention map exporting
# ----------------------------
@torch.inference_mode()
def export_attention_maps(model, dataset, device, out_dir: Path, num_samples: int = 4, layer_ids: str = ""):
    out_dir = Path(out_dir) / "attn_maps"; out_dir.mkdir(parents=True, exist_ok=True)
    layers = None
    if layer_ids:
        layers = [int(x) for x in layer_ids.split(",") if x.strip().isdigit()]
    print(f"[ATTN] Exporting attention for {num_samples} samples. Layers: {layers if layers is not None else 'all'}")
    model_dtype = next(model.parameters()).dtype
    for i in range(min(num_samples, len(dataset))):
        V, S, y = dataset[i]
        V = V.to(model_dtype).unsqueeze(0).to(device)
        S = S.to(model_dtype).unsqueeze(0).to(device)
        logits, attn_sv, attn_vs = model.forward_with_attn(V, S)
        pred = int(logits.argmax(dim=-1).item())
        # Choose layers
        L = len(attn_sv); chosen = (layers if layers is not None else list(range(L)))
        for l in chosen:
            if l < 0 or l >= L: continue
            w_sv = attn_sv[l].mean(dim=1).squeeze(0).cpu().numpy()  # (Ts+1, Tv+1)
            w_vs = attn_vs[l].mean(dim=1).squeeze(0).cpu().numpy()  # (Tv+1, Ts+1)
            # Save .npy
            np.save(out_dir / f"sample{i:02d}_layer{l:02d}_SattV.npy", w_sv)
            np.save(out_dir / f"sample{i:02d}_layer{l:02d}_VattS.npy", w_vs)
            # Save PNGs (temporal attention heatmaps)
            if plt is not None:
                for tag, mat in [("SattV", w_sv), ("VattS", w_vs)]:
                    fig = plt.figure(figsize=(6,4), dpi=150); ax = plt.gca()
                    im = ax.imshow(mat, aspect='auto', interpolation='nearest', cmap='magma')
                    plt.colorbar(im, fraction=0.046, pad=0.04)
                    ax.set_xlabel("Keys (attended)"); ax.set_ylabel("Queries")
                    ax.set_title(f"sample {i} | layer {l} | {tag} | y={y} pred={pred}")
                    plt.tight_layout()
                    plt.savefig(out_dir / f"sample{i:02d}_layer{l:02d}_{tag}.png"); plt.close(fig)
        print(f"[ATTN] Saved sample {i} attention maps -> {out_dir}")

# ----------------------------
# Training
# ----------------------------
def train(args):
    for p, flag in [(args.train_csv, "--train_csv"), (args.val_csv, "--val_csv")]:
        if not exists(p): raise FileNotFoundError(f"{flag} not found: {p}")
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Speed knobs: TF32 & Flash/Mem-efficient SDPA
    # put near start of train(args), right after device = "cuda" ...
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
    log_path, LOG_FH, ts = start_text_logging(txt_dir, prefix="fusion", ts=run_id)

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

    # TSN normalization
    if getattr(args, "tsn_segments", 0) and args.tsn_segments > 0:
        if not getattr(args, "tsn_snippet", 0) or args.tsn_snippet <= 0:
            args.tsn_snippet = max(1, args.T // args.tsn_segments)
        T_eff = args.tsn_segments * args.tsn_snippet
        if T_eff != args.T:
            print(f"[TSN] Adjusting T from {args.T} -> {T_eff} to match N*k."); args.T = T_eff
        print(f"[TSN] Using N={args.tsn_segments}, k={args.tsn_snippet}, T={args.T} (train=jitter, val=center)")

    # Datasets / loaders
    base_ds_kwargs = dict(
        T=args.T, decode_smpl=args.decode_smpl, smpl_model_dir=args.smpl_model_dir,
        center_root=not args.no_center_root, include_trans=args.include_trans, include_betas=args.include_betas,
        cache_dir=args.cache_dir, drop_missing=True,
        tsn_segments=getattr(args, "tsn_segments", 0), tsn_snippet=getattr(args, "tsn_snippet", 0), d_video_in=args.d_video_in
    )
    train_ds = FusionIndexDataset(args.train_csv, label2id, jitter=True, **base_ds_kwargs)
    val_ds   = FusionIndexDataset(args.val_csv,   label2id, jitter=False, **base_ds_kwargs)

    # Infer d_skel_in
    Vi, Si, _ = train_ds[0]; d_skel_in = int(Si.shape[1]); print(f"Inferred d_skel_in={d_skel_in} | d_video_in={args.d_video_in}")

    # Loaders
    num_workers = max(0, int(args.num_workers)); prefetch_factor = int(getattr(args, "prefetch_factor", 2))
    loader_kwargs = dict(num_workers=num_workers, pin_memory=True)
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True; loader_kwargs["prefetch_factor"] = prefetch_factor
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True, **loader_kwargs)
    val_loader   = DataLoader(val_ds,   batch_size=max(1, args.batch_size*2), shuffle=False, **loader_kwargs)

    # Model
    model = FusionXAttnModel(d_video_in=args.d_video_in, d_skel_in=d_skel_in, d_model=args.d_model, n_heads=args.n_heads,
                             depth=args.depth, ff_mult=args.ff_mult, dropout=args.dropout, num_classes=num_classes, T=args.T,
                             enable_self_attn=not args.disable_self_attn).to(device)
    params_m = count_trainable_params(model) / 1e6; print(f"Trainable parameters: {params_m:.3f}M")

    # Optim & schedule
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, fused=True)
    total_steps = args.epochs * max(1, math.ceil(len(train_loader)))
    warmup_steps = int(0.05 * total_steps)
    def lr_schedule(step):
        if step < warmup_steps: return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scaler = torch.amp.GradScaler('cuda', enabled=args.amp and torch.cuda.is_available())
    criterion = nn.CrossEntropyLoss()

    out_dir = Path(args.run_dir); out_dir.mkdir(parents=True, exist_ok=True)
    best_map = -1.0; global_step = 0

    # Train loop
    epoch_rows = []
    for epoch in range(1, args.epochs+1):
        model.train(); epoch_loss = 0.0
        pbar = tqdm(
            train_loader,
            desc=f"Epoch {epoch}/{args.epochs}",
            ncols=100,
            dynamic_ncols=True,
            leave=True,          # keep one summary line per epoch
            file=sys.stdout,     # <— ensure it goes through Tee
            mininterval=0.1,
        )
        for V, S, y in pbar:
            V = V.to(device, non_blocking=True); S = S.to(device, non_blocking=True); y = y.to(device, non_blocking=True)
            # LR schedule
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

        # Eval
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
            "epoch": epoch,
            "train_loss": avg_loss,
            "val_top1": metrics["top1"],
            "val_top5": metrics.get("top5", float("nan")),
            "val_mAP_all": metrics["mAP_all"],
            "val_mAP_valid": metrics["mAP_valid"],
            "val_macroP": metrics["macroP"],
            "val_macroR": metrics["macroR"],
            "val_macroF1": metrics["macroF1"],
            "val_mAP_noBG": metrics.get("mAP_noBG", None),
        }
        epoch_rows.append(row)
        pd.DataFrame(epoch_rows).to_csv(Path(args.run_dir) / "epoch_metrics.csv", index=False)

        # Save best by mAP_valid
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
            torch.save(ckpt, out_dir / "best_fusion.ckpt")
            print(f"Saved best checkpoint -> {out_dir/'best_fusion.ckpt'} (mAP_valid {best_map:.4f})")

    # Final evaluation
    best_path = out_dir / "best_fusion.ckpt"
    if best_path.exists():
        print(f"\n[FINAL] Loading best checkpoint: {best_path}")
        ckpt = torch.load(best_path, map_location="cpu")
        model.load_state_dict(ckpt["model"]); model.to(device)

        final_metrics = final_evaluation_and_report(
            model, val_loader, device, num_classes, label2id, out_dir, amp=args.amp
        )
        fps = measure_inference_fps(model, val_loader, device, amp=args.amp)
        print(f"[FINAL] Inference throughput: {fps:.2f} FPS")

        # ---- Save final eval to TXT + JSON ----
        run_dir = Path(args.run_dir)
        final_txt  = run_dir / "final_eval.txt"
        final_json = run_dir / "final_eval.json"

        headline = (
            f"Fusion (X-Attn) | Top-1 {final_metrics['top1']*100:5.2f}% | "
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
            json.dump(
                {
                    **final_metrics,
                    "fps": float(fps),
                    "params_m": float(params_m),
                    "run_dir": str(run_dir),
                    "best_ckpt": str(best_path),
                },
                jf,
                indent=2,
            )

        # still print headline to console
        print("\n=== Headline Row (copy into paper table) ===")
        print(headline)

        # Optional: export attention maps
        if args.dump_attention > 0:
            export_attention_maps(model, val_ds, device, out_dir, num_samples=args.dump_attention, layer_ids=args.attn_layers)
    else:
        print("[FINAL] Best checkpoint not found; skipping final evaluation.")


# ----------------------------
# Main / CLI
# ----------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    # training CSVs become optional (we'll check conditionally)
    ap.add_argument("--train_csv", type=str, default=None)
    ap.add_argument("--val_csv",   type=str, default=None)

    # logging / runs
    ap.add_argument("--log_dir",     type=str, default="/workspace/har-fusion/runs/fusion")
    ap.add_argument("--txt_log_dir", type=str, default="/workspace/trainheads/runs/fusion")
    ap.add_argument("--run_name",    type=str, default=None)
    ap.add_argument("--labelmap",    type=str, default="/workspace/har-fusion/runs/fusion/label2id.json")

    # data/model
    ap.add_argument("--T",           type=int,   default=64)
    ap.add_argument("--d_video_in",  type=int,   default=1408)
    ap.add_argument("--d_model",     type=int,   default=512)
    ap.add_argument("--n_heads",     type=int,   default=8)
    ap.add_argument("--depth",       type=int,   default=4)
    ap.add_argument("--ff_mult",     type=int,   default=4)
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

    # attention export
    ap.add_argument("--dump_attention", type=int, default=0,
                    help="Export attention maps for N validation samples (0=off).")
    ap.add_argument("--attn_layers",   type=str, default="",
                    help="Comma-separated layer ids to export (default: all).")
    ap.add_argument("--disable_self_attn", action="store_true")

    # -------- Occlusion study mode --------
    ap.add_argument("--occ_eval", action="store_true",
                    help="Run occlusion ablation (no training).")
    ap.add_argument("--occ_low_csv",  type=str, default=None,
                    help="CSV for low-occlusion split.")
    ap.add_argument("--occ_high_csv", type=str, default=None,
                    help="CSV for high-occlusion split.")
    ap.add_argument("--ckpt",         type=str, default=None,
                    help="Path to trained Fusion checkpoint (.ckpt).")

    args = ap.parse_args()

    if args.occ_eval:
        # minimal sanity checks
        missing = [k for k in ("occ_low_csv", "occ_high_csv", "ckpt")
                   if not getattr(args, k, None)]
        if missing:
            raise SystemExit(f"--occ_eval requires: {', '.join('--'+m for m in missing)}")
        run_occlusion_study(args)
    else:
        # training path still requires these
        if not args.train_csv or not args.val_csv:
            raise SystemExit("Missing --train_csv and/or --val_csv (or pass --occ_eval to run the ablation).")
        train(args)

