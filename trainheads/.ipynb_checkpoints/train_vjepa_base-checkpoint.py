import os, json, math, argparse, random
from pathlib import Path
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import sys, io
from datetime import datetime
import time
import numpy as np
import warnings
from sklearn.metrics import (
    average_precision_score,
    precision_recall_fscore_support,
    confusion_matrix,
    top_k_accuracy_score,
)

# optional heatmap
try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None

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

def start_text_logging(out_dir: str, prefix: str = "vjepa_only", ts: str | None = None):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    ts = ts or datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = Path(out_dir) / f"{prefix}_{ts}.txt"
    fh = open(log_path, "a", buffering=1, encoding="utf-8")  # line-buffered
    sys.stdout = Tee(sys.stdout, fh)
    sys.stderr = Tee(sys.stderr, fh)
    print("="*80)
    print(f"[RUN START] {ts}  |  log={log_path}")
    return str(log_path), fh, ts



# ----------------------------
# Utilities
# ----------------------------

def uniform_to_T(x: torch.Tensor, T: int) -> torch.Tensor:
    """Uniformly sample or pad to exactly T frames. x: [Tin, D]."""
    Tin = x.shape[0]
    if Tin == T:
        return x
    if Tin > T:
        idx = torch.linspace(0, Tin - 1, T).round().long()
        return x.index_select(0, idx)
    # pad
    pad = torch.zeros(T - Tin, x.shape[1], dtype=x.dtype)
    return torch.cat([x, pad], dim=0)

def tsn_snippet_indices(T_in: int, N: int, k: int, jitter: bool) -> torch.Tensor:
    """
    Split [0, T_in) into N segments; pick a k-length contiguous window per segment.
    Concatenate to length N*k. If a segment is shorter than k, repeat last index.
    """
    if T_in <= 0:
        return torch.zeros(0, dtype=torch.long)

    bounds = torch.linspace(0, T_in, steps=N + 1)
    idxs = []
    for i in range(N):
        s = int(math.floor(bounds[i].item()))
        e = int(math.ceil(bounds[i + 1].item()))  # exclusive
        seg_len = max(1, e - s)
        win = min(k, seg_len)

        start_lo = s
        start_hi = max(s, e - win)
        start = (np.random.randint(start_lo, start_hi + 1) if jitter
                 else (start_lo + start_hi) // 2)

        cont = torch.arange(start, start + win)
        cont = torch.clamp(cont, max=e - 1)

        if win < k:
            pad = cont[-1].repeat(k - win)
            cont = torch.cat([cont, pad], dim=0)

        idxs.append(cont)
    return torch.cat(idxs, dim=0)  # [N*k]

def tsn_sample(x: torch.Tensor, N: int, k: int, train_mode: bool) -> torch.Tensor:
    """x: [Tin, D] -> [N*k, D] via TSN sampling."""
    Tin = x.shape[0]
    if Tin == 0:
        return torch.zeros(N * k, x.shape[1], dtype=x.dtype)
    idx = tsn_snippet_indices(Tin, N, k, jitter=train_mode)
    return x.index_select(0, idx)

def count_trainable_params(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

@torch.no_grad()
def measure_inference_fps(model, loader, device, amp: bool = True):
    """Rough throughput (frames/sec) over the val loader."""
    model.eval()
    total_frames = 0
    start = None
    ctx = (torch.amp.autocast if torch.cuda.is_available() else torch.cpu.amp.autocast)
    warmup = 5
    for i, (feats, _) in enumerate(loader):
        feats = feats.to(device, non_blocking=True)
        if i < warmup:
            with ctx(device_type="cuda", enabled=amp and torch.cuda.is_available()):
                _ = model(feats)
            continue
        if start is None:
            if torch.cuda.is_available(): torch.cuda.synchronize()
            start = time.time()
        with ctx(device_type="cuda", enabled=amp and torch.cuda.is_available()):
            _ = model(feats)
        B, T = feats.shape[0], feats.shape[1]
        total_frames += B * T
    if start is None:
        return 0.0
    if torch.cuda.is_available(): torch.cuda.synchronize()
    return float(total_frames / max(time.time() - start, 1e-9))

def _save_confusion_csv_png(y_true, y_pred, class_names, out_dir: Path):
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(len(class_names)), normalize='true')
    # CSV
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "confusion_matrix_normalized.csv"
    with open(csv_path, "w") as f:
        f.write(",".join(["class"] + class_names) + "\n")
        for i, row in enumerate(cm):
            f.write(",".join([class_names[i]] + [f"{x:.6f}" for x in row]) + "\n")
    print(f"[FINAL] Saved confusion matrix CSV -> {csv_path}")
    # PNG
    if plt is not None and len(class_names) <= 100:
        h = max(6, min(18, int(len(class_names) * 0.20)))
        fig = plt.figure(figsize=(h, h), dpi=150)
        ax = plt.gca()
        im = ax.imshow(cm, interpolation='nearest', cmap='viridis')
        plt.colorbar(im, fraction=0.046, pad=0.04)
        ax.set_xticks(np.arange(len(class_names)))
        ax.set_yticks(np.arange(len(class_names)))
        ax.set_xticklabels(class_names, rotation=90, fontsize=6)
        ax.set_yticklabels(class_names, fontsize=6)
        ax.set_xlabel("Predicted"); ax.set_ylabel("True")
        ax.set_title("Normalized Confusion Matrix")
        plt.tight_layout()
        png_path = out_dir / "confusion_matrix_normalized.png"
        plt.savefig(png_path); plt.close(fig)
        print(f"[FINAL] Saved confusion matrix PNG -> {png_path}")

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

# ----------------------------
# Dataset
# ----------------------------
class VJEPAIndexDataset(Dataset):
    def __init__(self, csv_path, label2id, T=64, d_in=1408, drop_missing=True, tsn_segments: int = 0, tsn_snippet: int = 0, jitter: bool = False):
        self.df = pd.read_csv(csv_path)
        needed = {"clip", "label", "path_vjepa"}
        if not needed.issubset(set(self.df.columns)):
            raise ValueError(f"CSV must have columns: {needed}. Got: {self.df.columns.tolist()}")

        rows = []
        for _, r in self.df.iterrows():
            p = str(r["path_vjepa"])
            if drop_missing and not exists(p):
                continue
            rows.append({"clip": str(r["clip"]), "label": str(r["label"]), "path_vjepa": p})
        if len(rows) == 0:
            raise RuntimeError(f"No valid rows in {csv_path}. Check paths.")

        self.rows = rows
        self.label2id = label2id
        self.T = T
        self.d_in = d_in

        # TSN controls
        self.tsn_segments = int(tsn_segments)
        self.tsn_snippet  = int(tsn_snippet)
        self.jitter       = bool(jitter)
        self._use_tsn     = (self.tsn_segments > 0 and self.tsn_snippet > 0)

    def __len__(self): return len(self.rows)

    def __getitem__(self, idx):
        rec = self.rows[idx]
        feat = torch.load(rec["path_vjepa"], map_location="cpu")  # [Tin, D]
        if feat.ndim != 2:
            raise RuntimeError(f"Expected 2D tensor [T, D], got {feat.shape} in {rec['path_vjepa']}")

        Tin, Din = feat.shape
        feat = feat.to(torch.float32)
        if Din != self.d_in:
            raise RuntimeError(f"D mismatch: expected {self.d_in}, got {Din} for {rec['path_vjepa']}")

        if self._use_tsn:
            feat = tsn_sample(feat, N=self.tsn_segments, k=self.tsn_snippet, train_mode=self.jitter)  # -> [N*k, D]
        else:
            feat = uniform_to_T(feat, self.T)  # consistent uniform sampling fallback

        y = self.label2id[str(rec["label"])]
        return feat, y

# ----------------------------
# Model: Attentive probe (Transformer with CLS)
# ----------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 2000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe)  # [max_len, d_model]

    def forward(self, x):  # x: [B, T, D]
        T = x.size(1)
        return x + self.pe[:T, :].unsqueeze(0)

class VideoOnlyAttnProbe(nn.Module):
    def __init__(self, d_in=1408, d_model=1408, n_heads=8, depth=2, ff_mult=4, num_classes=13, dropout=0.1, T=64):
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

        nn.init.trunc_normal_(self.head.weight, std=0.02)
        nn.init.zeros_(self.head.bias)

    def forward(self, x):  # x: [B, T, d_in]
        x = self.proj(x)  # [B, T, d_model]
        B, T, D = x.shape
        cls = self.cls_token.expand(B, 1, D)
        x = torch.cat([cls, x], dim=1)  # [B, T+1, D]
        x = self.pos_enc(x)
        x = self.encoder(x)
        cls_out = self.norm(x[:, 0])
        logits = self.head(cls_out)
        return logits

# ----------------------------
# Training / Evaluation
# ----------------------------
from sklearn.metrics import average_precision_score
import numpy as np

@torch.no_grad()
def evaluate(model, loader, device, num_classes, amp=False, label2id=None, bg_name='[OP000] No action'):
    model.eval()
    all_probs, all_targets = [], []
    correct = total = 0
    ctx = (torch.amp.autocast if torch.cuda.is_available() else torch.cpu.amp.autocast)  # PyTorch 2.x
    with ctx(device_type="cuda", enabled=amp and torch.cuda.is_available()):
        for feats, y in loader:
            feats = feats.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            probs = torch.softmax(model(feats), dim=-1)
            pred = probs.argmax(-1)
            correct += (pred == y).sum().item(); total += y.numel()
            all_probs.append(probs.cpu()); all_targets.append(y.cpu())

    acc = correct / max(total, 1)
    probs = torch.cat(all_probs).numpy()                 # [N, C]
    t = torch.cat(all_targets).numpy().astype(int)       # [N]

    # One-vs-rest targets
    y_true = np.zeros((t.shape[0], num_classes), dtype=np.int32)
    y_true[np.arange(t.shape[0]), t] = 1

    # Compute support mask once
    support = y_true.sum(axis=0)             # [C]
    valid_mask = support > 0                 # classes present in this split

    # mAP over only classes that appear in val (avoids sklearn warnings)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        mAP_all   = average_precision_score(y_true[:, valid_mask], probs[:, valid_mask], average="macro")
        mAP_valid = mAP_all  # same quantity; keep both names for logs

    # Optional: mAP excluding a background class if you have one
    mAP_no_bg = None
    if label2id and ('[OP000] No action' in label2id):
        bg_id = label2id['[OP000] No action']
        mask_no_bg = valid_mask.copy()
        if 0 <= bg_id < num_classes:
            mask_no_bg[bg_id] = False
        if mask_no_bg.any():
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=UserWarning)
                mAP_no_bg = average_precision_score(y_true[:, mask_no_bg], probs[:, mask_no_bg], average="macro")

    return acc, mAP_all, mAP_valid, mAP_no_bg


@torch.inference_mode()
def final_evaluation_and_report(model, loader, device, num_classes, label2id, out_dir, amp=False):
    model.eval()
    all_logits, all_targets = [], []
    ctx = (torch.amp.autocast if torch.cuda.is_available() else torch.cpu.amp.autocast)
    with ctx(device_type="cuda", enabled=amp and torch.cuda.is_available()):
        for feats, y in loader:
            feats = feats.to(device, non_blocking=True)
            y     = y.to(device, non_blocking=True)
            all_logits.append(model(feats).detach().cpu())
            all_targets.append(y.detach().cpu())

    logits  = torch.cat(all_logits, dim=0)
    probs   = torch.softmax(logits, dim=-1).numpy()
    targets = torch.cat(all_targets, dim=0).numpy().astype(int)
    preds   = probs.argmax(axis=1)

    # Headline metrics
    top1 = float((preds == targets).mean())
    k_eff = min(5, probs.shape[1])
    top5 = float(top_k_accuracy_score(targets, probs, k=k_eff, labels=list(range(probs.shape[1]))))

    y_true = np.zeros_like(probs, dtype=np.int32)
    y_true[np.arange(targets.shape[0]), targets] = 1
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        mAP_all = float(average_precision_score(y_true, probs, average="macro"))
    support_mask = (y_true.sum(axis=0) > 0)
    mAP_valid = float(average_precision_score(y_true[:, support_mask], probs[:, support_mask], average="macro"))

    macroP, macroR, macroF1, _ = precision_recall_fscore_support(
        targets, preds, average="macro", zero_division=0
    )

    # Per-class PRF + AP
    P_c, R_c, F1_c, Supp_c = precision_recall_fscore_support(
        targets, preds, average=None, labels=np.arange(probs.shape[1]), zero_division=0
    )
    AP_c = []
    for c in range(probs.shape[1]):
        if support_mask[c]:
            AP_c.append(float(average_precision_score(y_true[:, c], probs[:, c])))
        else:
            AP_c.append(float("nan"))

    id2label = {v: k for k, v in label2id.items()}
    class_names = [id2label.get(i, str(i)) for i in range(probs.shape[1])]

    # Save per-class CSV
    per_class_df = pd.DataFrame({
        "class_id": np.arange(probs.shape[1]),
        "class_name": class_names,
        "precision": P_c, "recall": R_c, "f1": F1_c, "AP": AP_c, "support": Supp_c
    })
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    per_class_csv = out_dir / "per_class_metrics.csv"
    per_class_df.to_csv(per_class_csv, index=False)
    print(f"[FINAL] Saved per-class metrics -> {per_class_csv}")

    # Confusion artifacts
    _save_confusion_csv_png(targets, preds, class_names, out_dir)

    # Pretty print
    print("\n================ FINAL EVALUATION (best checkpoint) ================")
    print(f"Top-1 Acc: {top1:.4f}  |  Top-5 Acc: {top5:.4f}")
    print(f"Macro mAP (all):   {mAP_all:.4f}")
    print(f"Macro mAP (valid): {mAP_valid:.4f}")
    print(f"Macro Precision:   {macroP:.4f}")
    print(f"Macro Recall:      {macroR:.4f}")
    print(f"Macro F1-Score:    {macroF1:.4f}")

    return {
        "top1": top1, "top5": top5,
        "mAP_all": mAP_all, "mAP_valid": mAP_valid,
        "macroP": float(macroP), "macroR": float(macroR), "macroF1": float(macroF1),
    }


# ===== Occlusion Study (Video-only: V-JEPA) =====

@torch.inference_mode()
def _occ_eval_on_csv_vjepa(model, csv_path, label2id, args):
    """Build a val loader from a CSV and return {'top1': float}."""
    ds = VJEPAIndexDataset(
        csv_path, label2id,
        T=args.T,
        d_in=getattr(args, "d_in", 1408),
        drop_missing=True,
        tsn_segments=getattr(args, "tsn_segments", 0),
        tsn_snippet=getattr(args, "tsn_snippet", 0),
        jitter=False,
    )
    loader = DataLoader(
        ds, batch_size=max(1, getattr(args, "batch_size", 128) * 2),
        shuffle=False,
        num_workers=max(0, int(getattr(args, "num_workers", 12) // 2)),
        pin_memory=True
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.eval()
    all_logits, all_targets = [], []
    ctx = torch.amp.autocast if torch.cuda.is_available() else torch.cpu.amp.autocast
    with ctx(device_type="cuda", enabled=getattr(args, "amp", False) and torch.cuda.is_available()):
        for X, y in loader:
            X = X.to(device, non_blocking=True); y = y.to(device, non_blocking=True)
            logits = model(X)
            all_logits.append(logits.detach().cpu()); all_targets.append(y.detach().cpu())
    logits = torch.cat(all_logits, dim=0).numpy()
    targets = torch.cat(all_targets, dim=0).numpy().astype(int)
    preds = logits.argmax(axis=1)
    top1 = float((preds == targets).mean())
    return {"top1": top1}

def run_occlusion_study(args):
    """
    Evaluate a trained Video-only checkpoint on low- and high-occlusion CSVs,
    print a small table, and write JSON/MD artifacts.
    """
    ckpt_path = Path(args.ckpt)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location="cpu")
    label2id = ckpt.get("label2id", None)
    if label2id is None:
        if not (args.labelmap and Path(args.labelmap).exists()):
            raise RuntimeError("label2id missing in ckpt; please pass --labelmap pointing to label2id.json")
        label2id = load_label_map(args.labelmap)

    # Rebuild model from checkpoint args (fall back to CLI if missing)
    margs = ckpt.get("args", {})
    args.d_model = margs.get("d_model", getattr(args, "d_model", 1408))
    args.n_heads = margs.get("n_heads", getattr(args, "n_heads", 8))
    args.depth   = margs.get("depth",   getattr(args, "depth", 2))
    args.ff_mult = margs.get("ff_mult", getattr(args, "ff_mult", 4))
    args.dropout = margs.get("dropout", getattr(args, "dropout", 0.1))
    args.T       = margs.get("T",       getattr(args, "T", 64))
    args.d_in    = margs.get("d_in",    getattr(args, "d_in", 1408))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = VideoOnlyAttnProbe(
        d_in=args.d_in, d_model=args.d_model, n_heads=args.n_heads,
        depth=args.depth, ff_mult=args.ff_mult, num_classes=len(label2id),
        dropout=args.dropout, T=args.T
    ).to(device)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()

    # Evaluate both splits (CSV must have clip,label,path_vjepa)
    low_metrics  = _occ_eval_on_csv_vjepa(model, args.occ_low_csv,  label2id, args)
    high_metrics = _occ_eval_on_csv_vjepa(model, args.occ_high_csv, label2id, args)

    low_top1  = 100.0 * float(low_metrics["top1"])
    high_top1 = 100.0 * float(high_metrics["top1"])
    delta_pp  = high_top1 - low_top1  # High - Low

    print("\n=== Occlusion Study (V-JEPA video-only) ===")
    print(f"Low  ({Path(args.occ_low_csv).name}):  Top-1 = {low_top1:.1f}%")
    print(f"High ({Path(args.occ_high_csv).name}): Top-1 = {high_top1:.1f}%")
    print(f"Δ (High - Low): {delta_pp:+.1f} pp\n")

    # Save artifacts next to the checkpoint directory
    out_dir = ckpt_path.parent
    payload = {
        "model": "V-JEPA (video-only)",
        "low_csv":  str(args.occ_low_csv),
        "high_csv": str(args.occ_high_csv),
        "low_top1":  low_top1,
        "high_top1": high_top1,
        "delta_pp":  delta_pp,
        "ckpt": str(ckpt_path)
    }
    with open(out_dir / "occ_eval_vjepa.json", "w") as f:
        json.dump(payload, f, indent=2)

    with open(out_dir / "occ_eval_vjepa.md", "w") as f:
        f.write("| Model | Top-1 (Low) | Top-1 (High) | Δ (pp) |\n")
        f.write("|---|---:|---:|---:|\n")
        f.write(f"| V-JEPA (video-only) | {low_top1:.1f}% | {high_top1:.1f}% | {delta_pp:+.1f} |\n")

    return payload






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
    # ---- TSN normalization: enforce T = N * k if TSN is enabled ----
    if getattr(args, "tsn_segments", 0) and args.tsn_segments > 0:
        if not getattr(args, "tsn_snippet", 0) or args.tsn_snippet <= 0:
            # derive k from desired T
            args.tsn_snippet = max(1, args.T // args.tsn_segments)
        T_eff = args.tsn_segments * args.tsn_snippet
        if T_eff != args.T:
            print(f"[TSN] Adjusting T from {args.T} -> {T_eff} to match N*k.")
            args.T = T_eff
        print(f"[TSN] Using N={args.tsn_segments}, k={args.tsn_snippet}, T={args.T} "
              f"(train=jitter, val=center)")

    # Datasets / loaders
    train_ds = VJEPAIndexDataset(
        args.train_csv, label2id, T=args.T, d_in=args.d_in, drop_missing=True,
        tsn_segments=getattr(args, "tsn_segments", 0),
        tsn_snippet=getattr(args, "tsn_snippet", 0),
        jitter=True,   # random snippet per segment for training
    )
    val_ds = VJEPAIndexDataset(
        args.val_csv, label2id, T=args.T, d_in=args.d_in, drop_missing=True,
        tsn_segments=getattr(args, "tsn_segments", 0),
        tsn_snippet=getattr(args, "tsn_snippet", 0),
        jitter=False,  # center (deterministic) snippet for eval
    )


    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
        persistent_workers=True if args.num_workers > 0 else False,
        prefetch_factor=4 if args.num_workers > 0 else None,
        pin_memory_device="cuda" if torch.cuda.is_available() else ""
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size*2, shuffle=False,
        num_workers=max(1, args.num_workers//2), pin_memory=True,
        persistent_workers=True if args.num_workers > 0 else False,
        prefetch_factor=4 if args.num_workers > 0 else None,
        pin_memory_device="cuda" if torch.cuda.is_available() else ""
    )

    # Model / Optim
    model = VideoOnlyAttnProbe(
        d_in=args.d_in, d_model=args.d_model, n_heads=args.n_heads,
        depth=args.depth, ff_mult=args.ff_mult, num_classes=num_classes,
        dropout=args.dropout, T=args.T
    ).to(device)

    params_m = count_trainable_params(model) / 1e6
    print(f"Trainable parameters: {params_m:.3f}M")

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = args.epochs * max(1, math.ceil(len(train_loader)))
    warmup_steps = int(0.05 * total_steps)
    def lr_schedule(step):
        if step < warmup_steps: return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scaler = torch.amp.GradScaler('cuda', enabled=args.amp and torch.cuda.is_available())
    criterion = nn.CrossEntropyLoss()

    out_dir = Path(args.run_dir); out_dir.mkdir(parents=True, exist_ok=True)
    best_map = -1.0
    global_step = 0

    # ===== Train loop =====
    for epoch in range(1, args.epochs+1):
        model.train()
        epoch_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", ncols=100)
        for feats, y in pbar:
            feats = feats.to(device, non_blocking=True)
            y     = y.to(device, non_blocking=True)

            # LR schedule
            for g in optim.param_groups:
                g["lr"] = args.lr * lr_schedule(global_step)

            optim.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=args.amp and torch.cuda.is_available()):
                logits = model(feats)
                loss   = criterion(logits, y)

            scaler.scale(loss).backward()
            if args.grad_clip is not None:
                scaler.unscale_(optim)
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optim); scaler.update()

            global_step += 1
            epoch_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}", "step": global_step})

        # ===== Eval on val =====
        # ===== Eval on val =====
        val_acc, val_map_all, val_map_valid, val_map_nobg = evaluate(model, val_loader, device, num_classes, amp=args.amp, label2id=label2id)
        avg_loss = epoch_loss / max(len(train_loader), 1)

        # pretty strings
        no_bg_str = (
            f"{val_map_nobg:.4f}" if (val_map_nobg is not None and np.isfinite(val_map_nobg))
            else "n/a"
        )

        print(
            f"[Epoch {epoch}] "
            f"train_loss={avg_loss:.4f}  "
            f"val_acc={val_acc:.4f}  "
            f"val_mAP_all={val_map_all:.4f}  "
            f"val_mAP_valid={val_map_valid:.4f}  "
            f"val_mAP_noBG={no_bg_str}"
        )

        # Save best by mAP_valid
        if val_map_valid > best_map:
            best_map = val_map_valid
            ckpt = {
                "model": model.state_dict(),
                "args": vars(args),
                "label2id": label2id,
                "val_acc": val_acc,
                "val_mAP_all": val_map_all,
                "val_mAP_valid": val_map_valid,
                "val_mAP_noBG": val_map_nobg,
                "params_m": params_m,
            }
            torch.save(ckpt, out_dir / "best_video_only.ckpt")
            print(f"Saved best checkpoint to {out_dir/'best_video_only.ckpt'} (mAP_valid {best_map:.4f})")


    # ===== Final single-pass evaluation on best checkpoint =====
    best_path = out_dir / "best_video_only.ckpt"
    if best_path.exists():
        print(f"\n[FINAL] Loading best checkpoint: {best_path}")
        try:
            ckpt = torch.load(best_path, map_location="cpu")  # tries weights_only=True 
        except Exception:
            # Trusted file (we just saved it), so fall back to full pickle
            ckpt = torch.load(best_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model"]); model.to(device)

        final_metrics = final_evaluation_and_report(
            model=model,
            loader=val_loader,
            device=device,
            num_classes=num_classes,
            label2id=label2id,
            out_dir=out_dir,
            amp=args.amp
        )

        fps = measure_inference_fps(model, val_loader, device, amp=args.amp)
        print(f"[FINAL] Inference throughput: {fps:.2f} FPS (frames/sec)")

        # Print a headline row like your table
        print("\n=== Headline Row (copy into paper table) ===")
        print(
            f"Video Only | "
            f"Top-1 {final_metrics['top1']*100:5.2f}% | "
            f"Top-5 {final_metrics['top5']*100:5.2f}% | "
            f"Macro mAP {final_metrics['mAP_valid']*100:5.2f}% | "
            f"Macro F1 {final_metrics['macroF1']*100:5.2f}% | "
            f"Params {params_m:.3f}M | "
            f"FPS {fps:.2f}"
        )
    else:
        print("[FINAL] Best checkpoint not found; skipping final evaluation.")



if __name__ == "__main__":
    ap = argparse.ArgumentParser()

    # training CSVs
    ap.add_argument("--train_csv", type=str, default=None)
    ap.add_argument("--val_csv",   type=str, default=None)

    # logs / run config
    ap.add_argument("--log_dir",     type=str, default="/workspace/har-fusion/runs/vjepa")
    ap.add_argument("--txt_log_dir", type=str, default="/workspace/trainheads/runs/vjepa_only")
    ap.add_argument("--run_name",    type=str, default=None)
    ap.add_argument("--labelmap",    type=str, default="/workspace/har-fusion/runs/fusion/14cls_fusion_v1/label2id.json")

    # model/data basics
    ap.add_argument("--T",          type=int,   default=64)
    ap.add_argument("--d_in",       type=int,   default=1408)
    ap.add_argument("--d_model",    type=int,   default=1408)
    ap.add_argument("--n_heads",    type=int,   default=8)
    ap.add_argument("--depth",      type=int,   default=2)
    ap.add_argument("--ff_mult",    type=int,   default=4)
    ap.add_argument("--dropout",    type=float, default=0.1)

    # TSN settings
    ap.add_argument("--tsn_segments", type=int, default=8)
    ap.add_argument("--tsn_snippet",  type=int, default=8)

    # training knobs
    ap.add_argument("--epochs",       type=int,   default=30)
    ap.add_argument("--batch_size",   type=int,   default=128)
    ap.add_argument("--num_workers",  type=int,   default=12)
    ap.add_argument("--lr",           type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=0.05)
    ap.add_argument("--amp",          action="store_true")
    ap.add_argument("--seed",         type=int,   default=42)
    ap.add_argument("--grad_clip",    type=float, default=1.0,
                    help="Max grad-norm for clipping (None to disable)")

    args = ap.parse_args()

    # === ensure run_dir exists on args ===
    if not getattr(args, "run_name", None):
        args.run_name = "run_" + time.strftime("%Y%m%d_%H%M%S")
    if not getattr(args, "log_dir", None):
        raise ValueError("log_dir must be provided")
    args.run_dir = str(Path(args.log_dir) / args.run_name)

    # --- text logging: mirror the CoMotion style ---
    os.makedirs(args.txt_log_dir, exist_ok=True)
    log_path, log_fh, _ = start_text_logging(
        out_dir=args.txt_log_dir,
        prefix=f"vjepa_only_{args.run_name}",
        ts=args.run_name
    )
    print("=" * 100)
    print(f"[RUN START] {args.run_name} | log={log_path}")
    print(f"[RUN DIR] {args.run_dir}")
    print("Args:", json.dumps(vars(args), indent=2, sort_keys=False))
    print("=" * 100)

    try:
        if not args.train_csv or not args.val_csv:
            raise SystemExit("Missing --train_csv and/or --val_csv.")
        train(args)
    finally:
        try:
            log_fh.flush()
            log_fh.close()
        except Exception:
            pass


