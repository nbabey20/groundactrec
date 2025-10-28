import os, json, math, argparse, random
from pathlib import Path
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import average_precision_score
from tqdm import tqdm

# ----------------------------
# Utilities
# ----------------------------
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
    def __init__(self, csv_path, label2id, T=64, d_in=1408, drop_missing=True):
        self.df = pd.read_csv(csv_path)
        needed = {"clip", "label", "path_vjepa"}
        if not needed.issubset(set(self.df.columns)):
            raise ValueError(f"CSV must have columns: {needed}. Got: {self.df.columns.tolist()}")

        rows = []
        for _, r in self.df.iterrows():
            p = str(r["path_vjepa"])
            if drop_missing and not exists(p):
                continue
            rows.append({
                "clip": str(r["clip"]),
                "label": str(r["label"]),
                "path_vjepa": p
            })
        if len(rows) == 0:
            raise RuntimeError(f"No valid rows in {csv_path}. Check paths.")

        self.rows = rows
        self.label2id = label2id
        self.T = T
        self.d_in = d_in

    def __len__(self): return len(self.rows)

    def __getitem__(self, idx):
        rec = self.rows[idx]
        feat = torch.load(rec["path_vjepa"], map_location="cpu")  # [T, D] float16 expected
        if feat.ndim != 2:
            raise RuntimeError(f"Expected 2D tensor [T, D], got {feat.shape} in {rec['path_vjepa']}")
        T_in, D_in = feat.shape
        # Fix dtype and ensure target shape
        feat = feat.to(torch.float32)
        if D_in != self.d_in:
            raise RuntimeError(f"D mismatch: expected {self.d_in}, got {D_in} for {rec['path_vjepa']}")
        # Pad/truncate time to self.T
        if T_in < self.T:
            pad = torch.zeros(self.T - T_in, self.d_in, dtype=feat.dtype)
            feat = torch.cat([feat, pad], dim=0)
        elif T_in > self.T:
            feat = feat[:self.T]
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
#fusion model, check skeleton dimensions to make sure everything fits, try residual connection with attention, also cross attention, self attention and residual of cross attention
#train and test 3 times, compare mean and standard deviation, for each model
#start writing paper, download template and create new proj on overleaf, invite everyone as editors, related works (1 subsection about vision foundation models, talk about different ones (vjepa, dino, convexnet, deep search with gemini, and section about action recoginition). Then write methodologies section, experimentals (setting of everything, dataset, baselines, model selection, seeds, metrics for evaluation, and explain task). Finally results section to present results, add as many citations as possible (at least 30) most about action recognition when fusing data and vision foundational models, vision foundational models with sensory fusion. Skeletons are recognized as sensory mode. Need fusion model, iterate a lot, play with model and train until get one that works better than the baseline models. 
#(extra task if we get everything else done on time) when extracting comotion skeletons, try to plot image with skeleton on top
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
        self.norm_images = nn.LayerNorm(d_model)
        self.norm_skeletons = nn.LayerNorm(d_skeletons)
        self.head_skeletons = nn.Linear(d_skeletons, d_middle)
        self.head_images = nn.Linear(d_model, d_middle)
        self.head = nn.Linear(d_middle, num_classes)

        nn.init.trunc_normal_(self.head.weight, std=0.02)
        nn.init.zeros_(self.head.bias)

    def forward(self, x, skeletons):  # x: [B, T, d_in]
        x = self.proj(x)  # [B, T, d_model]
        B, T, D = x.shape
        cls = self.cls_token.expand(B, 1, D)
        x = torch.cat([cls, x], dim=1)  # [B, T+1, D]
        x = self.pos_enc(x)
        x = self.encoder(x)
        cls_out = self.norm_images(x[:, 0])
        cls_skeletons = self.norm_skeletons(skeletons)
        x = self.head_images(cls_out)
        x = x + self.head_skeletons(cls_skeletons)
        logits = self.head(x)
        return logits

# ----------------------------
# Training / Evaluation
# ----------------------------
@torch.no_grad()
def evaluate(model, loader, device, num_classes, amp=False):
    model.eval()
    all_probs = []
    all_targets = []
    correct = 0
    total = 0
    scaler_ctx = torch.cuda.amp.autocast if amp else torch.cpu.amp.autocast  # cpu autocast is no-op
    with scaler_ctx():
        for feats, y in loader:
            feats = feats.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            logits = model(feats)
            probs = torch.softmax(logits, dim=-1)
            pred = probs.argmax(dim=-1)
            correct += (pred == y).sum().item()
            total += y.numel()
            all_probs.append(probs.detach().cpu())
            all_targets.append(y.detach().cpu())
    acc = correct / max(total, 1)
    probs = torch.cat(all_probs).numpy()
    targets = torch.cat(all_targets).numpy()
    # macro mAP over one-vs-rest
    y_true = torch.zeros((targets.shape[0], num_classes), dtype=torch.int32)
    y_true.scatter_(1, torch.tensor(targets).view(-1,1), 1)
    y_true = y_true.numpy()
    mAP = average_precision_score(y_true, probs, average="macro")
    return acc, mAP

def train(args):
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ---- Build label map from train CSV (or load if provided) ----
    train_df = pd.read_csv(args.train_csv)
    train_labels = [str(x) for x in train_df["label"].tolist()]
    if args.labelmap and exists(args.labelmap):
        label2id = load_label_map(args.labelmap)
    else:
        label2id = make_label_map(train_labels)
        if args.labelmap:
            save_json(label2id, args.labelmap)
    num_classes = len(label2id)
    print(f"Classes: {num_classes}  -> {label2id}")

    # ---- Datasets & loaders ----
    train_ds = VJEPAIndexDataset(args.train_csv, label2id, T=args.T, d_in=args.d_in, drop_missing=True)
    val_ds   = VJEPAIndexDataset(args.val_csv,   label2id, T=args.T, d_in=args.d_in, drop_missing=True)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=True)

    # ---- Model / Optim ----
    model = VideoOnlyAttnProbe(d_in=args.d_in, d_model=args.d_model, n_heads=args.n_heads,
                               depth=args.depth, ff_mult=args.ff_mult, num_classes=num_classes,
                               dropout=args.dropout, T=args.T).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = args.epochs * math.ceil(len(train_loader))
    warmup_steps = int(0.05 * total_steps)

    def lr_schedule(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scaler = torch.cuda.amp.GradScaler(enabled=args.amp and device == "cuda")
    criterion = nn.CrossEntropyLoss()

    # ---- Logging / ckpts ----
    out_dir = Path(args.log_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    best_map = -1.0
    global_step = 0

    # ---- Train ----
    for epoch in range(1, args.epochs+1):
        model.train()
        epoch_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", ncols=100)
        for feats, y in pbar:
            feats = feats.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            for g in optim.param_groups:
                base_lr = args.lr
                g["lr"] = base_lr * lr_schedule(global_step)

            optim.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=args.amp and device == "cuda"):
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

        # ---- Eval ----
        val_acc, val_map = evaluate(model, val_loader, device, num_classes, amp=args.amp)
        avg_loss = epoch_loss / max(len(train_loader), 1)
        print(f"[Epoch {epoch}] train_loss={avg_loss:.4f}  val_acc={val_acc:.4f}  val_mAP={val_map:.4f}")

        # ---- Save best ----
        if val_map > best_map:
            best_map = val_map
            ckpt = {
                "model": model.state_dict(),
                "args": vars(args),
                "label2id": label2id,
                "val_acc": val_acc,
                "val_mAP": val_map
            }
            torch.save(ckpt, out_dir / "best_video_only.ckpt")
            print(f"Saved best checkpoint to {out_dir/'best_video_only.ckpt'} (mAP={best_map:.4f})")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_csv", type=str, required=True)
    ap.add_argument("--val_csv",   type=str, required=True)
    ap.add_argument("--log_dir",   type=str, default="/workspace/trainheads/runs/vjepa_only")
    ap.add_argument("--labelmap",  type=str, default="/workspace/trainheads/runs/vjepa_only/label2id.json")

    # model/data
    ap.add_argument("--d_in",      type=int, default=1408)   # V-JEPA dim you observed
    ap.add_argument("--d_model",   type=int, default=1408)   # keep same or project
    ap.add_argument("--T",         type=int, default=64)     # tokens per clip
    ap.add_argument("--n_heads",   type=int, default=8)
    ap.add_argument("--depth",     type=int, default=2)
    ap.add_argument("--ff_mult",   type=int, default=4)
    ap.add_argument("--dropout",   type=float, default=0.1)

    # training
    ap.add_argument("--epochs",    type=int, default=30)
    ap.add_argument("--batch_size",type=int, default=128)
    ap.add_argument("--num_workers", type=int, default=6)
    ap.add_argument("--lr",        type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=0.05)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--amp",       action="store_true")
    ap.add_argument("--seed",      type=int, default=42)
    args = ap.parse_args()
    train(args)
