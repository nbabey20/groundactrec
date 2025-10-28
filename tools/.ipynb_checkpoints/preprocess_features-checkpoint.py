import argparse
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import sys
import os
import json
import time
from datetime import datetime
import warnings

# --- Py3.11 / NumPy 2.0+ compatibility shims ---
import numpy as _np
if not hasattr(_np, "bool"): _np.bool = _np.bool_
if not hasattr(_np, "int"): _np.int = int
if not hasattr(_np, "float"): _np.float = float
if not hasattr(_np, "complex"): _np.complex = complex
if not hasattr(_np, "object"): _np.object = object
if not hasattr(_np, "unicode"): _np.unicode = str
if not hasattr(_np, "str"): _np.str = str

import inspect as _inspect
if not hasattr(_inspect, "getargspec"):
    from collections import namedtuple as _namedtuple
    def _getargspec(func):
        fs = _inspect.getfullargspec(func)
        return _namedtuple("ArgSpec", ["args", "varargs", "keywords", "defaults"])(
            fs.args, fs.varargs, fs.varkw, fs.defaults
        )
    _inspect.getargspec = _getargspec
# --- End of shims ---

try:
    import smplx
    from torch.utils.data import Dataset, DataLoader
    import torch.nn as nn
    import torch.nn.functional as F
    from sklearn.metrics import top_k_accuracy_score
except ImportError as e:
    print(f"A required library is not installed: {e}")
    sys.exit(1)

# --- MODEL ARCHITECTURE (Identical to previous) ---
class LayerNorm(nn.Module):
    def __init__(self, dim): super().__init__(); self.gamma = nn.Parameter(torch.ones(dim)); self.register_buffer("beta", torch.zeros(dim))
    def forward(self, x): return F.layer_norm(x, x.shape[-1:], self.gamma, self.beta)
class SwiGLU(nn.Module):
    def __init__(self, in_dim, out_dim): super().__init__(); self.linear_gate = nn.Linear(in_dim, out_dim); self.linear_proj = nn.Linear(in_dim, out_dim)
    def forward(self, x): return F.silu(self.linear_gate(x)) * self.linear_proj(x)
class Attention(nn.Module):
    def __init__(self, d, h): super().__init__(); self.h=h; self.dh=d//h; self.s=self.dh**-0.5; self.qkv=nn.Linear(d,d*3); self.o=nn.Linear(d,d)
    def forward(self,x): B,T,C=x.shape; q,k,v=self.qkv(x).reshape(B,T,3,self.h,self.dh).permute(2,0,3,1,4); a=F.softmax((q@k.transpose(-2,-1))*self.s,-1); return self.o((a@v).transpose(1,2).reshape(B,T,C))
class ConfidenceAwareGating(nn.Module):
    def __init__(self,d): super().__init__(); self.mlp=nn.Sequential(nn.Linear(1,d//4), nn.GELU(), nn.Linear(d//4,d), nn.Sigmoid())
    def forward(self, f, c): return f * self.mlp(c.unsqueeze(-1))
class OMFormerLayer(nn.Module):
    def __init__(self, d, h, df): super().__init__(); self.a=Attention(d,h); self.f=SwiGLU(d,df); self.n1=LayerNorm(d); self.n2=LayerNorm(d)
    def forward(self, x): x=x+self.a(self.n1(x)); x=x+self.f(self.n2(x)); return x
class OMFormer(nn.Module):
    def __init__(self, nc, d, nl, nh, df, vfd, sfd):
        super().__init__(); self.nc=nc; self.vp=nn.Linear(vfd,d); self.sp=nn.Linear(sfd,d); self.cag=ConfidenceAwareGating(d); self.t=nn.ModuleList([OMFormerLayer(d,nh,df) for _ in range(nl)]); self.n=LayerNorm(d); self.ct=nn.Parameter(torch.randn(1,1,d)); self.pe=nn.Parameter(torch.randn(1,512,d)); self.h=nn.Linear(d,nc*4)
    def forward(self, vf, sx, sc):
        B,Tv,_=vf.shape; _,Ts,J,_=sx.shape; vt=self.vp(vf); st=self.sp(sx.reshape(B,Ts,J*3)); st=self.cag(st,sc.mean(-1)); t=torch.cat([vt,st],1); B,T,_=t.shape; ct=self.ct.expand(B,-1,-1); t=torch.cat((ct,t),1); t+=self.pe[:,:(T+1)]; [t:=l(t) for l in self.t]; return self.h(self.n(t[:,0])).view(B,self.nc,4)
class EvidentialLoss(nn.Module):
    def __init__(self, nc): super().__init__(); self.nc=nc
    def forward(self, logits, target): e=F.softplus(logits); a=e+1; S=torch.sum(a,1); return torch.sum(target*(torch.log(S)-torch.log(a))).mean()

# --- GPU-ACCELERATED DATASET (Identical to previous) ---
class OnTheFlyFusionDataset(Dataset):
    def __init__(self, csv_path, label_map, smpl_model, device):
        self.df=pd.read_csv(csv_path); self.label_map=label_map; self.smpl=smpl_model; self.dev=device; self.smpl.to(self.dev).eval()
    def __len__(self): return len(self.df)
    def __getitem__(self, idx):
        row=self.df.iloc[idx]; label_id=self.label_map.get(row["label"],-1)
        if label_id == -1: return None
        try:
            vf=torch.load(row["path_vjepa"],self.dev); cd=torch.load(row["path_skel"],self.dev)
            if 'j3d' in cd and 'conf' in cd: sx,sc=cd['j3d'],cd['conf']
            else:
                with torch.no_grad(): so=self.smpl(betas=cd['betas'].unsqueeze(0),body_pose=cd['pose'][:,3:].unsqueeze(0),global_orient=cd['pose'][:,:3].unsqueeze(0),transl=cd['trans'].unsqueeze(0)); sx=so.joints[:,:24,:].squeeze(0); sc=torch.ones_like(sx[...,0])
            return {"vf":vf.float(),"sx":sx.float(),"sc":sc.float(),"lbl":torch.tensor(label_id,dtype=torch.long,device=self.dev)}
        except Exception: return None

def custom_collate_fn(batch):
    batch = [item for item in batch if item is not None]
    if not batch: return None
    # Remap keys for default collate
    remapped_batch = [{"video_feats": item["vf"], "skel_xyz": item["sx"], "skel_conf": item["sc"], "label": item["lbl"]} for item in batch]
    return torch.utils.data.dataloader.default_collate(remapped_batch)

# --- TRAINING AND EVALUATION (Now with safeguards) ---
def run_validation(model, loader, criterion, num_classes, device):
    # --- FIX: Safeguard against empty validation loader ---
    if not loader or len(loader.dataset) == 0:
        print("Warning: Validation loader is empty. Skipping validation.")
        return 0.0, 0.0, 0.0, 0.0
        
    model.eval()
    total_loss = 0
    all_preds, all_labels, all_uncertainty = [], [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Validating", leave=False):
            if batch is None: continue
            logits = model(batch["video_feats"], batch["skel_xyz"], batch["skel_conf"])
            labels_one_hot = F.one_hot(batch["label"], num_classes=num_classes)
            loss = criterion(logits, labels_one_hot); total_loss += loss.item()
            e=F.softplus(logits); a=e+1; S=torch.sum(a,1); p=a/S.unsqueeze(1); u=num_classes/S
            all_preds.append(p.cpu().numpy()); all_labels.append(batch["label"].cpu().numpy()); all_uncertainty.append(u.cpu().numpy())

    # --- FIX: Safeguard against cases where all batches were invalid ---
    if not all_preds:
        print("Warning: No valid batches were found during validation. Returning zero metrics.")
        return 0.0, 0.0, 0.0, 0.0

    avg_loss = total_loss/len(loader); all_p=np.concatenate(all_preds); all_l=np.concatenate(all_labels); all_u=np.concatenate(all_uncertainty)
    t1=top_k_accuracy_score(all_l,all_p,k=1); t5=top_k_accuracy_score(all_l,all_p,k=5,labels=np.arange(num_classes)); avg_u=all_u.mean()
    return avg_loss, t1, t5, avg_u

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_dir = Path(args.run_dir); run_dir.mkdir(parents=True, exist_ok=True)
    
    # --- FIX: Build a comprehensive label map from BOTH train and val sets ---
    print("Building comprehensive label map...")
    train_df = pd.read_csv(args.train_csv)
    val_df = pd.read_csv(args.val_csv)
    all_labels = pd.concat([train_df['label'], val_df['label']]).unique()
    all_labels = sorted([label for label in all_labels if pd.notna(label)])
    label_map = {label: i for i, label in enumerate(all_labels)}
    num_classes = len(label_map)
    print(f"Found {num_classes} unique classes across train and val sets.")
    with open(run_dir / 'label_map.json', 'w') as f: json.dump(label_map, f, indent=2)

    print("Loading SMPL model to GPU...")
    smpl_model=smplx.SMPL(str(args.smpl_model_dir),gender='neutral',ext='pkl').to(device)
    
    train_ds=OnTheFlyFusionDataset(args.train_csv,label_map,smpl_model,device); val_ds=OnTheFlyFusionDataset(args.val_csv,label_map,smpl_model,device)
    train_loader=DataLoader(train_ds,args.batch_size,shuffle=True,num_workers=0,collate_fn=custom_collate_fn); val_loader=DataLoader(val_ds,args.batch_size,num_workers=0,collate_fn=custom_collate_fn)

    model=OMFormer(num_classes,args.d_model,args.n_layers,args.n_heads,args.d_ff,1024,24*3).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6:.2f}M")

    crit=EvidentialLoss(num_classes); opt=torch.optim.AdamW(model.parameters(),lr=args.lr); sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=args.epochs)
    
    best_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", leave=False):
            if batch is None: continue
            opt.zero_grad()
            logits = model(batch["video_feats"], batch["skel_xyz"], batch["skel_conf"])
            labels_one_hot = F.one_hot(batch["label"], num_classes=num_classes)
            loss = crit(logits, labels_one_hot); loss.backward(); opt.step()
        
        val_loss, top1, top5, uncertainty = run_validation(model, val_loader, crit, num_classes, device)
        sched.step()
        print(f"E{epoch}: V Loss:{val_loss:.3f}, Top1:{top1:.3f}, Top5:{top5:.3f}, Uncert:{uncertainty:.3f}")

        if top1 > best_acc: best_acc=top1; torch.save(model.state_dict(),run_dir/'best.pth'); print(f"  -> Best model saved: {best_acc:.3f}")

if __name__ == "__main__":
    p=argparse.ArgumentParser("OM-Fusion Final Trainer");p.add_argument("--train_csv",type=str,required=True);p.add_argument("--val_csv",type=str,required=True);p.add_argument("--labelmap",type=str,required=True, help="Note: This is now only used to check for backwards compatibility. A new map is generated in the run_dir.");p.add_argument("--smpl_model_dir",type=str,required=True);p.add_argument("--run_dir",type=str,default="runs/omf_final");p.add_argument("--d_model",type=int,default=512);p.add_argument("--n_layers",type=int,default=6);p.add_argument("--n_heads",type=int,default=8);p.add_argument("--d_ff",type=int,default=2048);p.add_argument("--epochs",type=int,default=100);p.add_argument("--batch_size",type=int,default=64);p.add_argument("--lr",type=float,default=1e-4);args=p.parse_args()
    main(args)