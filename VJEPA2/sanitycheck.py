import torch, pathlib
import torch.nn.functional as F, random

sample_root = pathlib.Path("/workspace/results/vjepa/Turn sheets")

pt = random.choice(list(sample_root.rglob("*.pt")))
x  = torch.load(pt)                       # (64, 1408)

sim_adjacent = F.cosine_similarity(x[:-1], x[1:], dim=1)  # 63 values
print(f"\n{pt.name}  mean adjacent-frame cosine :", sim_adjacent.mean().item())

i = 1
for pt in sample_root.rglob("*.pt"):
    x = torch.load(pt)                    # (64, 1408) fp16
    print(f"\n{pt.relative_to(sample_root)}")
    print("  shape :", tuple(x.shape))
    print("  dtype :", x.dtype)
    print("  min/max :", float(x.min()), float(x.max()))
    print("  mean L2 per-frame :", x.norm(dim=1).mean().item())
    i += 1
    if i == 7:
        break
