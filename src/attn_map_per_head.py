"""Per-head attention map visualization for DINOv2 checkpoint."""

import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms

sys.path.insert(0, '/home/yspark/cxr-ssl/src')
from stable_pretraining.methods import DINOv2
from mimic_cxr import MIMICCXRDataset

CKPT     = "/home/yspark/cxr-ssl/cxr-ssl/92yv4b9t/checkpoints/epoch=163-step=472648.ckpt"
OUT      = "/home/yspark/cxr-ssl/figures/attn_map_per_head_raw_epoch163.png"
N_IMAGES = 4
DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"

# ── 모델 로드 ──────────────────────────────────────────────────────────────
ckpt = torch.load(CKPT, map_location=DEVICE, weights_only=False)
sd   = {k[len("model."):]: v for k, v in ckpt['state_dict'].items() if k.startswith("model.")}
model = DINOv2("vit_small_patch16_224")
model.load_state_dict(sd, strict=False)
model.eval().to(DEVICE)
vit = model.backbone.teacher

NUM_HEADS  = vit.blocks[-1].attn.num_heads  # 6
PATCH_SIZE = 16
N_PATCHES  = (224 // PATCH_SIZE) ** 2       # 196
H = W = 224 // PATCH_SIZE                   # 14

# ── attention hook ─────────────────────────────────────────────────────────
attn_store = {}

def _hook(module, inp, out):
    B, N, C = inp[0].shape
    head_dim = C // module.num_heads
    qkv = module.qkv(inp[0]).reshape(B, N, 3, module.num_heads, head_dim)
    qkv = qkv.permute(2, 0, 3, 1, 4)
    q, k = qkv[0], qkv[1]
    attn = (q @ k.transpose(-2, -1)) * (head_dim ** -0.5)
    attn = attn.softmax(dim=-1)   # [B, heads, N, N]
    attn_store['attn'] = attn.detach().cpu()

hook = vit.blocks[-1].attn.register_forward_hook(_hook)

# ── 전처리 ─────────────────────────────────────────────────────────────────
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3),
])

ds = MIMICCXRDataset(root="/mnt/nvme1/mimic-cxr-jpg", split="validate",
                     label_csv="chexpert", transform=None)

# ── 시각화: rows=이미지, cols=original + 6 heads ──────────────────────────
N_COLS = 1 + NUM_HEADS   # 7
fig, axes = plt.subplots(N_IMAGES, N_COLS, figsize=(N_COLS * 3.5, N_IMAGES * 3.5))
fig.suptitle(f"DINOv2 Per-Head Attention Maps — Epoch 162  (last block, CLS→patch)",
             fontsize=11, y=1.01)

col_titles = ["Original"] + [f"Head {i}" for i in range(NUM_HEADS)]
for j, t in enumerate(col_titles):
    axes[0, j].set_title(t, fontsize=9, fontweight='bold')

collected = 0
idx = 0
while collected < N_IMAGES:
    sample = ds[idx]; idx += 1
    img_pil = sample[0] if isinstance(sample, tuple) else sample['image']
    if not isinstance(img_pil, Image.Image):
        continue

    tensor = preprocess(img_pil).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        _ = vit(tensor)

    attn = attn_store['attn'][0]           # [heads, N+1, N+1]
    cls_attn = attn[:, 0, 1:N_PATCHES+1]  # [heads, 196]

    orig = np.array(img_pil.resize((224, 224)))
    orig_norm = orig / orig.max() if orig.max() > 0 else orig

    row = collected
    axes[row, 0].imshow(orig_norm, cmap='gray')
    axes[row, 0].axis('off')

    for h in range(NUM_HEADS):
        a = cls_attn[h].numpy().reshape(H, W)  # 14×14 raw
        im = axes[row, h + 1].imshow(a, cmap='jet', interpolation='nearest')
        # 각 셀에 값 표시
        vmin, vmax = a.min(), a.max()
        for r in range(H):
            for c in range(W):
                v = a[r, c]
                brightness = (v - vmin) / (vmax - vmin + 1e-8)
                color = 'white' if brightness < 0.6 else 'black'
                axes[row, h + 1].text(c, r, f'{v:.3f}', ha='center', va='center',
                                      fontsize=3.2, color=color)
        axes[row, h + 1].axis('off')

    collected += 1

hook.remove()
plt.tight_layout()
plt.savefig(OUT, dpi=130, bbox_inches='tight')
print(f"Saved: {OUT}")
