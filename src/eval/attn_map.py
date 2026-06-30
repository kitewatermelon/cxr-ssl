"""Quick attention map visualization for DINOv2 checkpoint."""

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image

from stable_pretraining.methods import DINOv2
from data.mimic_cxr import MIMICCXRDataset

CKPT = "/home/yspark/cxr-ssl/cxr-ssl/92yv4b9t/checkpoints/epoch=162-step=469766.ckpt"
OUT  = "/home/yspark/cxr-ssl/figures/attn_map_epoch162.png"
N_IMAGES = 6
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ── 모델 로드 ──────────────────────────────────────────────────────────────
import torch

ckpt = torch.load(CKPT, map_location=DEVICE, weights_only=False)
sd = ckpt['state_dict']
# "model." prefix 제거
sd = {k[len("model."):]: v for k, v in sd.items() if k.startswith("model.")}

inner = DINOv2("vit_small_patch16_224")
inner.load_state_dict(sd, strict=False)
inner.eval().to(DEVICE)
vit = inner.backbone.teacher

# ── attention hook ─────────────────────────────────────────────────────────
attn_store = {}

def _hook(module, inp, out):
    # timm ViT Attention: out = (x,) or x depending on version
    # We need the attention weights → re-run softmax on qk
    B, N, C = inp[0].shape
    qkv = module.qkv(inp[0]).reshape(B, N, 3, module.num_heads, C // module.num_heads)
    qkv = qkv.permute(2, 0, 3, 1, 4)
    q, k = qkv[0], qkv[1]
    scale = (C // module.num_heads) ** -0.5
    attn = (q @ k.transpose(-2, -1)) * scale
    attn = attn.softmax(dim=-1)          # [B, heads, N, N]
    attn_store['attn'] = attn.detach().cpu()

# 마지막 블록의 attention에 hook
hook_handle = vit.blocks[-1].attn.register_forward_hook(_hook)

# ── 전처리 ─────────────────────────────────────────────────────────────────
from torchvision import transforms
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3),
])

ds = MIMICCXRDataset(root="/mnt/nvme1/mimic-cxr-jpg", split="validate",
                     label_csv="chexpert", transform=None)

# ── 시각화 ─────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(N_IMAGES, 3, figsize=(9, N_IMAGES * 3))
fig.suptitle(f"DINOv2 Attention Maps — Epoch 162", fontsize=13)

patch_size = 16
n_patches  = (224 // patch_size) ** 2  # 196

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

    attn = attn_store['attn'][0]          # [heads, N+1, N+1]
    # CLS token이 각 패치에 주는 attention
    cls_attn = attn[:, 0, 1:n_patches+1] # [heads, 196]
    mean_attn = cls_attn.mean(0)          # [196]  head 평균
    h = w = 224 // patch_size
    attn_map = mean_attn.reshape(h, w).numpy()
    attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min() + 1e-8)
    attn_up  = np.array(Image.fromarray((attn_map * 255).astype(np.uint8)).resize((224, 224), Image.BILINEAR)) / 255.0

    orig = np.array(img_pil.resize((224, 224)))
    orig_rgb = np.stack([orig]*3, axis=-1) if orig.ndim == 2 else orig
    orig_norm = orig_rgb / orig_rgb.max()

    # overlay
    heat = cm.jet(attn_up)[..., :3]
    overlay = 0.5 * orig_norm + 0.5 * heat

    row = collected
    axes[row, 0].imshow(orig_norm, cmap='gray')
    axes[row, 0].set_title("Original", fontsize=8)
    axes[row, 1].imshow(attn_up, cmap='jet')
    axes[row, 1].set_title("Attention (mean heads)", fontsize=8)
    axes[row, 2].imshow(overlay)
    axes[row, 2].set_title("Overlay", fontsize=8)
    for ax in axes[row]: ax.axis('off')

    collected += 1

hook_handle.remove()
plt.tight_layout()
plt.savefig(OUT, dpi=120, bbox_inches='tight')
print(f"Saved: {OUT}")
