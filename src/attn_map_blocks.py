"""Block-wise attention map comparison for DINOv2 checkpoint.
rows = blocks (0,2,4,6,8,10,11), cols = heads (6)
"""

import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms

sys.path.insert(0, '/home/yspark/cxr-ssl/src')
from stable_pretraining.methods import DINOv2
from mimic_cxr import MIMICCXRDataset

import glob, os
_ckpts = sorted(glob.glob("/home/yspark/cxr-ssl/cxr-ssl/92yv4b9t/checkpoints/*.ckpt"))
CKPT   = _ckpts[-1]
_epoch = os.path.basename(CKPT).split("=")[1].split("-")[0]
OUT    = f"/home/yspark/cxr-ssl/figures/attn_map_blocks_epoch{_epoch}.png"
print(f"Using: {CKPT}")
DEVICE    = "cuda" if torch.cuda.is_available() else "cpu"
BLOCKS    = [0, 2, 4, 6, 8, 10, 11]   # 비교할 블록 인덱스
N_IMAGES  = 2

# ── 모델 로드 ──────────────────────────────────────────────────────────────
ckpt = torch.load(CKPT, map_location=DEVICE, weights_only=False)
sd   = {k[len("model."):]: v for k, v in ckpt['state_dict'].items() if k.startswith("model.")}
model = DINOv2("vit_small_patch16_224")
model.load_state_dict(sd, strict=False)
model.eval().to(DEVICE)
vit = model.backbone.teacher

NUM_HEADS = vit.blocks[0].attn.num_heads  # 6
H = W = 14

# ── 모든 지정 블록에 hook 등록 ─────────────────────────────────────────────
attn_store = {}

def make_hook(block_idx):
    def _hook(module, inp, out):
        B, N, C = inp[0].shape
        head_dim = C // module.num_heads
        qkv = module.qkv(inp[0]).reshape(B, N, 3, module.num_heads, head_dim)
        q, k = qkv.permute(2, 0, 3, 1, 4)[0], qkv.permute(2, 0, 3, 1, 4)[1]
        attn = (q @ k.transpose(-2, -1)) * (head_dim ** -0.5)
        attn_store[block_idx] = attn.softmax(dim=-1).detach().cpu()
    return _hook

hooks = [vit.blocks[b].attn.register_forward_hook(make_hook(b)) for b in BLOCKS]

# ── 전처리 ─────────────────────────────────────────────────────────────────
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3),
])

ds = MIMICCXRDataset(root="/mnt/nvme1/mimic-cxr-jpg", split="validate",
                     label_csv="chexpert", transform=None)

# ── 이미지 N_IMAGES장 수집 ────────────────────────────────────────────────
images, tensors = [], []
idx = 0
while len(images) < N_IMAGES:
    sample = ds[idx]; idx += 1
    img_pil = sample[0] if isinstance(sample, tuple) else sample['image']
    if not isinstance(img_pil, Image.Image):
        continue
    images.append(img_pil)
    tensors.append(preprocess(img_pil).unsqueeze(0).to(DEVICE))

# ── figure: 이미지마다 서브플롯 그룹 (rows=블록, cols=original+heads) ─────
N_COLS = 1 + NUM_HEADS   # 7
N_ROWS = len(BLOCKS)

for img_i, (img_pil, tensor) in enumerate(zip(images, tensors)):
    with torch.no_grad():
        _ = vit(tensor)

    fig, axes = plt.subplots(N_ROWS, N_COLS, figsize=(N_COLS * 2.2, N_ROWS * 2.2))
    fig.suptitle(f"Block-wise Attention — Image {img_i+1}  (Epoch 163)", fontsize=11)

    # 열 헤더
    axes[0, 0].set_title("Original", fontsize=8, fontweight='bold')
    for h in range(NUM_HEADS):
        axes[0, h+1].set_title(f"Head {h}", fontsize=8, fontweight='bold')

    orig = np.array(img_pil.resize((224, 224)))
    orig_norm = orig / orig.max() if orig.max() > 0 else orig

    for row, b in enumerate(BLOCKS):
        attn = attn_store[b][0]            # [heads, N+1, N+1]
        n_patch = attn.shape[-1] - 1
        hw = int(n_patch ** 0.5)
        cls_attn = attn[:, 0, 1:n_patch+1]  # [heads, n_patch]

        # original (첫 열, 블록마다 동일)
        axes[row, 0].imshow(orig_norm, cmap='gray')
        axes[row, 0].set_ylabel(f"Block {b}", fontsize=8, rotation=0,
                                labelpad=36, va='center')
        axes[row, 0].axis('off')

        for h in range(NUM_HEADS):
            a = cls_attn[h].numpy().reshape(hw, hw)
            axes[row, h+1].imshow(a, cmap='jet', interpolation='nearest')
            axes[row, h+1].axis('off')

    plt.tight_layout()
    out_path = OUT.replace('.png', f'_img{img_i+1}.png')
    plt.savefig(out_path, dpi=130, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path}")

for h in hooks:
    h.remove()
