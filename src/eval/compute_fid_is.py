"""
FID & IS computation for RCJiT-MAE flow-matching model.

Pipeline:
  val image (cond) → MAE encoder → CLS pool → JiT denoiser → generated image
  FID: real val images vs generated images (InceptionV3)
  IS:  generated images only
"""

import os, argparse
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import torchvision.utils as vutils
from tqdm import tqdm

from rcjit.denoiser import RCJiTDenoiser
from data.mimic_cxr import MIMICCXRDataset

# ── config ────────────────────────────────────────────────────────────────
CKPT         = "/home/yspark/cxr-ssl/output/rcjit_s16_mae_pool_cxr/checkpoints/stepstep=0300000.ckpt"
ENCODER_CKPT = "/home/yspark/cxr-ssl/cxr-ssl/ssnlqx8u/checkpoints/epoch=799-step=576000.ckpt"
DATA_DIR     = "/mnt/nvme1/mimic-cxr-jpg"
OUT_REAL     = "/tmp/rcjit_fid/real"
OUT_GEN      = "/tmp/rcjit_fid/gen"
BATCH_SIZE   = 16
ODE_STEPS    = 50
CFG          = 1.0
USE_EMA      = True        # ema_params1 (decay=0.9999) 사용
DEVICE       = "cuda"
os.makedirs(OUT_REAL, exist_ok=True)
os.makedirs(OUT_GEN,  exist_ok=True)

# ── 전처리 ([-1,1] range) ─────────────────────────────────────────────────
preprocess = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

class MIMICWrapper(Dataset):
    def __init__(self, split):
        self.ds = MIMICCXRDataset(root=DATA_DIR, split=split,
                                  label_csv="chexpert", transform=None,
                                  frontal_only=False)
    def __len__(self):
        return len(self.ds)
    def __getitem__(self, idx):
        sample = self.ds[idx]
        img = sample[0] if isinstance(sample, tuple) else sample['image']
        if not isinstance(img, Image.Image):
            return self.__getitem__((idx + 1) % len(self))
        return preprocess(img)

# ── 모델 로드 ─────────────────────────────────────────────────────────────
print("Loading checkpoint...")
raw_ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
hp = raw_ckpt['hyper_parameters']

import argparse
args = argparse.Namespace(
    img_size        = hp['img_size'],
    model_variant   = hp['model_variant'],
    ctx_mode        = "pool",
    encoder_type    = "mae",
    encoder_ckpt    = ENCODER_CKPT,
    attn_dropout    = hp['attn_dropout'],
    proj_dropout    = hp['proj_dropout'],
    cond_drop_prob  = hp['cond_drop_prob'],
    P_mean          = hp['P_mean'],
    P_std           = hp['P_std'],
    noise_scale     = hp['noise_scale'],
    t_eps           = hp['t_eps'],
    ema_decay1      = hp['ema_decay1'],
    ema_decay2      = hp['ema_decay2'],
    sampling_method = hp['ode_method'],
    num_sampling_steps = ODE_STEPS,
    cfg             = CFG,
    interval_min    = hp['interval_min'],
    interval_max    = hp['interval_max'],
)

denoiser = RCJiTDenoiser(args)

# state_dict 로드 (denoiser. prefix 제거)
sd = {k[len("denoiser."):]: v
      for k, v in raw_ckpt['state_dict'].items()
      if k.startswith("denoiser.")}
denoiser.load_state_dict(sd, strict=False)

# EMA params 적용 (dict 형태: {param_name: tensor})
if USE_EMA and 'ema_params1' in raw_ckpt:
    print("Applying EMA params (decay=0.9999)...")
    ema_dict = raw_ckpt['ema_params1']
    named = dict(denoiser.named_parameters())
    for name, ema_tensor in ema_dict.items():
        if name in named and isinstance(ema_tensor, torch.Tensor):
            named[name].data.copy_(ema_tensor)

denoiser.eval().to(DEVICE)
denoiser.steps = ODE_STEPS
denoiser.cfg_scale = CFG
print(f"Model loaded. ODE steps={ODE_STEPS}, CFG={CFG}")

# ── 이미지 생성 & 저장 ────────────────────────────────────────────────────
loader = DataLoader(MIMICWrapper("validate"), batch_size=BATCH_SIZE,
                    shuffle=False, num_workers=8, pin_memory=True)

total = 0
print(f"\nGenerating {len(loader.dataset)} images...")
with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
    for batch_idx, imgs in enumerate(tqdm(loader)):
        imgs = imgs.to(DEVICE)
        gen  = denoiser.generate(imgs)   # cond_img → generated

        # [-1,1] → [0,255]
        real_u8 = ((imgs.clamp(-1,1) * 0.5 + 0.5) * 255).byte().cpu()
        gen_u8  = ((gen .clamp(-1,1) * 0.5 + 0.5) * 255).byte().cpu()

        for i in range(imgs.size(0)):
            idx = total + i
            # real: grayscale (3ch → 1ch for saving)
            r = real_u8[i].permute(1,2,0).numpy()
            Image.fromarray(r.astype(np.uint8)).save(f"{OUT_REAL}/{idx:05d}.png")
            g = gen_u8[i].permute(1,2,0).numpy()
            Image.fromarray(g.astype(np.uint8)).save(f"{OUT_GEN}/{idx:05d}.png")

        total += imgs.size(0)

print(f"\nSaved {total} real + {total} generated images.")

# ── FID & IS 계산 ─────────────────────────────────────────────────────────
print("\nComputing FID and IS with torch_fidelity...")
from torch_fidelity import calculate_metrics

metrics = calculate_metrics(
    input1=OUT_GEN,
    input2=OUT_REAL,
    cuda=True,
    isc=True,         # Inception Score
    fid=True,         # FID
    verbose=True,
)

print("\n" + "="*40)
print(f"  FID  : {metrics['frechet_inception_distance']:.4f}")
print(f"  IS   : {metrics['inception_score_mean']:.4f} ± {metrics['inception_score_std']:.4f}")
print("="*40)
