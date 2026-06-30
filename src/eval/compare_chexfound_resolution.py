"""
Compare CheXFound CLS-token features extracted at 224x224 vs 512x512
on the MIMIC-CXR validation set.

Images: original .jpg, resize-only (no crop, no augmentation).
Normalisation: ImageNet mean/std (what CheXFound expects).

Usage:
    python cxr-ssl/src/compare_chexfound_resolution.py \
        --data_dir     /mnt/nvme1/mimic-cxr-jpg \
        --ckpt         cxr-ssl/CheXFound/CheXFound/teacher_checkpoint.pth \
        --config       cxr-ssl/CheXFound/CheXFound/config.yaml \
        --chexfound_dir cxr-ssl/CheXFound \
        --batch_size   32 \
        --num_workers  8
"""

import os, sys, argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd


# ---------------------------------------------------------------------------
# Dataset — original .jpg, resize only
# ---------------------------------------------------------------------------

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

def make_transform(size: int):
    return transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


class MIMICValJPG(Dataset):
    """Loads original MIMIC-CXR .jpg validation images (frontal only)."""

    def __init__(self, root: str):
        self.root = root

        split_df = pd.read_csv(os.path.join(root, "mimic-cxr-2.0.0-split.csv.gz"))
        split_df = split_df[split_df["split"] == "validate"].reset_index(drop=True)

        meta_df = pd.read_csv(os.path.join(root, "mimic-cxr-2.0.0-metadata.csv.gz"))
        frontal = meta_df[meta_df["ViewPosition"].isin(["PA", "AP"])][["dicom_id"]]
        self.df = split_df.merge(frontal, on="dicom_id").reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def _path(self, row) -> str:
        sid  = str(int(row["subject_id"]))
        stid = str(int(row["study_id"]))
        did  = str(row["dicom_id"])
        return os.path.join(self.root, "files", f"p{sid[:2]}", f"p{sid}", f"s{stid}", f"{did}.jpg")

    def __getitem__(self, idx):
        row  = self.df.iloc[idx]
        path = self._path(row)
        img  = Image.open(path).convert("RGB")
        return img   # raw PIL; collation applies transform per resolution


def collate_dual(batch, tf224, tf512):
    imgs224 = torch.stack([tf224(img) for img in batch])
    imgs512 = torch.stack([tf512(img) for img in batch])
    return imgs224, imgs512


# ---------------------------------------------------------------------------
# Model loader (identical to train_rcjit_mae.py)
# ---------------------------------------------------------------------------

def load_chexfound(ckpt_path, config_path):
    import warnings, logging, contextlib, io
    warnings.filterwarnings("ignore")
    logging.disable(logging.CRITICAL)

    from chexfound.eval.setup import build_model_for_eval
    from chexfound.utils.config import setup

    args = argparse.Namespace(
        config_file=config_path, pretrained_weights=ckpt_path,
        output_dir="", opts=[]
    )
    with contextlib.redirect_stderr(io.StringIO()):
        config = setup(args)
        config.student.block_chunks = 0
        model  = build_model_for_eval(config, ckpt_path)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir",      required=True)
    p.add_argument("--ckpt",          required=True)
    p.add_argument("--config",        required=True)
    p.add_argument("--chexfound_dir", required=True)
    p.add_argument("--batch_size",    type=int, default=32)
    p.add_argument("--num_workers",   type=int, default=8)
    args = p.parse_args()

    sys.path.insert(0, args.chexfound_dir)

    print("Loading CheXFound model...")
    model = load_chexfound(args.ckpt, args.config)

    tf224 = make_transform(224)
    tf512 = make_transform(512)

    ds = MIMICValJPG(args.data_dir)
    print(f"Validation set: {len(ds)} images")

    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        collate_fn=lambda b: collate_dual(b, tf224, tf512),
        pin_memory=True,
    )

    feats224, feats512 = [], []

    print("Extracting features...")
    with torch.no_grad():
        for i, (imgs224, imgs512) in enumerate(loader):
            imgs224 = imgs224.cuda()
            imgs512 = imgs512.cuda()

            out224 = model.forward_features(imgs224)
            out512 = model.forward_features(imgs512)

            feats224.append(out224["x_norm_clstoken"].cpu())
            feats512.append(out512["x_norm_clstoken"].cpu())

            if (i + 1) % 10 == 0:
                print(f"  {(i+1)*args.batch_size}/{len(ds)}")

    feats224 = torch.cat(feats224, dim=0)   # (N, 1024)
    feats512 = torch.cat(feats512, dim=0)   # (N, 1024)

    # ── Cosine similarity (per-image) ────────────────────────────────────
    cos_sim = F.cosine_similarity(feats224, feats512, dim=1)   # (N,)

    print("\n=== CheXFound: 224 vs 512 feature cosine similarity ===")
    print(f"  N images     : {len(cos_sim)}")
    print(f"  Mean         : {cos_sim.mean():.4f}")
    print(f"  Std          : {cos_sim.std():.4f}")
    print(f"  Min          : {cos_sim.min():.4f}")
    print(f"  Median       : {cos_sim.median():.4f}")
    print(f"  Max          : {cos_sim.max():.4f}")
    print(f"  > 0.99       : {(cos_sim > 0.99).float().mean()*100:.1f}%")
    print(f"  > 0.95       : {(cos_sim > 0.95).float().mean()*100:.1f}%")
    print(f"  > 0.90       : {(cos_sim > 0.90).float().mean()*100:.1f}%")

    # ── L2 norm 비교 ─────────────────────────────────────────────────────
    norm224 = feats224.norm(dim=1)
    norm512 = feats512.norm(dim=1)
    print(f"\n  L2 norm (224): {norm224.mean():.3f} ± {norm224.std():.3f}")
    print(f"  L2 norm (512): {norm512.mean():.3f} ± {norm512.std():.3f}")

    # ── 분포 간 정렬 확인: 224 feature로 NN retrieval 시 512 feature 매칭률 ──
    f224_n = F.normalize(feats224, dim=1)
    f512_n = F.normalize(feats512, dim=1)

    chunk = 256
    correct = 0
    for i in range(0, len(f224_n), chunk):
        q = f224_n[i:i+chunk]               # (chunk, 1024)
        sims = q @ f512_n.T                 # (chunk, N)
        nn_idx = sims.argmax(dim=1)
        gt_idx = torch.arange(i, i + len(q))
        correct += (nn_idx == gt_idx).sum().item()

    recall = correct / len(f224_n) * 100
    print(f"\n  NN-retrieval R@1 (224→512): {recall:.1f}%")
    print("  (100% = 224 feature retrieves same image from 512 feature space)")


if __name__ == "__main__":
    main()
