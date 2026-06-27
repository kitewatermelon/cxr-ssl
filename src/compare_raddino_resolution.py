"""
Compare RAD-DINO CLS-token features extracted at 224x224 vs 518x518
on the MIMIC-CXR validation set.

Images: original .jpg, resize-only (no crop, no augmentation).

Usage:
    python cxr-ssl/src/compare_raddino_resolution.py \
        --data_dir   /mnt/nvme1/mimic-cxr-jpg \
        --batch_size 32 \
        --num_workers 8
"""

import os, sys, argparse
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd
from transformers import AutoModel


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)


def make_transform(size: int):
    return transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


class MIMICValJPG(Dataset):
    def __init__(self, root: str):
        self.root = root
        split_df = pd.read_csv(os.path.join(root, "mimic-cxr-2.0.0-split.csv.gz"))
        split_df = split_df[split_df["split"] == "validate"].reset_index(drop=True)
        meta_df  = pd.read_csv(os.path.join(root, "mimic-cxr-2.0.0-metadata.csv.gz"))
        frontal  = meta_df[meta_df["ViewPosition"].isin(["PA", "AP"])][["dicom_id"]]
        self.df  = split_df.merge(frontal, on="dicom_id").reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def _path(self, row):
        sid  = str(int(row["subject_id"]))
        stid = str(int(row["study_id"]))
        did  = str(row["dicom_id"])
        return os.path.join(self.root, "files", f"p{sid[:2]}", f"p{sid}", f"s{stid}", f"{did}.jpg")

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        return Image.open(self._path(row)).convert("RGB")


def collate_dual(batch, tf_small, tf_native):
    return (
        torch.stack([tf_small(img)  for img in batch]),
        torch.stack([tf_native(img) for img in batch]),
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir",    required=True)
    p.add_argument("--batch_size",  type=int, default=32)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--small_size",  type=int, default=224)
    p.add_argument("--native_size", type=int, default=518)
    args = p.parse_args()

    print("Loading RAD-DINO from HuggingFace...")
    model = AutoModel.from_pretrained("microsoft/rad-dino")
    model = model.cuda().eval()

    tf_small  = make_transform(args.small_size)
    tf_native = make_transform(args.native_size)

    ds = MIMICValJPG(args.data_dir)
    print(f"Validation set: {len(ds)} images")

    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        collate_fn=lambda b: collate_dual(b, tf_small, tf_native),
        pin_memory=True,
    )

    feats_small, feats_native = [], []

    print(f"Extracting features at {args.small_size} and {args.native_size}...")
    with torch.no_grad():
        for i, (imgs_small, imgs_native) in enumerate(loader):
            out_small  = model(pixel_values=imgs_small.cuda())
            out_native = model(pixel_values=imgs_native.cuda())

            feats_small.append(out_small.last_hidden_state[:, 0].cpu())    # CLS token
            feats_native.append(out_native.last_hidden_state[:, 0].cpu())

            if (i + 1) % 10 == 0:
                done = min((i + 1) * args.batch_size, len(ds))
                print(f"  {done}/{len(ds)}")

    feats_small  = torch.cat(feats_small,  dim=0)   # (N, D)
    feats_native = torch.cat(feats_native, dim=0)   # (N, D)

    cos_sim = F.cosine_similarity(feats_small, feats_native, dim=1)

    print(f"\n=== RAD-DINO: {args.small_size} vs {args.native_size} (native) ===")
    print(f"  N          : {len(cos_sim)}")
    print(f"  Mean cosine: {cos_sim.mean():.4f}")
    print(f"  Std        : {cos_sim.std():.4f}")
    print(f"  Min        : {cos_sim.min():.4f}")
    print(f"  Median     : {cos_sim.median():.4f}")
    print(f"  Max        : {cos_sim.max():.4f}")
    print(f"  > 0.99     : {(cos_sim > 0.99).float().mean()*100:.1f}%")
    print(f"  > 0.95     : {(cos_sim > 0.95).float().mean()*100:.1f}%")
    print(f"  > 0.90     : {(cos_sim > 0.90).float().mean()*100:.1f}%")

    norm_s = feats_small.norm(dim=1)
    norm_n = feats_native.norm(dim=1)
    print(f"\n  L2 norm ({args.small_size}): {norm_s.mean():.3f} ± {norm_s.std():.3f}")
    print(f"  L2 norm ({args.native_size}): {norm_n.mean():.3f} ± {norm_n.std():.3f}")

    # NN retrieval: 224 feature → find nearest in 518 feature space
    fn_s = F.normalize(feats_small,  dim=1)
    fn_n = F.normalize(feats_native, dim=1)
    chunk, correct = 256, 0
    for i in range(0, len(fn_s), chunk):
        q   = fn_s[i:i+chunk]
        nn  = (q @ fn_n.T).argmax(dim=1)
        gt  = torch.arange(i, i + len(q))
        correct += (nn == gt).sum().item()

    print(f"\n  NN R@1 ({args.small_size}→{args.native_size}): {correct/len(fn_s)*100:.1f}%")
    print(f"  (100% = same image retrieved across resolutions)")


if __name__ == "__main__":
    main()
