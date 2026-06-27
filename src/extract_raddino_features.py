"""
Extract RAD-DINO CLS-token features from MIMIC-CXR and save each as a
per-image .npy file co-located with the source image:

    mimic-cxr-jpg/files/p10/p10000032/s50414267/raddino_<dicom_id>.npy  (768,) float32

Resumable: already-existing files are skipped automatically.
Multi-GPU: splits work across all available GPUs with mp.spawn.

Usage:
    python cxr-ssl/src/extract_raddino_features.py \
        --data_dir   /mnt/nvme1/mimic-cxr-jpg \
        --batch_size 128 \
        --num_workers 8
"""

import os, argparse
import numpy as np
import torch
import torch.multiprocessing as mp
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd
from transformers import AutoModel
from tqdm import tqdm


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


def _img_dir(root: str, row) -> str:
    sid  = str(int(row["subject_id"]))
    stid = str(int(row["study_id"]))
    return os.path.join(root, "files", f"p{sid[:2]}", f"p{sid}", f"s{stid}")


def _feat_path(root: str, row) -> str:
    return os.path.join(_img_dir(root, row), f"raddino_{row['dicom_id']}.npy")


def _npy_path(root: str, row) -> str:
    return os.path.join(_img_dir(root, row), f"{row['dicom_id']}.npy")


class MIMICNpyDataset(Dataset):
    """Loads preprocessed 256×256 .npy grayscale files, skipping already-extracted."""

    def __init__(self, root: str, split: str, rank: int, world_size: int):
        self.root = root

        df = pd.read_csv(os.path.join(root, "mimic-cxr-2.0.0-split.csv.gz"))
        df = df[df["split"] == split].reset_index(drop=True)

        # assign rows to this GPU
        df = df.iloc[rank::world_size].reset_index(drop=True)

        # keep only rows where source .npy exists AND feature not yet extracted
        todo = [
            os.path.exists(_npy_path(root, row)) and not os.path.exists(_feat_path(root, row))
            for _, row in df.iterrows()
        ]
        self.df = df[todo].reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row  = self.df.iloc[idx]
        path = _npy_path(self.root, row)
        arr  = np.load(path)                          # (256, 256) uint8
        img  = Image.fromarray(arr, mode="L").convert("RGB")
        return TRANSFORM(img), str(row["dicom_id"]), str(int(row["subject_id"])), str(int(row["study_id"]))


def collate_fn(batch):
    imgs      = torch.stack([b[0] for b in batch])
    dicom_ids = [b[1] for b in batch]
    sids      = [b[2] for b in batch]
    stids     = [b[3] for b in batch]
    return imgs, dicom_ids, sids, stids


def extract_split(model, root, split, batch_size, num_workers, rank, world_size):
    ds = MIMICNpyDataset(root, split, rank, world_size)
    print(f"  [GPU {rank}] {split}: {len(ds)} images to extract")

    if len(ds) == 0:
        return

    loader = DataLoader(
        ds,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    with torch.no_grad():
        for imgs, dicom_ids, sids, stids in tqdm(loader, desc=f"  GPU{rank}/{split}", unit="batch"):
            out  = model(pixel_values=imgs.cuda())
            cls  = out.last_hidden_state[:, 0].cpu().float().numpy()  # (B, 768)

            for i, (did, sid, stid) in enumerate(zip(dicom_ids, sids, stids)):
                out_path = os.path.join(
                    root, "files", f"p{sid[:2]}", f"p{sid}", f"s{stid}",
                    f"raddino_{did}.npy"
                )
                np.save(out_path, cls[i])


def worker(rank: int, world_size: int, args):
    torch.cuda.set_device(rank)
    print(f"[GPU {rank}] Loading RAD-DINO...")
    model = AutoModel.from_pretrained("microsoft/rad-dino").cuda().eval()

    workers_per_gpu = max(1, args.num_workers // world_size)
    for split in args.splits:
        extract_split(model, args.data_dir, split, args.batch_size,
                      workers_per_gpu, rank, world_size)

    print(f"[GPU {rank}] Done.")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir",    required=True, help="MIMIC-CXR-JPG root")
    p.add_argument("--splits",      nargs="+", default=["train", "validate", "test"])
    p.add_argument("--batch_size",  type=int, default=128)
    p.add_argument("--num_workers", type=int, default=8)
    args = p.parse_args()

    world_size = torch.cuda.device_count()
    print(f"Found {world_size} GPU(s)")

    if world_size > 1:
        mp.spawn(worker, args=(world_size, args), nprocs=world_size, join=True)
    else:
        worker(0, 1, args)


if __name__ == "__main__":
    main()
