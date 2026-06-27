import os
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
# import torchvision.transforms as transforms
from torchvision import transforms as tv_transforms
from stable_pretraining.data import transforms, gpu_transforms as gt
from stable_pretraining.data.datasets import FromTorchDataset


class MIMICCXRDataset(Dataset):
    """
    MIMIC-CXR-JPG Dataset
    
    Directory structure:
        root/
            files/
                p{group}/
                    p{subject_id}/
                        s{study_id}/
                            *.jpg
            mimic-cxr-2.0.0-chexpert.csv.gz
            mimic-cxr-2.0.0-split.csv.gz
            mimic-cxr-2.0.0-metadata.csv.gz
    """

    # CheXpert 14개 레이블
    CHEXPERT_LABELS = [
        "Atelectasis", "Cardiomegaly", "Consolidation", "Edema",
        "Enlarged Cardiomediastinum", "Fracture", "Lung Lesion",
        "Lung Opacity", "No Finding", "Pleural Effusion",
        "Pleural Other", "Pneumonia", "Pneumothorax", "Support Devices"
    ]

    def __init__(
        self,
        root: str,
        split: str = "train",          # "train" | "validate" | "test"
        label_csv: str = "chexpert",   # "chexpert" | "negbio"
        transform=None,
        frontal_only: bool = True,     # PA/AP view만 사용
        uncertainty: str = "zero",     # "zero" | "one" | "ignore"
    ):
        self.root = root
        self.transform = transform
        self.uncertainty = uncertainty

        # ── 1. split CSV 로드 ──────────────────────────────────────
        split_df = pd.read_csv(
            os.path.join(root, "mimic-cxr-2.0.0-split.csv.gz")
        )
        split_df = split_df[split_df["split"] == split].reset_index(drop=True)

        # ── 2. metadata CSV 로드 (ViewPosition 필터링용) ───────────
        if frontal_only:
            meta_df = pd.read_csv(
                os.path.join(root, "mimic-cxr-2.0.0-metadata.csv.gz")
            )
            frontal_views = meta_df[
                meta_df["ViewPosition"].isin(["PA", "AP"])
            ][["dicom_id"]]
            split_df = split_df.merge(frontal_views, on="dicom_id")

        # ── 3. label CSV 로드 ──────────────────────────────────────
        label_file = f"mimic-cxr-2.0.0-{label_csv}.csv.gz"
        label_df = pd.read_csv(os.path.join(root, label_file))

        # study_id 기준으로 merge
        self.df = split_df.merge(label_df, on=["subject_id", "study_id"], how="left")
        self.df = self.df.reset_index(drop=True)

        # ── 4. uncertainty 처리 (-1 값) ───────────────────────────
        for col in self.CHEXPERT_LABELS:
            if col not in self.df.columns:
                self.df[col] = 0.0
                continue
            if uncertainty == "zero":
                self.df[col] = self.df[col].replace(-1, 0)
            elif uncertainty == "one":
                self.df[col] = self.df[col].replace(-1, 1)
            # "ignore": -1 그대로 유지 (loss masking 시 사용)
            self.df[col] = self.df[col].fillna(0)

    def __len__(self):
        return len(self.df)

    def _get_image_path(self, row) -> str:
        subject_id = str(int(row["subject_id"]))
        study_id   = str(int(row["study_id"]))
        dicom_id   = str(row["dicom_id"])

        # p10000032 → p10/ 그룹 디렉토리
        group = "p" + subject_id[:2]

        path = os.path.join(
            self.root, "files",
            group,
            f"p{subject_id}",
            f"s{study_id}",
            f"{dicom_id}.jpg"
        )
        return path

    # def __getitem__(self, idx):
    #     row = self.df.iloc[idx]
    #     img_path = self._get_image_path(row)

    #     image = Image.open(img_path).convert("RGB")
    #     if self.transform:
    #         image = self.transform(image)

    #     labels = row[self.CHEXPERT_LABELS].values.astype("float32")

        # return {
        #     "image":      image,
        #     "labels":     labels,              # (14,) float32
        #     "subject_id": int(row["subject_id"]),
        #     "study_id":   int(row["study_id"]),
        #     "dicom_id":   row["dicom_id"],
        #     "path":       img_path,
        # }

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        npy_path = self._get_image_path(row)[:-4] + ".npy"

        try:
            arr = np.load(npy_path)                      # (256, 256) uint8 grayscale
            image = Image.fromarray(arr, mode="L")
        except Exception:
            return self.__getitem__((idx + 1) % len(self))

        if self.transform:
            image = self.transform(image)

        labels = row[self.CHEXPERT_LABELS].values.astype("float32")

        return image, labels

def get_LeJEPA_aug(device):
    cpu_transform = transforms.Compose(
        transforms.Resize((256, 256)),
        transforms.ToImage(rgb=True, scale=True),
    )

    # global views: 2 × 224×224 (large crop)
    global_aug = gt.StackedMultiView(
        gt.GPUCompose([
            gt.GPURandomResizedCrop(size=224, scale=(0.4, 1.0)),
            gt.GPURandomHorizontalFlip(p=0.5),
            gt.GPUColorJitter(0.4, 0.4, 0.2, 0.1, p=0.8),
            gt.GPURandomGrayscale(p=0.2),
            gt.GPUGaussianBlur(kernel_size=23, p=0.5),
            gt.GPUNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]),
        n_views=2,
    ).to(device)

    # local views: 6 × 96×96 (small crop)
    local_aug = gt.StackedMultiView(
        gt.GPUCompose([
            gt.GPURandomResizedCrop(size=96, scale=(0.05, 0.4)),
            gt.GPURandomHorizontalFlip(p=0.5),
            gt.GPUColorJitter(0.4, 0.4, 0.2, 0.1, p=0.8),
            gt.GPURandomGrayscale(p=0.2),
            gt.GPUGaussianBlur(kernel_size=11, p=0.5),
            gt.GPUNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]),
        n_views=6,
    ).to(device)

    return cpu_transform, global_aug, local_aug

def get_DINOv2_aug(device):
    cpu_transform = transforms.Compose(
        transforms.Resize((256, 256)),
        transforms.ToImage(rgb=True, scale=True),
    )

    # global views: 2 × 224×224
    global_aug = gt.StackedMultiView(
        gt.GPUCompose([
            gt.GPURandomResizedCrop(size=224, scale=(0.32, 1.0)),
            gt.GPURandomHorizontalFlip(p=0.5),
            gt.GPUColorJitter(0.4, 0.4, 0.2, 0.1, p=0.8),
            gt.GPURandomGrayscale(p=0.2),
            gt.GPUGaussianBlur(kernel_size=23, p=1.0),
            gt.GPUNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]),
        n_views=2,
    ).to(device)

    # local views: 8 × 96×96
    local_aug = gt.StackedMultiView(
        gt.GPUCompose([
            gt.GPURandomResizedCrop(size=96, scale=(0.05, 0.32)),
            gt.GPURandomHorizontalFlip(p=0.5),
            gt.GPUColorJitter(0.4, 0.4, 0.2, 0.1, p=0.8),
            gt.GPURandomGrayscale(p=0.2),
            gt.GPUGaussianBlur(kernel_size=11, p=0.5),
            gt.GPUNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]),
        n_views=8,
    ).to(device)

    val_aug = gt.GPUCompose([
        gt.GPURandomResizedCrop(size=224, scale=(0.32, 1.0)),
        gt.GPUNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]).to(device)

    return cpu_transform, global_aug, local_aug, val_aug


def get_MAE_aug(device):
    cpu_transform = transforms.Compose(
        transforms.Resize((256, 256)),
        transforms.ToImage(rgb=True, scale=True),
    )

    gpu_aug = gt.GPUCompose([
        gt.GPURandomResizedCrop(size=224, scale=(0.2, 1.0)),
        gt.GPURandomHorizontalFlip(p=0.5),
        gt.GPUNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]).to(device)

    val_aug = gt.GPUCompose([
        gt.GPURandomResizedCrop(size=224, scale=(0.2, 1.0)),
        gt.GPUNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]).to(device)

    return cpu_transform, gpu_aug, val_aug


if __name__ == "__main__":
    from torch.utils.data import DataLoader
    import torchvision.transforms as transforms

    meta_df = pd.read_csv("/mnt/nvme1/mimic-cxr-jpg/mimic-cxr-2.0.0-metadata.csv.gz")
    print(meta_df["ViewPosition"].value_counts(dropna=False))

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])

    train_dataset = MIMICCXRDataset(
        root="/mnt/nvme1/mimic-cxr-jpg",
        split="train",
        label_csv="chexpert",
        transform=transform,
        frontal_only=False,
        uncertainty="zero",
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=64,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
    )

    print(f"Train samples: {len(train_dataset)}")

    batch = next(iter(train_loader))
    print(batch["image"].shape)
    print(batch["labels"].shape)