"""
Linear probing evaluation for SSL models (RadJEPA / RAD-DINO) on MIMIC-CXR.

Frozen backbone → online feature extraction → train linear classifier.

Usage:
    python src/eval/linear_probing.py \\
        --data_dir   /mnt/d/data/physionet.org/files/mimic-cxr-jpg/2.1.0 \\
        --model_name "AIDElab-IITBombay/RadJEPA" \\
        --output_dir output/linear_probing_radjepa \\
        --wandb_run_name radjepa-linear-probe
"""

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd
from transformers import AutoModel
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.strategies import DDPStrategy
from torchmetrics.classification import MultilabelAUROC


MIMICCXR_MEAN = (0.4720, 0.4720, 0.4720)
MIMICCXR_STD  = (0.3030, 0.3030, 0.3030)

RADDINO_MEAN  = (0.5307, 0.5307, 0.5307)
RADDINO_STD   = (0.2583, 0.2583, 0.2583)

IN1K_MEAN     = (0.485, 0.456, 0.406)
IN1K_STD      = (0.229, 0.224, 0.225)

NORM_PRESETS  = {
    "mimiccxr": (MIMICCXR_MEAN, MIMICCXR_STD),
    "raddino":  (RADDINO_MEAN,  RADDINO_STD),
    "in1k":     (IN1K_MEAN,     IN1K_STD),
}

CHEXPERT_LABELS = [
    "Atelectasis", "Cardiomegaly", "Consolidation", "Edema",
    "Enlarged Cardiomediastinum", "Fracture", "Lung Lesion",
    "Lung Opacity", "No Finding", "Pleural Effusion",
    "Pleural Other", "Pneumonia", "Pneumothorax", "Support Devices",
]


def make_transforms(norm: str):
    mean, std = NORM_PRESETS[norm]
    train = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomResizedCrop(224, scale=(0.6, 1.0)),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.3, contrast=0.3),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    val = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    return train, val


TRAIN_TRANSFORM, VAL_TRANSFORM = make_transforms("mimiccxr")


# ---------------------------------------------------------------------------
# Path helpers  (same convention as extract_radjepa_features.py)
# ---------------------------------------------------------------------------

def _study_dir(root: str, row) -> str:
    sid  = str(int(row["subject_id"]))
    stid = str(int(row["study_id"]))
    return os.path.join(root, "files", f"p{sid[:2]}", f"p{sid}", f"s{stid}")

def _npy_path(root: str, row) -> str:
    return os.path.join(_study_dir(root, row), f"{row['dicom_id']}.npy")


# ---------------------------------------------------------------------------
# Label helper
# ---------------------------------------------------------------------------

def _load_label_df(root: str, split: str, uncertainty: str) -> pd.DataFrame:
    """
    Returns a DataFrame with columns [subject_id, study_id, dicom_id, *CHEXPERT_LABELS].
    Frontal-only (PA/AP). Uncertainty: 'zero' → -1 to 0 | 'one' → -1 to 1 | 'ignore' → keep -1.
    """
    split_df = pd.read_csv(os.path.join(root, "mimic-cxr-2.0.0-split.csv.gz"))
    split_df = split_df[split_df["split"] == split].reset_index(drop=True)

    label_df = pd.read_csv(os.path.join(root, "mimic-cxr-2.0.0-chexpert.csv.gz"))
    df = split_df.merge(label_df, on=["subject_id", "study_id"], how="left").reset_index(drop=True)

    for col in CHEXPERT_LABELS:
        if col not in df.columns:
            df[col] = 0.0
        else:
            if uncertainty == "zero":
                df[col] = df[col].replace(-1, 0)
            elif uncertainty == "one":
                df[col] = df[col].replace(-1, 1)
            df[col] = df[col].fillna(0)

    return df


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class MIMICImageDataset(Dataset):
    """
    Loads preprocessed 256×256 .npy grayscale images.
    Image files: files/p{group}/p{sid}/s{stid}/{dicom_id}.npy
    Pass TRAIN_TRANSFORM for augmented training, VAL_TRANSFORM for eval.
    """

    def __init__(self, root: str, split: str, transform, uncertainty: str = "zero"):
        self.root      = root
        self.transform = transform

        df = _load_label_df(root, split, uncertainty)

        exists    = [os.path.exists(_npy_path(root, row)) for _, row in df.iterrows()]
        n_skipped = sum(1 for e in exists if not e)
        if n_skipped:
            print(f"  [{split}] skipping {n_skipped} rows with missing .npy files")
        self.df = df[exists].reset_index(drop=True)
        print(f"  [{split}] {len(self.df)} samples")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        arr = np.load(_npy_path(self.root, row))             # (256, 256) uint8 grayscale
        img = Image.fromarray(arr, mode="L").convert("RGB")
        img = self.transform(img)
        labels = torch.tensor(row[CHEXPERT_LABELS].values.astype("float32"))
        return img, labels


# ---------------------------------------------------------------------------
# LightningModule
# ---------------------------------------------------------------------------

def _load_backbone(args):
    """HuggingFace 모델 또는 MAE Lightning 체크포인트에서 backbone 로드."""
    if args.model_name.startswith("/") or args.model_name.startswith("."):
        # MAE Lightning 체크포인트 경로
        import timm
        ckpt = torch.load(args.model_name, map_location="cpu", weights_only=False)
        arch = ckpt.get("hyper_parameters", {}).get("arch", "vit_small_patch16_224")
        encoder = timm.create_model(arch, pretrained=False, num_classes=0, img_size=224)
        prefix = "mae.encoder.vit."
        enc_state = {k[len(prefix):]: v for k, v in ckpt["state_dict"].items()
                     if k.startswith(prefix)}
        missing, unexpected = encoder.load_state_dict(enc_state, strict=False)
        if missing:
            print(f"[MAE] missing keys: {len(missing)}")

        # pixel_values 인터페이스로 래핑
        class _MAEWrapper(nn.Module):
            def __init__(self, enc):
                super().__init__()
                self.enc = enc
            def forward(self, pixel_values):
                return self.enc.forward_features(pixel_values)[:, 0]  # CLS token

        return _MAEWrapper(encoder)
    else:
        return AutoModel.from_pretrained(args.model_name, trust_remote_code=True)


class LitLinearProbe(L.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.save_hyperparameters(vars(args))
        self.args = args

        self.backbone = _load_backbone(args)
        self.backbone.eval()
        for p in self.backbone.parameters():
            p.requires_grad = False

        self.classifier = nn.Linear(args.feat_dim, args.n_classes)
        self.criterion   = nn.BCEWithLogitsLoss()

        n = args.n_classes
        self.val_auroc      = MultilabelAUROC(num_labels=n, average="macro")
        self.val_auroc_per  = MultilabelAUROC(num_labels=n, average="none")
        self.test_auroc     = MultilabelAUROC(num_labels=n, average="macro")
        self.test_auroc_per = MultilabelAUROC(num_labels=n, average="none")

    def train(self, mode: bool = True):
        super().train(mode)
        self.backbone.eval()  # keep backbone in eval mode to disable dropout
        return self

    @torch.no_grad()
    def _extract(self, x: torch.Tensor) -> torch.Tensor:
        out = self.backbone(pixel_values=x)
        if hasattr(out, "last_hidden_state"):
            return out.last_hidden_state[:, 0]   # CLS token (RAD-DINO)
        return out                                 # mean-pooled (RadJEPA)

    def training_step(self, batch, batch_idx):
        x, y   = batch
        logits = self.classifier(self._extract(x))
        loss   = self.criterion(logits, y)
        self.log("train/loss", loss, on_step=True, on_epoch=False, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y   = batch
        logits = self.classifier(self._extract(x))
        loss   = self.criterion(logits, y)
        probs  = torch.sigmoid(logits)
        self.log("val/loss", loss, on_step=False, on_epoch=True, sync_dist=True)
        self.val_auroc.update(probs, y.long())
        self.val_auroc_per.update(probs, y.long())

    def on_validation_epoch_end(self):
        macro = self.val_auroc.compute()
        per   = self.val_auroc_per.compute()
        self.log("val/auroc_macro", macro, prog_bar=True)
        for label, auc in zip(CHEXPERT_LABELS, per):
            self.log(f"val/auroc_{label}", auc)
        self.val_auroc.reset()
        self.val_auroc_per.reset()

    def test_step(self, batch, batch_idx):
        x, y  = batch
        probs = torch.sigmoid(self.classifier(self._extract(x)))
        self.test_auroc.update(probs, y.long())
        self.test_auroc_per.update(probs, y.long())

    def on_test_epoch_end(self):
        macro = self.test_auroc.compute()
        per   = self.test_auroc_per.compute()
        print(f"\n{'='*50}")
        print(f"Test macro AUC : {macro:.4f}")
        for label, auc in zip(CHEXPERT_LABELS, per):
            print(f"  {label:35s}: {auc:.4f}")
        self.log("test/auroc_macro", macro)
        for label, auc in zip(CHEXPERT_LABELS, per):
            self.log(f"test/auroc_{label}", auc)
        self.test_auroc.reset()
        self.test_auroc_per.reset()

    def configure_optimizers(self):
        opt = torch.optim.AdamW(
            self.classifier.parameters(),
            lr=self.args.lr,
            weight_decay=self.args.weight_decay,
        )
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.args.max_epochs)
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sched, "interval": "epoch"}}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser("Linear probing on MIMIC-CXR")

    # ── data ────────────────────────────────────────────────────────────────
    p.add_argument("--data_dir",    default="/mnt/d/data/physionet.org/files/mimic-cxr-jpg/2.1.0")
    p.add_argument("--model_name",  default="AIDElab-IITBombay/RadJEPA",
                   help="HuggingFace model ID for feature extraction")
    p.add_argument("--feat_dim",    type=int, default=768)
    p.add_argument("--n_classes",   type=int, default=14)
    p.add_argument("--uncertainty", default="zero", choices=["zero", "one", "ignore"])
    p.add_argument("--norm", default="mimiccxr", choices=list(NORM_PRESETS.keys()),
                   help="Normalization preset: mimiccxr | raddino | in1k")
    p.add_argument("--num_workers", type=int, default=8)

    # ── training ─────────────────────────────────────────────────────────────
    p.add_argument("--batch_size",   type=int,   default=128)
    p.add_argument("--max_epochs",   type=int,   default=10)
    p.add_argument("--lr",           type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--num_gpus",     type=int,   default=1,
                   help="Number of GPUs (-1 = all available)")

    # ── checkpointing / logging ──────────────────────────────────────────────
    p.add_argument("--output_dir",     default="output/linear_probing")
    p.add_argument("--resume_from",    default=None)
    p.add_argument("--wandb_project",  default="cxr-linear-probing")
    p.add_argument("--wandb_run_name", default=None)
    p.add_argument("--wandb_run_id",   default=None)
    p.add_argument("--log_every",      type=int, default=50)
    p.add_argument("--seed",           type=int, default=42)
    args = p.parse_args()

    import random
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False
    L.seed_everything(args.seed, workers=True)
    os.makedirs(args.output_dir, exist_ok=True)

    # ── datasets ─────────────────────────────────────────────────────────────
    train_tf, val_tf = make_transforms(args.norm)
    print(f"Loading datasets from {args.data_dir} (norm={args.norm}) ...")
    train_ds = MIMICImageDataset(args.data_dir, "train",    train_tf, args.uncertainty)
    val_ds   = MIMICImageDataset(args.data_dir, "validate", val_tf,   args.uncertainty)
    test_ds  = MIMICImageDataset(args.data_dir, "test",     val_tf,   args.uncertainty)

    loader_kw = dict(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=(args.num_workers > 0),
        multiprocessing_context="fork",
    )
    train_loader = DataLoader(train_ds, shuffle=True,  drop_last=True,  **loader_kw)
    val_loader   = DataLoader(val_ds,   shuffle=False, drop_last=False, **loader_kw)
    test_loader  = DataLoader(test_ds,  shuffle=False, drop_last=False, **loader_kw)

    # ── model ─────────────────────────────────────────────────────────────────
    model = LitLinearProbe(args)

    # ── callbacks ─────────────────────────────────────────────────────────────
    ckpt_cb = ModelCheckpoint(
        dirpath=os.path.join(args.output_dir, "checkpoints"),
        monitor="val/auroc_macro",
        mode="max",
        save_top_k=1,
        filename="best",
        save_last=True,
    )
    lr_cb = LearningRateMonitor(logging_interval="epoch")

    # ── logger ────────────────────────────────────────────────────────────────
    logger = WandbLogger(
        project=args.wandb_project,
        name=args.wandb_run_name,
        id=args.wandb_run_id or None,
        resume="allow" if args.wandb_run_id else None,
        save_dir=args.output_dir,
        log_model=False,
    )

    # ── trainer ───────────────────────────────────────────────────────────────
    trainer = L.Trainer(
        max_epochs=args.max_epochs,
        accelerator="gpu",
        devices=args.num_gpus,
        precision="bf16-mixed",
        strategy=DDPStrategy(find_unused_parameters=False) if args.num_gpus != 1 else "auto",
        callbacks=[ckpt_cb, lr_cb],
        logger=logger,
        log_every_n_steps=args.log_every,
        enable_progress_bar=True,
    )

    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader,
                ckpt_path=args.resume_from)
    trainer.test(model, dataloaders=test_loader, ckpt_path="best")


if __name__ == "__main__":
    main()
