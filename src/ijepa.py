"""
I-JEPA Pre-training with PyTorch Lightning + MIMIC-CXR
"""

import torch
import lightning as pl
from torch.utils.data import DataLoader
from functools import partial
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.strategies import DDPStrategy

import stable_pretraining as spt
from stable_pretraining.methods import IJEPA
from stable_pretraining.data.datasets import FromTorchDataset
from stable_pretraining.callbacks import TeacherStudentCallback

from mimic_cxr import MIMICCXRDataset, get_MAE_aug
from utils import get_common_callbacks


class MIMICCXRDataModule(pl.LightningDataModule):
    def __init__(
        self,
        root: str = "/mnt/nvme1/mimic-cxr-jpg",
        batch_size: int = 256,
        num_workers: int = 12,
        frontal_only: bool = False,
        uncertainty: str = "zero",
    ):
        super().__init__()
        self.save_hyperparameters()

    def setup(self, stage=None):
        cpu_transform, _, _ = get_MAE_aug(torch.device("cpu"))

        def make_ds(split):
            return FromTorchDataset(
                dataset=MIMICCXRDataset(
                    root=self.hparams.root,
                    split=split,
                    label_csv="chexpert",
                    transform=None,
                    frontal_only=self.hparams.frontal_only,
                    uncertainty=self.hparams.uncertainty,
                ),
                names=["image", "labels"],
                transform=cpu_transform,
                gpu_transform=None,
            )

        self.train_ds = make_ds("train")
        self.val_ds = make_ds("validate")

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            pin_memory=True,
            persistent_workers=True,
            multiprocessing_context="fork",
            drop_last=True,
            prefetch_factor=4,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            pin_memory=True,
            persistent_workers=True,
            multiprocessing_context="fork",
            prefetch_factor=4,
        )


class IJEPAModule(spt.Module):
    def __init__(self, arch: str = "vit_small_patch16_224", lr: float = 1e-3):
        super().__init__(hparams={"arch": arch, "lr": lr})
        self.model = IJEPA(arch)
        self.gpu_aug = None
        self.val_aug = None
        self.optim = {
            "optimizer": partial(torch.optim.AdamW, lr=lr),
            "scheduler": "CosineAnnealingLR",
            "interval": "step",
        }

    @property
    def embed_dim(self):
        return self.model.embed_dim

    def on_fit_start(self):
        super().on_fit_start()
        _, self.gpu_aug, self.val_aug = get_MAE_aug(self.device)

    def forward(self, batch, stage="fit"):
        img = batch["image"]
        aug = self.gpu_aug if stage == "fit" else self.val_aug
        img = aug({"image": img})["image"]

        out = self.model(img)
        # embedding: [B, N, D] patch tokens → mean pool → [B, D]
        cls_token = out.embedding.mean(dim=1)

        if stage != "fit":
            return {
                "loss": torch.tensor(0.0, device=img.device),
                "cls_token": cls_token,
                "label": batch["labels"].long(),
            }

        self.log("train/loss", out.loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log("train/num_context", float(out.num_context), on_step=True, on_epoch=False)
        self.log("train/num_targets", float(out.num_targets), on_step=True, on_epoch=False)

        return {
            "loss": out.loss,
            "cls_token": cls_token,
            "label": batch["labels"].long(),
        }


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────

if __name__ == "__main__":
    pl.seed_everything(42)
    datamodule = MIMICCXRDataModule(
        root="/mnt/nvme1/mimic-cxr-jpg",
        batch_size=256,
        num_workers=22,
        frontal_only=False,
    )

    model = IJEPAModule(arch="vit_small_patch16_224", lr=1.5e-4)

    logger = WandbLogger(
        entity="RCJiT",
        project="cxr-ssl",
        name="IJEPA-vit_small-mimic",
    )

    trainer = pl.Trainer(
        max_epochs=800,
        accelerator="gpu",
        devices="auto",
        strategy=DDPStrategy(find_unused_parameters=True),
        precision="16-mixed",
        log_every_n_steps=50,
        logger=logger,
        callbacks=[
            TeacherStudentCallback(),
            *get_common_callbacks(
                model,
                num_classes=14,
                task="multilabel",
                queue_length=4096,
            ),
        ],
    )

    trainer.fit(model, datamodule)
