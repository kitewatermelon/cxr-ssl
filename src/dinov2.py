"""
DINOv2 Pre-training with PyTorch Lightning + MIMIC-CXR
- global views: 2 × [B, 3, 224, 224]
- local views:  8 × [B, 3, 96, 96]
"""

import torch
import lightning as pl
from torch.utils.data import DataLoader
from functools import partial
from lightning.pytorch.loggers import WandbLogger

import stable_pretraining as spt
from stable_pretraining.methods import DINOv2
from stable_pretraining.data.datasets import FromTorchDataset
from stable_pretraining.callbacks import TeacherStudentCallback

from mimic_cxr import MIMICCXRDataset, get_DINOv2_aug
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
        cpu_transform, _, _, _ = get_DINOv2_aug(torch.device("cpu"))

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


class DINOv2Module(spt.Module):
    def __init__(self, arch: str = "vit_small_patch16_224", lr: float = 1e-3):
        super().__init__(hparams={"arch": arch, "lr": lr})
        self.model = DINOv2(arch)
        self.global_aug = None
        self.local_aug = None
        self.val_aug = None
        self.optim = {
            "optimizer": partial(torch.optim.AdamW, lr=lr),
            "scheduler": "CosineAnnealingLR",
            "interval": "step",
        }

    @property
    def embed_dim(self):
        return self.model.backbone.student.embed_dim

    def on_fit_start(self):
        super().on_fit_start()
        _, self.global_aug, self.local_aug, self.val_aug = get_DINOv2_aug(self.device)

    def forward(self, batch, stage="fit"):
        img = batch["image"]

        if stage != "fit":
            img = self.val_aug({"image": img})["image"]
            out = self.model(images=img)
            return {
                "loss": out.loss,
                "cls_token": out.embedding,
                "label": batch["labels"].long(),
            }

        global_views = self.global_aug({"image": img})["views"]  # list of 2 dicts
        local_views = self.local_aug({"image": img})["views"]    # list of 8 dicts

        global_images = [v["image"] for v in global_views]
        local_images = [v["image"] for v in local_views]

        out = self.model(global_views=global_images, local_views=local_images)

        # out.embedding = t_cls from cat(global_views) → [2B, D]; take first B (view1)
        B = img.shape[0]
        cls_token = out.embedding[:B]

        self.log("train/loss", out.loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log("train/loss_cls", out.loss_cls, on_step=True, on_epoch=False)
        self.log("train/loss_patch", out.loss_patch, on_step=True, on_epoch=False)

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
        batch_size=64,
        num_workers=22,
        frontal_only=False,
    )

    model = DINOv2Module(arch="vit_small_patch16_224", lr=1.5e-4)

    logger = WandbLogger(
        entity="RCJiT",
        project="cxr-ssl",
        name="DINOv2-vit_small-mimic",
    )

    trainer = pl.Trainer(
        max_epochs=300,
        accelerator="gpu",
        devices="auto",
        strategy="ddp",
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
