"""
MAE Pre-training with PyTorch Lightning + MIMIC-CXR
"""

import torch
import lightning as pl
from torch.utils.data import DataLoader
from functools import partial
from lightning.pytorch.loggers import WandbLogger

import stable_pretraining as spt
from stable_pretraining.methods import MAE
from stable_pretraining.data.datasets import FromTorchDataset

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
            shuffle=False,  # DDP: Lightning이 DistributedSampler로 shuffle 처리
            num_workers=self.hparams.num_workers,
            pin_memory=True,
            persistent_workers=True,
            multiprocessing_context="fork",
            drop_last=True,  # DDP: 마지막 불완전 배치 제거 (GPU간 크기 불일치 방지)
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


class MAEModule(spt.Module):
    def __init__(self, arch: str = "vit_small_patch16_224", lr: float = 1e-3):
        super().__init__(hparams={"arch": arch, "lr": lr})
        self.mae = MAE(arch)
        self.gpu_aug = None
        self.val_aug = None
        self.optim = {
            "optimizer": partial(torch.optim.AdamW, lr=lr),
            "scheduler": "CosineAnnealingLR",
            "interval": "step",
        }

    @property
    def embed_dim(self):
        return self.mae.encoder.embed_dim

    def on_fit_start(self):
        super().on_fit_start()
        _, self.gpu_aug, self.val_aug = get_MAE_aug(self.device)

    def on_train_start(self):
        super().on_train_start()

    def after_manual_backward(self):
        scaler = self.trainer.precision_plugin.scaler
        scale = scaler.get_scale() if scaler is not None else 1.0
        scaled_norm = torch.nn.utils.clip_grad_norm_(
            self.mae.parameters(), max_norm=float("inf")
        )
        self.log("train/grad_norm", scaled_norm / scale, on_step=True, on_epoch=False)

    def forward(self, batch, stage="fit"):
        img = batch["image"]
        aug = self.gpu_aug if stage == "fit" else self.val_aug
        img = aug({"image": img})["image"]

        enc_out = self.mae.encoder(img)
        cls_token = enc_out.encoded[:, 0]  # [B, D]

        if stage != "fit":
            return {"loss": torch.tensor(0.0, device=img.device), "cls_token": cls_token, "label": batch["labels"].long()}

        encoded_patches = enc_out.encoded[:, self.mae.encoder.num_prefix_tokens:]
        predictions = self.mae.decoder(
            encoded_patches,
            enc_out.mask,
            ids_keep=enc_out.ids_keep,
            output_masked_only=False,
        )
        loss = self.mae.loss_fn(predictions, img.to(predictions.dtype), enc_out.mask)

        param_norm = torch.stack([p.norm() for p in self.mae.parameters()]).norm()

        self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log("train/param_norm", param_norm, on_step=True, on_epoch=False)
        return {"loss": loss, "cls_token": cls_token, "label": batch["labels"].long()}


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

    model = MAEModule(arch="vit_small_patch16_224", lr=1.5e-4)

    logger = WandbLogger(
        entity="RCJiT",
        project="cxr-ssl",
        name="MAE-vit_small-mimic",
    )

    trainer = pl.Trainer(
        max_epochs=800,
        accelerator="gpu",
        devices="auto",
        strategy="ddp",
        precision="16-mixed",
        log_every_n_steps=50,
        logger=logger,
        callbacks=get_common_callbacks(
            model,
            num_classes=14,
            task="multilabel",
            queue_length=4096,
        ),
    )

    trainer.fit(model, datamodule)
