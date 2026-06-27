"""
RCJiT-S/16 training on MIMIC-CXR, conditioned on MAE-pretrained ViT-S/16 CLS token.

Pipeline:
  1. Frozen encoder: MAE-pretrained ViT-S/16 (from cxr-ssl Lightning checkpoint)
                     → CLS token (384-dim), ctx_mode="cls"
  2. Denoiser:       JiT-S/16 (hidden=384, patch=16) flow-matching on 224×224 CXR

Usage:
    python cxr-ssl/train_rcjit_mae.py \
        --encoder_ckpt cxr-ssl/ssnlqx8u/checkpoints/epoch=799-step=576000.ckpt \
        --data_dir     /mnt/nvme1/mimic-cxr-jpg \
        --output_dir   output/rcjit_s16_mae_cxr
"""

import os
import sys
import math
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'rcdm', 'RCJiT', 'JiT'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'rcdm', 'RCJiT', 'src'))

# CheXFound path is injected at runtime via --chexfound_dir (parsed early below)

import torch
import torchvision.utils as vutils
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.strategies import DDPStrategy

from stable_pretraining.data import transforms
from stable_pretraining.data import gpu_transforms as gt
from stable_pretraining.data.datasets import FromTorchDataset

from denoiser import DINOv2Denoiser
from mimic_cxr import MIMICCXRDataset


# ---------------------------------------------------------------------------
# Augmentation — images in [-1, 1] for flow-matching
# ---------------------------------------------------------------------------

def get_rcjit_aug(device):
    """CPU resize → GPU crop/flip/normalize to [-1, 1]."""
    cpu_transform = transforms.Compose(
        transforms.Resize((256, 256)),
        transforms.ToImage(rgb=True, scale=True),   # PIL L → [0,1] RGB tensor
    )
    gpu_aug = gt.GPUCompose([
        gt.GPURandomResizedCrop(size=224, scale=(0.2, 1.0)),
        gt.GPURandomHorizontalFlip(p=0.5),
        gt.GPUNormalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),   # → [-1, 1]
    ]).to(device)
    return cpu_transform, gpu_aug


# ---------------------------------------------------------------------------
# DataModule
# ---------------------------------------------------------------------------

class MIMICRCJiTDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str, batch_size: int, num_workers: int):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self._cpu_transform = None

    def setup(self, stage=None):
        cpu_transform, _ = get_rcjit_aug(torch.device("cpu"))
        self._cpu_transform = cpu_transform

        def make_ds(split):
            return FromTorchDataset(
                dataset=MIMICCXRDataset(
                    root=self.data_dir,
                    split=split,
                    label_csv="chexpert",
                    transform=None,
                    frontal_only=False,
                    uncertainty="zero",
                ),
                names=["image", "labels"],
                transform=cpu_transform,
                gpu_transform=None,
            )

        self.train_ds = make_ds("train")

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
            persistent_workers=(self.num_workers > 0),
            multiprocessing_context="fork",
            prefetch_factor=4,
        )


# ---------------------------------------------------------------------------
# LR schedule
# ---------------------------------------------------------------------------

def _warmup_cosine_lambda(warmup_steps: int, max_steps: int, min_lr_ratio: float = 0.01):
    def lr_lambda(step: int):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine
    return lr_lambda


# ---------------------------------------------------------------------------
# Sample callback
# ---------------------------------------------------------------------------

class SampleCallback(pl.Callback):
    def __init__(self, sample_imgs: torch.Tensor, every_n_steps: int, ode_steps: int, cfg: float):
        super().__init__()
        self.sample_imgs = sample_imgs   # (N, 3, 224, 224) in [-1, 1]
        self.every_n_steps = every_n_steps
        self.ode_steps = ode_steps
        self.cfg = cfg

    @torch.no_grad()
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if trainer.global_step == 0 or trainer.global_step % self.every_n_steps != 0:
            return
        if not trainer.is_global_zero:
            return

        denoiser = pl_module.denoiser
        denoiser.eval()
        cond = self.sample_imgs.to(pl_module.device)

        orig_steps, orig_cfg = denoiser.steps, denoiser.cfg_scale
        denoiser.steps, denoiser.cfg_scale = self.ode_steps, self.cfg

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            gen = denoiser.generate(cond)

        denoiser.steps, denoiser.cfg_scale = orig_steps, orig_cfg
        denoiser.train()

        cond_vis = (cond.clamp(-1, 1) + 1) / 2
        gen_vis  = (gen .clamp(-1, 1) + 1) / 2
        grid = vutils.make_grid(
            torch.cat([cond_vis.cpu(), gen_vis.cpu()], dim=0),
            nrow=cond.size(0), padding=2, pad_value=1.0,
        )
        import wandb
        trainer.logger.experiment.log(
            {"samples": wandb.Image(grid, caption=f"step {trainer.global_step} | top: cond, bottom: gen")},
            step=trainer.global_step,
        )


# ---------------------------------------------------------------------------
# LightningModule
# ---------------------------------------------------------------------------

class RCJiTMAEModule(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.save_hyperparameters(vars(args))
        self.args = args
        self._gpu_aug = None

        import argparse
        denoiser_args = argparse.Namespace(
            img_size        = args.img_size,
            model_variant   = args.model_variant,
            ctx_mode        = "cls" if args.encoder_type == "chexfound" else "pool",
            encoder_type    = args.encoder_type,
            encoder_ckpt    = args.encoder_ckpt,
            encoder_config  = getattr(args, 'encoder_config', None),
            attn_dropout    = args.attn_dropout,
            proj_dropout    = args.proj_dropout,
            cond_drop_prob  = args.cond_drop_prob,
            P_mean          = args.P_mean,
            P_std           = args.P_std,
            noise_scale     = args.noise_scale,
            t_eps           = args.t_eps,
            ema_decay1      = args.ema_decay1,
            ema_decay2      = args.ema_decay2,
            sampling_method = args.ode_method,
            num_sampling_steps = args.ode_steps,
            cfg             = args.cfg,
            interval_min    = args.interval_min,
            interval_max    = args.interval_max,
        )
        denoiser = DINOv2Denoiser(denoiser_args)
        if args.compile:
            torch._dynamo.config.cache_size_limit = 128
            torch._dynamo.config.optimize_ddp = False
            denoiser.net = torch.compile(denoiser.net)
        self.denoiser = denoiser

    # ── GPU aug + EMA init (모델이 GPU로 이동한 뒤 실행) ────────────────

    def on_fit_start(self):
        import copy
        _, self._gpu_aug = get_rcjit_aug(self.device)
        if self.denoiser.ema_params1 is None:
            params = self.denoiser._trainable_params()
            self.denoiser.ema_params1 = copy.deepcopy(params)
            self.denoiser.ema_params2 = copy.deepcopy(params)

    def on_save_checkpoint(self, checkpoint):
        names = [n for n, p in self.denoiser.named_parameters() if p.requires_grad]
        if self.denoiser.ema_params1 is not None:
            checkpoint["ema_params1"] = {n: p.cpu() for n, p in zip(names, self.denoiser.ema_params1)}
        if self.denoiser.ema_params2 is not None:
            checkpoint["ema_params2"] = {n: p.cpu() for n, p in zip(names, self.denoiser.ema_params2)}

    def on_load_checkpoint(self, checkpoint):
        names = [n for n, p in self.denoiser.named_parameters() if p.requires_grad]
        if "ema_params1" in checkpoint:
            self.denoiser.ema_params1 = [checkpoint["ema_params1"][n] for n in names]
        if "ema_params2" in checkpoint:
            self.denoiser.ema_params2 = [checkpoint["ema_params2"][n] for n in names]

    # ── training ────────────────────────────────────────────────────────

    def training_step(self, batch, batch_idx):
        x = batch["image"]
        x = self._gpu_aug({"image": x})["image"]   # → [-1, 1]
        loss = self.denoiser(x, cond_img=x)
        self.log("train/loss", loss, on_step=True, on_epoch=False, prog_bar=True, sync_dist=True)
        return loss

    def on_before_optimizer_step(self, optimizer):
        trainable = [p for p in self.denoiser.parameters() if p.requires_grad and p.grad is not None]
        if trainable:
            grad_norm  = torch.stack([p.grad.detach().norm() for p in trainable]).norm()
            param_norm = torch.stack([p.detach().norm()      for p in trainable]).norm()
            self.log("train/grad_norm",  grad_norm,  on_step=True, on_epoch=False, sync_dist=True)
            self.log("train/param_norm", param_norm, on_step=True, on_epoch=False, sync_dist=True)

    def on_train_batch_end(self, outputs, batch, batch_idx):
        self.denoiser.update_ema()

    # ── optimizer & LR ──────────────────────────────────────────────────

    def configure_optimizers(self):
        opt = torch.optim.AdamW(
            [p for p in self.denoiser.parameters() if p.requires_grad],
            lr=self.args.lr,
            betas=(0.9, 0.95),
            weight_decay=self.args.weight_decay,
        )
        sched = torch.optim.lr_scheduler.LambdaLR(
            opt,
            lr_lambda=_warmup_cosine_lambda(self.args.warmup_steps, self.args.max_steps),
        )
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sched, "interval": "step"}}


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser("RCJiT-S/16 MAE-conditioned trainer on MIMIC-CXR")

    # encoder
    p.add_argument("--encoder_ckpt", required=True,
                   help="Path to encoder checkpoint")
    p.add_argument("--encoder_type", default="mae",
                   choices=["mae", "chexfound"],
                   help="Encoder type: mae | chexfound")
    p.add_argument("--encoder_config", default=None,
                   help="Path to encoder config.yaml (required for chexfound)")
    p.add_argument("--chexfound_dir", default=None,
                   help="Path to CheXFound repo root (added to sys.path)")

    # data
    p.add_argument("--data_dir",    default="/mnt/nvme1/mimic-cxr-jpg")
    p.add_argument("--img_size",    type=int, default=224)
    p.add_argument("--num_workers", type=int, default=12)
    p.add_argument("--model_variant", default="S_16", choices=["S_16", "S_8", "B_16", "B_8"])

    # training
    p.add_argument("--batch_size",   type=int,   default=128)
    p.add_argument("--lr",           type=float, default=5e-5)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--max_steps",    type=int,   default=300_000)
    p.add_argument("--warmup_steps", type=int,   default=10_000)
    p.add_argument("--num_gpus",     type=int,   default=-1)

    # flow-matching
    p.add_argument("--cond_drop_prob", type=float, default=0.1)
    p.add_argument("--P_mean",         type=float, default=-0.8)
    p.add_argument("--P_std",          type=float, default=0.8)
    p.add_argument("--noise_scale",    type=float, default=1.0)
    p.add_argument("--t_eps",          type=float, default=0.05)
    p.add_argument("--ema_decay1",     type=float, default=0.9999)
    p.add_argument("--ema_decay2",     type=float, default=0.9996)
    p.add_argument("--attn_dropout",   type=float, default=0.0)
    p.add_argument("--proj_dropout",   type=float, default=0.0)

    # sampling / callback
    p.add_argument("--cfg",          type=float, default=1.5)
    p.add_argument("--ode_steps",    type=int,   default=50)
    p.add_argument("--ode_method",   default="heun", choices=["euler", "heun"])
    p.add_argument("--interval_min", type=float, default=0.0)
    p.add_argument("--interval_max", type=float, default=1.0)
    p.add_argument("--sample_every", type=int,   default=10_000)
    p.add_argument("--num_samples",  type=int,   default=8)

    # checkpointing
    p.add_argument("--output_dir",  default="output/rcjit_s16_mae_cxr")
    p.add_argument("--save_every",  type=int, default=10_000)
    p.add_argument("--resume_from", default=None)

    # logging
    p.add_argument("--wandb_project",  default="rcjit-mae-cxr")
    p.add_argument("--wandb_run_name", default=None)
    p.add_argument("--log_every",      type=int, default=50)

    p.add_argument("--compile", action="store_true")
    p.add_argument("--seed",    type=int, default=42)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    if args.chexfound_dir:
        sys.path.insert(0, args.chexfound_dir)

    pl.seed_everything(args.seed, workers=True)
    os.makedirs(args.output_dir, exist_ok=True)

    # ── data ────────────────────────────────────────────────────────────
    dm = MIMICRCJiTDataModule(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    dm.setup()

    # fixed sample images for SampleCallback (from training set)
    # CPU transform은 256×256 출력 → center crop 224 후 [-1,1] 변환
    import torchvision.transforms.functional as TF
    sample_imgs_raw = [dm.train_ds[i]["image"] for i in range(args.num_samples)]
    sample_imgs = torch.stack(sample_imgs_raw)          # (N, 3, 256, 256) in [0,1]
    sample_imgs = TF.center_crop(sample_imgs, args.img_size)  # → (N, 3, 224, 224)
    sample_imgs = sample_imgs * 2.0 - 1.0              # → [-1, 1]

    # ── model ────────────────────────────────────────────────────────────
    lit = RCJiTMAEModule(args)

    # ── logger & callbacks ───────────────────────────────────────────────
    logger = WandbLogger(
        project=args.wandb_project,
        name=args.wandb_run_name,
        save_dir=args.output_dir,
    )

    callbacks = [
        ModelCheckpoint(
            dirpath=os.path.join(args.output_dir, "checkpoints"),
            filename="step{step:07d}",
            every_n_train_steps=args.save_every,
            save_top_k=-1,
            save_last=True,
        ),
        LearningRateMonitor(logging_interval="step"),
        SampleCallback(
            sample_imgs=sample_imgs,
            every_n_steps=args.sample_every,
            ode_steps=args.ode_steps,
            cfg=args.cfg,
        ),
    ]

    # ── trainer ──────────────────────────────────────────────────────────
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=args.num_gpus,
        strategy=DDPStrategy(find_unused_parameters=True),
        precision="bf16-mixed",
        max_epochs=-1,
        max_steps=args.max_steps,
        gradient_clip_val=1.0,
        gradient_clip_algorithm="norm",
        log_every_n_steps=args.log_every,
        logger=logger,
        callbacks=callbacks,
        default_root_dir=args.output_dir,
        enable_progress_bar=True,
    )

    trainer.fit(lit, datamodule=dm, ckpt_path=args.resume_from)


if __name__ == "__main__":
    main()
