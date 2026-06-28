"""
RCJiT-S/16 training on MIMIC-CXR with pre-extracted RAD-DINO features.

Pipeline:
  1. Feature:  raddino_<dicom_id>.npy co-located with each image (768-dim CLS token)
  2. Image:    256×256 .npy → resize 224 → [-1, 1] RGB  (NO augmentation)
  3. Denoiser: JiT-S/16 flow-matching conditioned on loaded feature

Usage:
    # Step 1 — extract features (one-time, resumable, multi-GPU)
    python cxr-ssl/src/extract_raddino_features.py \
        --data_dir  /mnt/nvme1/mimic-cxr-jpg

    # Step 2 — train
    python cxr-ssl/src/train_rcjit_precomputed.py \
        --data_dir    /mnt/nvme1/mimic-cxr-jpg \
        --output_dir  output/rcjit_s16_raddino_cxr \
        --wandb_run_name rcjit-s16-raddino-300k
"""

import os, sys, math, argparse
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'rcdm', 'RCJiT', 'JiT'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'rcdm', 'RCJiT', 'src'))

import torch
import torchvision.transforms.functional as TF
import torchvision.utils as vutils
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.strategies import DDPStrategy
from PIL import Image
import pandas as pd

from denoiser import DINOv2Denoiser


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class MIMICPrecomputedDataset(Dataset):
    """
    Returns (image_tensor, feat_tensor) pairs.
    - image: 256×256 .npy uint8 → resize 224 → RGB → [-1, 1]  (no augmentation)
    - feat:  raddino_<dicom_id>.npy co-located with the image   (768-dim)
    """

    def __init__(self, data_dir: str, split: str, feat_prefix: str = "raddino_"):
        self.data_dir = data_dir
        self.feat_prefix = feat_prefix

        split_df = pd.read_csv(os.path.join(data_dir, "mimic-cxr-2.0.0-split.csv.gz"))
        split_df = split_df[split_df["split"] == split].reset_index(drop=True)

        # drop rows where image or feature file is missing
        def _study_dir_for(row):
            sid  = str(int(row["subject_id"]))
            stid = str(int(row["study_id"]))
            return os.path.join(data_dir, "files", f"p{sid[:2]}", f"p{sid}", f"s{stid}")

        mask = [
            os.path.exists(os.path.join(_study_dir_for(row), f"{row['dicom_id']}.npy")) and
            os.path.exists(os.path.join(_study_dir_for(row), f"{feat_prefix}{row['dicom_id']}.npy"))
            for _, row in split_df.iterrows()
        ]
        dropped = sum(1 for m in mask if not m)
        if dropped:
            print(f"[MIMICPrecomputedDataset] {split}: skipping {dropped} rows with missing files")
        self.df = split_df[mask].reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def _study_dir(self, row) -> str:
        sid  = str(int(row["subject_id"]))
        stid = str(int(row["study_id"]))
        return os.path.join(self.data_dir, "files", f"p{sid[:2]}", f"p{sid}", f"s{stid}")

    def __getitem__(self, idx):
        row  = self.df.iloc[idx]
        did  = str(row["dicom_id"])
        sdir = self._study_dir(row)

        arr  = np.load(os.path.join(sdir, f"{did}.npy"))   # (256, 256) uint8
        img  = Image.fromarray(arr, mode="L")
        img  = TF.resize(img, [224, 224])
        img  = TF.to_tensor(img)                           # (1, H, W) [0, 1]
        img  = img * 2.0 - 1.0                             # [-1, 1]

        feat = np.load(os.path.join(sdir, f"{self.feat_prefix}{did}.npy"))   # (768,) float32
        feat = torch.from_numpy(feat).float()
        return img, feat


# ---------------------------------------------------------------------------
# LR schedule
# ---------------------------------------------------------------------------

def _warmup_cosine_lambda(warmup_steps: int, max_steps: int, min_lr_ratio: float = 0.01):
    def lr_lambda(step: int):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
        cosine   = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine
    return lr_lambda


# ---------------------------------------------------------------------------
# Sample callback
# ---------------------------------------------------------------------------

class SampleCallback(pl.Callback):
    """Generates images from fixed sample features every N steps."""

    def __init__(self, sample_imgs: torch.Tensor, sample_feats: torch.Tensor,
                 every_n_steps: int, ode_steps: int, cfg: float):
        super().__init__()
        self.sample_imgs  = sample_imgs    # (N, 3, 224, 224) in [-1, 1]
        self.sample_feats = sample_feats   # (N, 768)
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

        feats = self.sample_feats.to(pl_module.device)
        orig_steps, orig_cfg = denoiser.steps, denoiser.cfg_scale
        denoiser.steps, denoiser.cfg_scale = self.ode_steps, self.cfg

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            gen = denoiser.generate_from_dino_cls(feats)

        denoiser.steps, denoiser.cfg_scale = orig_steps, orig_cfg
        denoiser.train()

        cond_vis = (self.sample_imgs.clamp(-1, 1) + 1) / 2
        gen_vis  = (gen.cpu().clamp(-1, 1) + 1) / 2
        grid = vutils.make_grid(
            torch.cat([cond_vis, gen_vis], dim=0),
            nrow=len(feats), padding=2, pad_value=1.0,
        )
        import wandb
        trainer.logger.experiment.log(
            {"samples": wandb.Image(grid, caption=f"step {trainer.global_step}")},
            step=trainer.global_step,
        )


# ---------------------------------------------------------------------------
# LightningModule
# ---------------------------------------------------------------------------

class RCJiTPrecomputedModule(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.save_hyperparameters(vars(args))
        self.args = args

        denoiser_args = argparse.Namespace(
            img_size           = args.img_size,
            in_channels        = 1,
            model_variant      = args.model_variant,
            ctx_mode           = args.ctx_mode,
            encoder_type       = "precomputed",
            encoder_ckpt       = None,
            encoder_config     = None,
            enc_dim            = 768,          # RAD-DINO ViT-B → 768-dim
            attn_dropout       = args.attn_dropout,
            proj_dropout       = args.proj_dropout,
            cond_drop_prob     = args.cond_drop_prob,
            P_mean             = args.P_mean,
            P_std              = args.P_std,
            noise_scale        = args.noise_scale,
            t_eps              = args.t_eps,
            ema_decay1         = args.ema_decay1,
            ema_decay2         = args.ema_decay2,
            sampling_method    = args.ode_method,
            num_sampling_steps = args.ode_steps,
            cfg                = args.cfg,
            interval_min       = args.interval_min,
            interval_max       = args.interval_max,
        )
        denoiser = DINOv2Denoiser(denoiser_args)
        if args.compile:
            torch._dynamo.config.cache_size_limit = 128
            torch._dynamo.config.optimize_ddp = False
            denoiser.net = torch.compile(denoiser.net)
        self.denoiser = denoiser

    def on_fit_start(self):
        import copy
        if self.denoiser.ema_params1 is None:
            params = self.denoiser._trainable_params()
            self.denoiser.ema_params1 = copy.deepcopy(params)
            self.denoiser.ema_params2 = copy.deepcopy(params)
        else:
            device = next(self.denoiser.parameters()).device
            self.denoiser.ema_params1 = [p.to(device) for p in self.denoiser.ema_params1]
            self.denoiser.ema_params2 = [p.to(device) for p in self.denoiser.ema_params2]

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

    def training_step(self, batch, batch_idx):
        imgs, feats = batch                          # (B,3,224,224), (B,768)
        loss = self.denoiser.forward_precomputed(imgs, feats)
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

    def configure_optimizers(self):
        opt = torch.optim.AdamW(
            [p for p in self.denoiser.parameters() if p.requires_grad],
            lr=self.args.lr, betas=(0.9, 0.95), weight_decay=self.args.weight_decay,
        )
        sched = torch.optim.lr_scheduler.LambdaLR(
            opt, lr_lambda=_warmup_cosine_lambda(self.args.warmup_steps, self.args.max_steps),
        )
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sched, "interval": "step"}}


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser("RCJiT precomputed-feature trainer")

    # data
    p.add_argument("--data_dir",    required=True, help="MIMIC-CXR-JPG root")
    p.add_argument("--feat_prefix", default="raddino_", help="Feature file prefix (e.g. raddino_ or radjepa_)")
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

    # sampling
    p.add_argument("--cfg",          type=float, default=1.0)
    p.add_argument("--ctx_mode",     default="cls", choices=["cls", "pool"])
    p.add_argument("--ode_steps",    type=int,   default=50)
    p.add_argument("--ode_method",   default="heun", choices=["euler", "heun"])
    p.add_argument("--interval_min", type=float, default=0.0)
    p.add_argument("--interval_max", type=float, default=1.0)
    p.add_argument("--sample_every", type=int,   default=10_000)
    p.add_argument("--num_samples",  type=int,   default=8)

    # checkpointing / logging
    p.add_argument("--output_dir",  default="output/rcjit_s16_raddino_cxr")
    p.add_argument("--save_every",  type=int, default=10_000)
    p.add_argument("--resume_from", default=None)
    p.add_argument("--wandb_project",  default="rcjit-mae-cxr")
    p.add_argument("--wandb_run_name", default=None)
    p.add_argument("--wandb_run_id",   default=None, help="Resume existing WandB run")
    p.add_argument("--log_every",      type=int, default=50)
    p.add_argument("--compile", action="store_true")
    p.add_argument("--seed",    type=int, default=42)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    pl.seed_everything(args.seed, workers=True)
    os.makedirs(args.output_dir, exist_ok=True)

    # ── dataset ─────────────────────────────────────────────────────────
    train_ds = MIMICPrecomputedDataset(args.data_dir, "train", feat_prefix=args.feat_prefix)
    print(f"Train set: {len(train_ds)} images")

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=(args.num_workers > 0),
        multiprocessing_context="fork",
        prefetch_factor=4,
    )

    # ── sample images & features for callback ───────────────────────────
    sample_indices = list(range(args.num_samples))
    sample_imgs  = torch.stack([train_ds[i][0] for i in sample_indices])   # (N, 3, 224, 224)
    sample_feats = torch.stack([train_ds[i][1] for i in sample_indices])   # (N, 768)

    # ── model ────────────────────────────────────────────────────────────
    model = RCJiTPrecomputedModule(args)

    # ── callbacks ────────────────────────────────────────────────────────
    ckpt_cb = ModelCheckpoint(
        dirpath=os.path.join(args.output_dir, "checkpoints"),
        every_n_train_steps=args.save_every,
        save_top_k=-1,
        filename="{step}",
    )
    lr_cb = LearningRateMonitor(logging_interval="step")
    sample_cb = SampleCallback(
        sample_imgs=sample_imgs,
        sample_feats=sample_feats,
        every_n_steps=args.sample_every,
        ode_steps=args.ode_steps,
        cfg=args.cfg,
    )

    # ── logger ───────────────────────────────────────────────────────────
    logger = WandbLogger(
        project=args.wandb_project,
        name=args.wandb_run_name,
        id=args.wandb_run_id if args.wandb_run_id else None,
        resume="allow" if args.wandb_run_id else None,
        save_dir=args.output_dir,
        log_model=False,
    )

    # ── trainer ──────────────────────────────────────────────────────────
    trainer = pl.Trainer(
        max_steps=args.max_steps,
        accelerator="gpu",
        devices=args.num_gpus,
        precision="bf16-mixed",
        strategy=DDPStrategy(find_unused_parameters=False),
        callbacks=[ckpt_cb, lr_cb, sample_cb],
        logger=logger,
        log_every_n_steps=args.log_every,
        enable_progress_bar=True,
        gradient_clip_val=1.0,
    )

    trainer.fit(model, train_dataloaders=train_loader, ckpt_path=args.resume_from)


if __name__ == "__main__":
    main()
