"""
RCDM training on MIMIC-CXR with pre-extracted SSL features (RadJEPA / RAD-DINO).

Pipeline:
  1. Feature:  radjepa_<dicom_id>.npy co-located with each image (768-dim CLS token)
  2. Image:    256×256 .npy uint8 → resize 224 → grayscale → [-1, 1]  (NO augmentation)
  3. Denoiser: UNet + Gaussian Diffusion conditioned on loaded feature (RCDM)

Usage:
    # Step 1 — extract features (one-time, same as RCJiT)
    python cxr-ssl/src/extract_radjepa_features.py \
        --data_dir  /mnt/nvme1/mimic-cxr-jpg

    # Step 2 — train (RadJEPA features)
    python cxr-ssl/src/train_rcdm_precomputed.py \
        --data_dir    /mnt/nvme1/mimic-cxr-jpg \
        --feat_prefix radjepa_ \
        --output_dir  output/rcdm_radjepa_cxr \
        --wandb_run_name rcdm-radjepa-300k

    # Step 2 — train (RAD-DINO features)
    python cxr-ssl/src/train_rcdm_precomputed.py \
        --data_dir    /mnt/nvme1/mimic-cxr-jpg \
        --feat_prefix raddino_ \
        --output_dir  output/rcdm_raddino_cxr \
        --wandb_run_name rcdm-raddino-300k
"""

import os, argparse
import numpy as np

import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
import torchvision.utils as vutils
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.strategies import DDPStrategy
from PIL import Image
import pandas as pd

from rcdm.guided_diffusion_rcdm.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
)
from rcdm.guided_diffusion_rcdm.resample import create_named_schedule_sampler

import rcdm.guided_diffusion_rcdm.nn as _rcdm_nn
import torch.utils.checkpoint as _torch_ckpt

def _checkpoint_bf16(func, inputs, params, flag):
    if flag:
        return _torch_ckpt.checkpoint(func, *inputs, use_reentrant=False)
    return func(*inputs)

_rcdm_nn.checkpoint = _checkpoint_bf16


# ---------------------------------------------------------------------------
# Dataset  (same logic as RCJiT counterpart)
# ---------------------------------------------------------------------------

class MIMICPrecomputedDataset(Dataset):
    """
    Returns (image_tensor, feat_tensor) pairs.
    - image: 256×256 .npy uint8 → resize 224 → grayscale → [-1, 1] (no augmentation)
    - feat:  raddino_<dicom_id>.npy co-located with the image (768-dim)
    """

    def __init__(self, data_dir: str, split: str, feat_prefix: str = "raddino_"):
        self.data_dir = data_dir
        self.feat_prefix = feat_prefix

        split_df = pd.read_csv(os.path.join(data_dir, "mimic-cxr-2.0.0-split.csv.gz"))
        split_df = split_df[split_df["split"] == split].reset_index(drop=True)

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

        arr  = np.load(os.path.join(sdir, f"{did}.npy"))     # (256, 256) uint8
        img  = Image.fromarray(arr, mode="L")
        img  = TF.resize(img, [224, 224])
        img  = TF.to_tensor(img)                              # (1, H, W) [0, 1]
        img  = img.repeat(3, 1, 1)                            # (3, H, W) — RCDM expects 3ch
        img  = img * 2.0 - 1.0                                # [-1, 1]

        feat = np.load(os.path.join(sdir, f"{self.feat_prefix}{did}.npy"))  # (768,) float32
        feat = torch.from_numpy(feat).float()
        return img, feat


# ---------------------------------------------------------------------------
# LightningModule
# ---------------------------------------------------------------------------

class RCDMPrecomputedModule(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.save_hyperparameters(vars(args))
        self.args = args

        # Build UNet + Gaussian diffusion — 256px ADM 원본 설정, image_size만 224로 변경
        defaults = model_and_diffusion_defaults()
        defaults.update(dict(
            image_size            = args.img_size,
            num_channels          = args.num_channels,
            num_res_blocks        = args.num_res_blocks,
            num_heads             = args.num_heads,
            num_heads_upsample    = args.num_heads_upsample,
            num_head_channels     = args.num_head_channels,
            attention_resolutions = args.attention_resolutions,
            channel_mult          = args.channel_mult,
            learn_sigma           = args.learn_sigma,
            diffusion_steps       = args.diffusion_steps,
            noise_schedule        = args.noise_schedule,
            timestep_respacing    = "",
            use_fp16              = False,
            dropout               = args.dropout,
            use_scale_shift_norm    = args.use_scale_shift_norm,
            resblock_updown         = args.resblock_updown,
            use_checkpoint          = args.use_checkpoint,
            use_new_attention_order = args.use_new_attention_order,
            rescale_learned_sigmas  = args.rescale_learned_sigmas,
        ))

        model, diffusion = create_model_and_diffusion(
            **defaults,
            G_shared = True,
            feat_cond = True,
            ssl_dim   = args.ssl_dim,
        )

        if args.compile:
            torch._dynamo.config.cache_size_limit = 128
            torch._dynamo.config.optimize_ddp = False
            model = torch.compile(model)

        self.model     = model
        self.diffusion = diffusion
        self.schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)
        self.ema_rates  = [float(r) for r in args.ema_rate.split(",")]
        self.ema_params = None  # on_fit_start에서 초기화

    def on_fit_start(self):
        if self.ema_params is None:
            self.ema_params = [
                [p.data.clone() for p in self.model.parameters()]
                for _ in self.ema_rates
            ]
        else:
            device = next(self.model.parameters()).device
            self.ema_params = [
                [p.to(device) for p in ema_list]
                for ema_list in self.ema_params
            ]

    def on_save_checkpoint(self, checkpoint):
        names = [n for n, _ in self.model.named_parameters()]
        if self.ema_params is not None:
            for rate, ema_list in zip(self.ema_rates, self.ema_params):
                checkpoint[f"ema_{rate}"] = {n: p.cpu() for n, p in zip(names, ema_list)}

    def on_load_checkpoint(self, checkpoint):
        names = [n for n, _ in self.model.named_parameters()]
        loaded = []
        for rate in self.ema_rates:
            key = f"ema_{rate}"
            if key in checkpoint:
                loaded.append([checkpoint[key][n] for n in names])
            else:
                loaded.append(None)
        self.ema_params = loaded if any(x is not None for x in loaded) else None

    def training_step(self, batch, batch_idx):
        imgs, feats = batch                         # (B, 3, 224, 224), (B, 768)
        t, weights = self.schedule_sampler.sample(imgs.shape[0], self.device)

        losses = self.diffusion.training_losses(
            self.model, imgs, t,
            model_kwargs={"feat": feats},
        )
        loss = (losses["loss"] * weights).mean()

        self.log("train/loss", loss, on_step=True, on_epoch=False, prog_bar=True, sync_dist=True)
        return loss

    def on_train_batch_end(self, outputs, batch, batch_idx):
        from rcdm.guided_diffusion_rcdm.nn import update_ema
        model_params = list(self.model.parameters())
        for rate, ema_list in zip(self.ema_rates, self.ema_params):
            update_ema(ema_list, model_params, rate=rate)

    def on_before_optimizer_step(self, optimizer):
        trainable = [p for p in self.model.parameters() if p.requires_grad and p.grad is not None]
        if trainable:
            grad_norm  = torch.stack([p.grad.detach().norm() for p in trainable]).norm()
            param_norm = torch.stack([p.detach().norm()      for p in trainable]).norm()
            self.log("train/grad_norm",  grad_norm,  on_step=True, on_epoch=False, sync_dist=False)
            self.log("train/param_norm", param_norm, on_step=True, on_epoch=False, sync_dist=False)

    def configure_optimizers(self):
        return torch.optim.AdamW(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=self.args.lr, weight_decay=self.args.weight_decay,
        )


# ---------------------------------------------------------------------------
# Sample callback
# ---------------------------------------------------------------------------

class SampleCallback(pl.Callback):
    """Generates images from fixed sample features every N steps."""

    def __init__(self, sample_imgs: torch.Tensor, sample_feats: torch.Tensor,
                 every_n_steps: int, ddpm_steps: int):
        super().__init__()
        self.sample_imgs   = sample_imgs
        self.sample_feats  = sample_feats
        self.every_n_steps = every_n_steps
        self.ddpm_steps    = ddpm_steps

    @torch.no_grad()
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if trainer.global_step == 0 or trainer.global_step % self.every_n_steps != 0:
            return
        if not trainer.is_global_zero:
            return

        model     = pl_module.model
        diffusion = pl_module.diffusion
        model.eval()

        feats = self.sample_feats.to(pl_module.device)
        B     = feats.shape[0]

        # Use DDIM-style respacing for fast sampling
        from rcdm.guided_diffusion_rcdm.respace import SpacedDiffusion, space_timesteps
        fast_diffusion = SpacedDiffusion(
            use_timesteps=space_timesteps(diffusion.num_timesteps, [self.ddpm_steps]),
            betas=diffusion.betas,
            model_mean_type=diffusion.model_mean_type,
            model_var_type=diffusion.model_var_type,
            loss_type=diffusion.loss_type,
        )

        shape = (B, 3, pl_module.args.img_size, pl_module.args.img_size)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            gen = fast_diffusion.ddim_sample_loop(
                model,
                shape,
                device=pl_module.device,
                model_kwargs={"feat": feats},
                progress=False,
            )

        model.train()

        cond_vis = (self.sample_imgs.clamp(-1, 1) + 1) / 2
        gen_vis  = (gen.cpu().clamp(-1, 1) + 1) / 2
        grid = vutils.make_grid(
            torch.cat([cond_vis, gen_vis], dim=0),
            nrow=B, padding=2, pad_value=1.0,
        )
        import wandb
        trainer.logger.experiment.log(
            {"samples": wandb.Image(grid, caption=f"step {trainer.global_step}")},
            step=trainer.global_step,
        )


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser("RCDM precomputed-feature trainer")

    # data
    p.add_argument("--data_dir",    required=True, help="MIMIC-CXR-JPG root")
    p.add_argument("--feat_prefix", default="radjepa_", help="Feature file prefix (e.g. radjepa_ or raddino_)")
    p.add_argument("--img_size",    type=int, default=224)
    p.add_argument("--num_workers", type=int, default=12)

    # model — 256px ADM 원본값 그대로, channel_mult만 224px용으로 조정
    p.add_argument("--use_checkpoint",           action="store_true", default=True,
                   help="gradient checkpointing — 553M UNet은 24GB에서 필수")
    p.add_argument("--no_use_checkpoint",        dest="use_checkpoint", action="store_false")
    p.add_argument("--num_channels",            type=int,   default=256)
    p.add_argument("--num_res_blocks",          type=int,   default=2)
    p.add_argument("--num_heads",               type=int,   default=4)
    p.add_argument("--num_heads_upsample",      type=int,   default=-1)
    p.add_argument("--num_head_channels",       type=int,   default=64)
    p.add_argument("--attention_resolutions",   default="28,14,7",
                   help="224px용: 224//28=8, 224//14=16, 224//7=32 → 256px ADM과 동일 위치에 attention")
    p.add_argument("--channel_mult",            default="1,1,2,2,4,4",
                   help="256px ADM 기본값; 224px에서 224→112→56→28→14→7로 동작")
    p.add_argument("--dropout",                 type=float, default=0.0)
    p.add_argument("--learn_sigma",             action="store_true", default=True)
    p.add_argument("--no_learn_sigma",          dest="learn_sigma", action="store_false")
    p.add_argument("--use_scale_shift_norm",    action="store_true", default=True)
    p.add_argument("--resblock_updown",         action="store_true", default=True)
    p.add_argument("--use_new_attention_order", action="store_true", default=False)
    p.add_argument("--rescale_learned_sigmas",  action="store_true", default=False)
    p.add_argument("--ssl_dim",                 type=int,   default=768,
                   help="SSL feature dim (RAD-DINO / RadJEPA ViT-B → 768)")

    # diffusion
    p.add_argument("--diffusion_steps",   type=int,   default=1000)
    p.add_argument("--noise_schedule",    default="linear", choices=["linear", "cosine"])
    p.add_argument("--schedule_sampler",  default="uniform")

    # training
    p.add_argument("--batch_size",      type=int,   default=64)
    p.add_argument("--lr",              type=float, default=1e-4)
    p.add_argument("--weight_decay",    type=float, default=0.0)
    p.add_argument("--max_steps",       type=int,   default=300_000)
    p.add_argument("--ema_rate",        default="0.9999",
                   help="comma-separated EMA rates (원본 RCDM 기본값)")
    p.add_argument("--lr_anneal_steps", type=int,   default=0,
                   help="linear LR decay (원본 RCDM 기본값=0: constant LR)")
    p.add_argument("--num_gpus",        type=int,   default=-1)
    p.add_argument("--compile",         action="store_true")

    # sampling / logging
    p.add_argument("--sample_every",  type=int,   default=10_000)
    p.add_argument("--num_samples",   type=int,   default=8)
    p.add_argument("--ddpm_steps",    type=int,   default=100,
                   help="DDIM steps for sample callback")
    p.add_argument("--output_dir",    default="output/rcdm_raddino_cxr")
    p.add_argument("--save_every",    type=int,   default=10_000)
    p.add_argument("--resume_from",   default=None)
    p.add_argument("--wandb_project",   default="rcdm-cxr")
    p.add_argument("--wandb_run_name",  default=None)
    p.add_argument("--wandb_run_id",    default=None)
    p.add_argument("--log_every",       type=int,   default=50)
    p.add_argument("--seed",            type=int,   default=42)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    torch.set_float32_matmul_precision('medium')  # 4090 Tensor Core 활용
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
    sample_imgs  = torch.stack([train_ds[i][0] for i in sample_indices])
    sample_feats = torch.stack([train_ds[i][1] for i in sample_indices])

    # ── model ────────────────────────────────────────────────────────────
    model = RCDMPrecomputedModule(args)

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
        ddpm_steps=args.ddpm_steps,
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
