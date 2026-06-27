"""
MAE Pre-training with RCJiT-S/16 architecture.

Architecture (matching JiT-S/16 spec):
  - ViT-S/16: depth=12, hidden=384, heads=6, patch=16, bottleneck=64
  - JiT-style components: BottleneckPatchEmbed, SwiGLU FFN, RMSNorm
  - MAE masking: 75% mask ratio on 224x224 images
  - CLS token only for RCJiT conditioning (ctx_mode="cls")

Design notes:
  - Sincos 2D positional embedding is added BEFORE masking so each visible
    patch carries its spatial position into the encoder (standard MAE trick).
  - RoPE is NOT used in the encoder because subsampling RoPE frequencies for
    masked patches requires non-trivial index-gathering; sincos+standard attn
    achieves the same goal more cleanly.
  - Decoder uses a lightweight 4-block transformer (decoder_dim=192).
  - Loss: per-patch normalised MSE on masked patches only (MAE paper §3.1).
"""

import sys
import os
import math
from functools import partial

sys.path.insert(0, "/home/yspark/rcdm/RCJiT/JiT")

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import stable_pretraining as spt
import lightning as pl
from lightning.pytorch.loggers import WandbLogger

from util.model_util import RMSNorm, get_2d_sincos_pos_embed
from model_jit import BottleneckPatchEmbed, SwiGLUFFN

from MAE import MIMICCXRDataModule
from mimic_cxr import get_MAE_aug
from utils import get_common_callbacks


# ---------------------------------------------------------------------------
# Building blocks (standard attention, no adaLN / no timestep conditioning)
# ---------------------------------------------------------------------------

class _Attention(nn.Module):
    def __init__(self, dim: int, num_heads: int, attn_drop: float = 0.0):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.q_norm = RMSNorm(head_dim)
        self.k_norm = RMSNorm(head_dim)
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)
        self.attn_drop = attn_drop

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)
        x = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.attn_drop if self.training else 0.0
        )
        return self.proj(x.transpose(1, 2).reshape(B, N, C))


class _Block(nn.Module):
    """Standard ViT block: RMSNorm + Attention + RMSNorm + SwiGLU (no adaLN)."""

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.attn = _Attention(dim, num_heads)
        self.norm2 = RMSNorm(dim)
        self.mlp = SwiGLUFFN(dim, int(dim * mlp_ratio))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


# ---------------------------------------------------------------------------
# RCJiT ViT-S/16 Encoder
# ---------------------------------------------------------------------------

class RCJiTViTEncoder(nn.Module):
    """
    ViT-S/16 encoder using JiT-style components for MAE pretraining.

    Specs (matching JiT-S/16 from ~/rcdm/RCJiT/JiT/model_jit.py):
        depth=12, hidden=384, heads=6, patch=16, bottleneck=64
    Image size: 224×224  →  196 patches
    """

    IMG_SIZE     = 224
    PATCH_SIZE   = 16
    IN_CHANNELS  = 3
    HIDDEN_SIZE  = 384
    DEPTH        = 12
    NUM_HEADS    = 6
    BOTTLENECK   = 64
    MASK_RATIO   = 0.75

    def __init__(self):
        super().__init__()
        num_patches = (self.IMG_SIZE // self.PATCH_SIZE) ** 2  # 196

        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.HIDDEN_SIZE))

        # Two-stage bottleneck patch projection (JiT-style)
        self.patch_embed = BottleneckPatchEmbed(
            img_size=self.IMG_SIZE,
            patch_size=self.PATCH_SIZE,
            in_chans=self.IN_CHANNELS,
            pca_dim=self.BOTTLENECK,
            embed_dim=self.HIDDEN_SIZE,
            bias=True,
        )

        # Fixed 2D sincos positional embedding (patches only; added before masking)
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches, self.HIDDEN_SIZE), requires_grad=False
        )

        self.blocks = nn.ModuleList([
            _Block(self.HIDDEN_SIZE, self.NUM_HEADS) for _ in range(self.DEPTH)
        ])
        self.norm = RMSNorm(self.HIDDEN_SIZE)

        self._init_weights()

    # ── init ────────────────────────────────────────────────────────────────

    def _init_weights(self):
        hw = self.IMG_SIZE // self.PATCH_SIZE
        pos = get_2d_sincos_pos_embed(self.HIDDEN_SIZE, hw)
        self.pos_embed.data.copy_(torch.from_numpy(pos).float().unsqueeze(0))

        nn.init.trunc_normal_(self.cls_token, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    @property
    def num_prefix_tokens(self) -> int:
        return 1  # CLS token

    # ── masking ─────────────────────────────────────────────────────────────

    def _mask(self, x: torch.Tensor, mask_ratio: float):
        B, N, D = x.shape
        len_keep = int(N * (1.0 - mask_ratio))

        noise = torch.rand(B, N, device=x.device)
        ids_shuffle = noise.argsort(dim=1)
        ids_restore = ids_shuffle.argsort(dim=1)

        ids_keep = ids_shuffle[:, :len_keep]
        x_vis = x.gather(1, ids_keep.unsqueeze(-1).expand(-1, -1, D))

        # mask: 1 = masked, 0 = visible
        mask = torch.ones(B, N, device=x.device)
        mask.scatter_(1, ids_keep, 0.0)

        return x_vis, mask, ids_restore, ids_keep

    # ── forward ─────────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor, mask_ratio: float | None = None):
        """
        x:          (B, C, H, W) normalised images
        mask_ratio: override MASK_RATIO; pass 0.0 for full encoding (eval)

        Returns:
            encoded:     (B, 1+len_keep, D)  — CLS + visible patch tokens
            mask:        (B, N)               — 1=masked, 0=visible
            ids_restore: (B, N)               — permutation to restore order
            ids_keep:    (B, len_keep)
        """
        if mask_ratio is None:
            mask_ratio = self.MASK_RATIO

        B = x.size(0)
        # Patch embed + sincos pos embed (all patches carry position info)
        tokens = self.patch_embed(x) + self.pos_embed  # (B, N, D)
        N = tokens.shape[1]

        if mask_ratio > 0.0:
            tokens, mask, ids_restore, ids_keep = self._mask(tokens, mask_ratio)
        else:
            mask = torch.zeros(B, N, device=x.device)
            ids_restore = torch.arange(N, device=x.device).unsqueeze(0).expand(B, -1)
            ids_keep = ids_restore.clone()

        # Prepend CLS token (no positional embedding for CLS)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, tokens], dim=1)  # (B, 1 + len_keep, D)

        for block in self.blocks:
            x = block(x)
        return self.norm(x), mask, ids_restore, ids_keep


# ---------------------------------------------------------------------------
# MAE Decoder
# ---------------------------------------------------------------------------

class _MAEDecoder(nn.Module):
    """
    Lightweight MAE decoder: enc_dim → 192, 4 blocks, pixel-space head.
    CLS token is threaded through the decoder but excluded from the loss.
    """

    DEC_DIM   = 192
    DEC_DEPTH = 4
    DEC_HEADS = 3

    def __init__(self, enc_dim: int, patch_size: int, in_channels: int, num_patches: int):
        super().__init__()

        self.embed = nn.Linear(enc_dim, self.DEC_DIM, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.DEC_DIM))

        # Decoder sincos positional embedding (all N patches, no CLS)
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches, self.DEC_DIM), requires_grad=False
        )

        self.blocks = nn.ModuleList([
            _Block(self.DEC_DIM, self.DEC_HEADS) for _ in range(self.DEC_DEPTH)
        ])
        self.norm = RMSNorm(self.DEC_DIM)
        self.head = nn.Linear(self.DEC_DIM, patch_size * patch_size * in_channels)

        self._init_weights(num_patches)

    def _init_weights(self, num_patches: int):
        hw = int(num_patches ** 0.5)
        pos = get_2d_sincos_pos_embed(self.DEC_DIM, hw)
        self.pos_embed.data.copy_(torch.from_numpy(pos).float().unsqueeze(0))
        nn.init.trunc_normal_(self.mask_token, std=0.02)

    def forward(self, encoded: torch.Tensor, ids_restore: torch.Tensor) -> torch.Tensor:
        """
        encoded:     (B, 1+len_keep, enc_dim)
        ids_restore: (B, N)
        Returns:     (B, N, patch_size^2 * C)  — predictions for all N patches
        """
        x = self.embed(encoded)
        cls, vis = x[:, :1], x[:, 1:]

        B, len_keep, _ = vis.shape
        N = ids_restore.shape[1]

        # Insert mask tokens to fill the masked positions
        mask_tokens = self.mask_token.expand(B, N - len_keep, -1)
        vis_full = torch.cat([vis, mask_tokens], dim=1)               # (B, N, dec_dim)
        vis_full = vis_full.gather(
            1, ids_restore.unsqueeze(-1).expand(-1, -1, self.DEC_DIM)
        )                                                               # restore order

        vis_full = vis_full + self.pos_embed                            # add position info

        x = torch.cat([cls, vis_full], dim=1)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)

        return self.head(x[:, 1:])  # skip CLS → (B, N, patch^2 * C)


# ---------------------------------------------------------------------------
# Full MAE model
# ---------------------------------------------------------------------------

class RCJiTMAE(nn.Module):
    """MAE with RCJiT ViT-S/16 encoder and lightweight decoder."""

    def __init__(self):
        super().__init__()
        self.encoder = RCJiTViTEncoder()
        num_patches = (RCJiTViTEncoder.IMG_SIZE // RCJiTViTEncoder.PATCH_SIZE) ** 2
        self.decoder = _MAEDecoder(
            enc_dim=RCJiTViTEncoder.HIDDEN_SIZE,
            patch_size=RCJiTViTEncoder.PATCH_SIZE,
            in_channels=RCJiTViTEncoder.IN_CHANNELS,
            num_patches=num_patches,
        )
        self.patch_size = RCJiTViTEncoder.PATCH_SIZE

    def patchify(self, x: torch.Tensor) -> torch.Tensor:
        """(B, C, H, W) → (B, N, patch_size^2 * C)"""
        p = self.patch_size
        B, C, H, W = x.shape
        h, w = H // p, W // p
        x = x.reshape(B, C, h, p, w, p).permute(0, 2, 4, 3, 5, 1)
        return x.reshape(B, h * w, p * p * C)

    def forward(self, x: torch.Tensor):
        """
        x: (B, 3, 224, 224) normalised
        Returns: (loss scalar, cls_token (B, D))
        """
        encoded, mask, ids_restore, _ = self.encoder(x)

        # CLS token — the only conditioning signal used by RCJiT (ctx_mode="cls")
        cls_token = encoded[:, 0]

        pred = self.decoder(encoded, ids_restore)   # (B, N, p^2*C)

        # Normalised per-patch target (MAE §3.1)
        target = self.patchify(x)
        mean   = target.mean(dim=-1, keepdim=True)
        var    = target.var(dim=-1, keepdim=True)
        target = (target - mean) / (var + 1e-6).sqrt()

        # Mean loss on masked patches only
        loss = ((pred - target) ** 2).mean(dim=-1)  # (B, N)
        loss = (loss * mask).sum() / mask.sum()

        return loss, cls_token


# ---------------------------------------------------------------------------
# Lightning Module (compatible with stable_pretraining callbacks)
# ---------------------------------------------------------------------------

class MAERCJiTS16Module(spt.Module):
    """
    MAE pretraining with RCJiT-S/16 encoder.

    After pretraining, encoder.cls_token + encoder.blocks + encoder.norm
    can be used as a drop-in replacement for DINOv2 in DINOv2Denoiser
    (ctx_mode="cls"; hidden_size=384 → dino_proj maps to RCJiT-S/16 hidden).
    """

    def __init__(self, lr: float = 1.5e-4):
        super().__init__(hparams={"lr": lr})
        self.mae = RCJiTMAE()
        self.gpu_aug = None
        self.val_aug = None
        self.optim = {
            "optimizer": partial(torch.optim.AdamW, lr=lr, betas=(0.9, 0.95), weight_decay=0.05),
            "scheduler": "CosineAnnealingLR",
            "interval": "epoch",
        }

    @property
    def embed_dim(self) -> int:
        return RCJiTViTEncoder.HIDDEN_SIZE

    def on_fit_start(self):
        super().on_fit_start()
        _, self.gpu_aug, self.val_aug = get_MAE_aug(self.device)

    def after_manual_backward(self):
        scaler = self.trainer.precision_plugin.scaler
        scale = scaler.get_scale() if scaler is not None else 1.0
        scaled_norm = torch.nn.utils.clip_grad_norm_(
            self.mae.parameters(), max_norm=float("inf")
        )
        self.log("train/grad_norm", scaled_norm / scale, on_step=True, on_epoch=False)

    def forward(self, batch, stage: str = "fit"):
        img = batch["image"]
        aug = self.gpu_aug if stage == "fit" else self.val_aug
        img = aug({"image": img})["image"]

        if stage != "fit":
            with torch.no_grad():
                # Full encoding (no masking) → CLS token for probe / KNN
                encoded, _, _, _ = self.mae.encoder(img, mask_ratio=0.0)
                cls_token = encoded[:, 0]
            return {
                "loss":      torch.tensor(0.0, device=img.device),
                "cls_token": cls_token,
                "label":     batch["labels"].long(),
            }

        loss, cls_token = self.mae(img)

        param_norm = torch.stack([p.norm() for p in self.mae.parameters()]).norm()
        self.log("train/loss",       loss,       prog_bar=True, on_step=True, on_epoch=True)
        self.log("train/param_norm", param_norm, on_step=True,  on_epoch=False)

        return {
            "loss":      loss,
            "cls_token": cls_token.detach(),
            "label":     batch["labels"].long(),
        }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    pl.seed_everything(42)

    datamodule = MIMICCXRDataModule(
        root="/mnt/nvme1/mimic-cxr-jpg",
        batch_size=256,
        num_workers=22,
        frontal_only=False,
    )

    model = MAERCJiTS16Module(lr=1.5e-4)

    logger = WandbLogger(
        entity="RCJiT",
        project="cxr-ssl",
        name="MAE-RCJiT-S16-224-cls",
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
