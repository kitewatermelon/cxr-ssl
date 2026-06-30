import copy
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

from rcjit.model import DINOv2JiT_B_16, DINOv2JiT_S_8, DINOv2JiT_B_8, DINOv2JiT_S_16

_MODEL_REGISTRY = {
    "B_16": DINOv2JiT_B_16,
    "B_8":  DINOv2JiT_B_8,
    "S_8":  DINOv2JiT_S_8,
    "S_16": DINOv2JiT_S_16,
}

# DINOv2-B/14 (the released DINOv2 base model; patch-14 at 224 → 16×16 grid)
DINOV2_MODEL = "vit_base_patch16_224.dino"
DINOV2_DIM = 768       # CLS token dimension
DINOV2_IMG_SIZE = 224  # resolution the model was trained at

# MAE-pretrained ViT-S/16 (cxr-ssl Lightning checkpoint)
MAE_VIT_MODEL = "vit_small_patch16_224"
MAE_VIT_DIM   = 384    # CLS token dimension for ViT-S

# ImageNet normalisation expected by both DINOv2 and MAE ViT
_DINO_MEAN = torch.tensor([0.485, 0.456, 0.406])
_DINO_STD  = torch.tensor([0.229, 0.224, 0.225])

# CheXFound ViT-L/16 (iBot, 4 register tokens)
CHEXFOUND_DIM = 1024


class _CheXFoundEncoder(nn.Module):
    """
    Wraps CheXFound ViT-L/16 teacher to expose a forward_features() interface
    compatible with RCJiTDenoiser.

    forward_features returns {'x_norm_clstoken': (B, 1024), ...}
    We return (B, 1, 1024) so that tokens[:, 0] gives the CLS token.
    """
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        out = self.model.forward_features(x)
        return out['x_norm_clstoken'].unsqueeze(1)   # (B, 1, 1024)


def _load_chexfound_encoder(ckpt_path: str, config_path: str) -> _CheXFoundEncoder:
    """
    Load CheXFound ViT-L/16 teacher from checkpoint.
    Requires CheXFound repo on sys.path (added via --chexfound_dir).
    """
    import argparse, warnings, logging, contextlib, io
    warnings.filterwarnings('ignore')
    logging.disable(logging.CRITICAL)

    from chexfound.eval.setup import build_model_for_eval
    from chexfound.utils.config import setup

    args = argparse.Namespace(
        config_file=config_path,
        pretrained_weights=ckpt_path,
        output_dir='', opts=[]
    )
    with contextlib.redirect_stderr(io.StringIO()):
        config = setup(args)
        config.student.block_chunks = 0   # flatten chunked blocks for weight loading
        model = build_model_for_eval(config, ckpt_path)

    model.eval()
    return _CheXFoundEncoder(model)


def _load_mae_encoder(ckpt_path: str):
    """
    Load MAE-pretrained ViT-S/16 from a cxr-ssl Lightning checkpoint.

    The checkpoint stores encoder weights under the prefix "mae.encoder.vit."
    (stable_pretraining MAEModule → self.mae.encoder.vit).
    """
    ckpt = torch.load(ckpt_path, map_location="cpu")
    prefix = "mae.encoder.vit."
    enc_state = {
        k[len(prefix):]: v
        for k, v in ckpt["state_dict"].items()
        if k.startswith(prefix)
    }
    encoder = timm.create_model(MAE_VIT_MODEL, pretrained=False, num_classes=0, img_size=DINOV2_IMG_SIZE)
    missing, unexpected = encoder.load_state_dict(enc_state, strict=False)
    if missing:
        print(f"[MAE encoder] missing keys: {missing[:5]} ...")
    return encoder


class RCJiTDenoiser(nn.Module):
    """
    Flow-matching denoiser for JiT-B/16 conditioned on DINOv2-B embeddings.

    During training  : forward(x, cond_img) → scalar loss
    During inference : generate(cond_img)   → generated image tensor
    """

    def __init__(self, args):
        super().__init__()

        # ── denoiser backbone ────────────────────────────────────────────
        model_variant = getattr(args, 'model_variant', 'B_16')
        ctx_mode = getattr(args, 'ctx_mode', 'cls')
        model_fn = _MODEL_REGISTRY[model_variant]
        in_channels = getattr(args, 'in_channels', 3)
        self.net = model_fn(
            input_size=args.img_size,
            in_channels=in_channels,
            attn_drop=args.attn_dropout,
            proj_drop=args.proj_dropout,
            ctx_mode=ctx_mode,
        )
        self.img_size = args.img_size
        self.in_channels = in_channels
        hidden_size = self.net.hidden_size

        # ── encoder (frozen) ─────────────────────────────────────────────
        encoder_type   = getattr(args, 'encoder_type', 'dinov2')
        encoder_ckpt   = getattr(args, 'encoder_ckpt', None)
        encoder_config = getattr(args, 'encoder_config', None)

        if encoder_type == 'precomputed':
            # No encoder loaded — features supplied externally at training time
            enc_dim = getattr(args, 'enc_dim', DINOV2_DIM)
            self.dino = None
        elif encoder_type == 'mae':
            assert encoder_ckpt is not None, "--encoder_ckpt required for encoder_type=mae"
            self.dino = _load_mae_encoder(encoder_ckpt)
            enc_dim = MAE_VIT_DIM
        elif encoder_type == 'chexfound':
            assert encoder_ckpt is not None, "--encoder_ckpt required for encoder_type=chexfound"
            assert encoder_config is not None, "--encoder_config required for encoder_type=chexfound"
            self.dino = _load_chexfound_encoder(encoder_ckpt, encoder_config)
            enc_dim = CHEXFOUND_DIM
        else:
            self.dino = timm.create_model(DINOV2_MODEL, pretrained=True, num_classes=0, img_size=DINOV2_IMG_SIZE)
            enc_dim = DINOV2_DIM

        if self.dino is not None:
            for p in self.dino.parameters():
                p.requires_grad = False

        self.ctx_mode = ctx_mode
        self.enc_img_size = DINOV2_IMG_SIZE

        # ── conditioning projection + CFG null tokens ────────────────────
        self.dino_proj = nn.Linear(enc_dim, hidden_size)
        self.null_cond = nn.Parameter(torch.zeros(1, hidden_size))
        if self.ctx_mode == "patch":
            num_dino_patches = (DINOV2_IMG_SIZE // 16) ** 2  # 196 for patch16/224
            self.null_patch_tokens = nn.Parameter(torch.zeros(1, num_dino_patches, enc_dim))

        # ── diffusion / flow-matching hyper-params ────────────────────────
        self.cond_drop_prob = args.cond_drop_prob
        self.P_mean = args.P_mean
        self.P_std = args.P_std
        self.t_eps = args.t_eps
        self.noise_scale = args.noise_scale

        # ── EMA bookkeeping (only trainable params) ───────────────────────
        self.ema_decay1 = args.ema_decay1
        self.ema_decay2 = args.ema_decay2
        self.ema_params1 = None
        self.ema_params2 = None

        # ── sampling config ───────────────────────────────────────────────
        self.method = args.sampling_method
        self.steps = args.num_sampling_steps
        self.cfg_scale = args.cfg
        self.cfg_interval = (args.interval_min, args.interval_max)

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def _normalize_for_dino(self, img: torch.Tensor) -> torch.Tensor:
        """Convert [-1, 1] image to DINOv2 ImageNet-normalised image."""
        img = (img * 0.5 + 0.5).clamp(0, 1)               # [0, 1]
        mean = _DINO_MEAN.to(img.device, img.dtype).view(1, 3, 1, 1)
        std  = _DINO_STD .to(img.device, img.dtype).view(1, 3, 1, 1)
        return (img - mean) / std

    @torch.no_grad()
    def encode_dino(self, cond_img: torch.Tensor):
        """
        cond_img: (B, 3, H, W) in [-1, 1]
        returns:  cls (B, enc_dim)
                  patches (B, N, enc_dim) or None when ctx_mode != "patch"
        """
        x = self._normalize_for_dino(cond_img)
        if x.shape[-2] != self.enc_img_size or x.shape[-1] != self.enc_img_size:
            x = F.interpolate(x, size=self.enc_img_size, mode='bilinear', align_corners=False)
        tokens = self.dino.forward_features(x)
        if self.ctx_mode == "pool":
            cls = tokens[:, 1:].mean(dim=1)
        else:
            cls = tokens[:, 0]                 # CLS token
        patches = tokens[:, 1:] if self.ctx_mode == "patch" else None
        return cls, patches

    def get_cond_emb(
        self,
        dino_cls: torch.Tensor,
        drop_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Project DINOv2 CLS → hidden_size; replace dropped rows with null_cond.

        drop_mask: (B,) bool, True = use null (CFG dropout during training)
        """
        cond = self.dino_proj(dino_cls)                        # (B, hidden_size)
        if drop_mask is not None and drop_mask.any():
            null = self.null_cond.expand(cond.size(0), -1)
            cond = torch.where(drop_mask.unsqueeze(-1), null, cond)
        return cond

    def get_patch_tokens(
        self,
        patches: torch.Tensor,
        drop_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Apply CFG dropout to patch tokens; replace dropped rows with null_patch_tokens.

        patches:   (B, N, DINOV2_DIM)
        drop_mask: (B,) bool, True = use null
        """
        if drop_mask is not None and drop_mask.any():
            null = self.null_patch_tokens.expand(patches.size(0), -1, -1)
            patches = torch.where(drop_mask.view(-1, 1, 1), null, patches)
        return patches

    def _sample_t(self, n: int, device: torch.device) -> torch.Tensor:
        z = torch.randn(n, device=device) * self.P_std + self.P_mean
        return torch.sigmoid(z)

    # ------------------------------------------------------------------
    # training forward
    # ------------------------------------------------------------------

    def forward_precomputed(self, x: torch.Tensor, cond_feat: torch.Tensor) -> torch.Tensor:
        """
        x:         (B, 3, H, W) clean image in [-1, 1]
        cond_feat: (B, enc_dim) pre-extracted CLS token feature
        returns:   scalar flow-matching loss
        """
        drop_mask = torch.rand(x.size(0), device=x.device) < self.cond_drop_prob
        cond_emb  = self.get_cond_emb(cond_feat, drop_mask)

        t = self._sample_t(x.size(0), x.device).view(-1, 1, 1, 1)
        e = torch.randn_like(x) * self.noise_scale

        z = t * x + (1 - t) * e
        v = (x - z) / (1 - t).clamp_min(self.t_eps)

        x_pred = self.net(z, t.flatten(), cond_emb, None)
        v_pred = (x_pred - z) / (1 - t).clamp_min(self.t_eps)
        return F.mse_loss(v_pred, v)

    def forward(self, x: torch.Tensor, cond_img: torch.Tensor) -> torch.Tensor:
        """
        x:        (B, 3, H, W) clean image in [-1, 1]
        cond_img: (B, 3, H, W) conditioning image in [-1, 1]
        returns:  scalar flow-matching loss
        """
        dino_cls, patches = self.encode_dino(cond_img)

        drop_mask = torch.rand(x.size(0), device=x.device) < self.cond_drop_prob
        cond_emb = self.get_cond_emb(dino_cls, drop_mask)
        patch_tokens = self.get_patch_tokens(patches, drop_mask) if self.ctx_mode == "patch" else None

        t = self._sample_t(x.size(0), x.device).view(-1, 1, 1, 1)
        e = torch.randn_like(x) * self.noise_scale

        z = t * x + (1 - t) * e
        v = (x - z) / (1 - t).clamp_min(self.t_eps)

        x_pred = self.net(z, t.flatten(), cond_emb, patch_tokens)
        v_pred = (x_pred - z) / (1 - t).clamp_min(self.t_eps)

        return ((v - v_pred) ** 2).mean(dim=(1, 2, 3)).mean()

    # ------------------------------------------------------------------
    # generation (ODE integration with CFG)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def generate(self, cond_img: torch.Tensor) -> torch.Tensor:
        """
        cond_img: (B, 3, H, W) in [-1, 1]
        returns:  (B, 3, H, W) generated image in [-1, 1]
        """
        dino_cls, patches = self.encode_dino(cond_img)
        return self.generate_from_dino_cls(dino_cls, patches if self.ctx_mode == "patch" else None)

    @torch.no_grad()
    def generate_from_dino_cls(
        self,
        dino_cls: torch.Tensor,
        patches: Optional[torch.Tensor] = None,
        noise: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Generate from pre-computed DINOv2 tokens.

        dino_cls: (B, DINOV2_DIM)
        patches:  (B, N, DINOV2_DIM) patch tokens; if None uses null_patch_tokens
        noise:    optional (B, 3, H, W) starting noise; sampled if None
        returns:  (B, 3, H, W) in [-1, 1]
        """
        device = dino_cls.device
        B = dino_cls.size(0)

        cond_emb   = self.get_cond_emb(dino_cls)
        uncond_emb = self.null_cond.expand(B, -1)

        if self.ctx_mode == "patch":
            cond_patches   = patches if patches is not None else self.null_patch_tokens.expand(B, -1, -1)
            uncond_patches = self.null_patch_tokens.expand(B, -1, -1)
        else:
            cond_patches = uncond_patches = None

        if noise is None:
            z = self.noise_scale * torch.randn(B, self.in_channels, self.img_size, self.img_size, device=device)
        else:
            z = noise.to(device)

        ts = torch.linspace(0.0, 1.0, self.steps + 1, device=device)
        ts = ts.view(-1, *([1] * z.ndim)).expand(-1, B, *([1] * (z.ndim - 1)))

        stepper = self._euler_step if self.method == "euler" else self._heun_step

        for i in range(self.steps - 1):
            z = stepper(z, ts[i], ts[i + 1], cond_emb, uncond_emb, cond_patches, uncond_patches)
        z = self._euler_step(z, ts[-2], ts[-1], cond_emb, uncond_emb, cond_patches, uncond_patches)
        return z

    @torch.no_grad()
    def _forward_sample(
        self,
        z: torch.Tensor,
        t: torch.Tensor,
        cond_emb: torch.Tensor,
        uncond_emb: torch.Tensor,
        cond_patches: torch.Tensor,
        uncond_patches: torch.Tensor,
    ) -> torch.Tensor:
        x_c = self.net(z, t.flatten(), cond_emb, cond_patches)
        v_c = (x_c - z) / (1.0 - t).clamp_min(self.t_eps)

        x_u = self.net(z, t.flatten(), uncond_emb, uncond_patches)
        v_u = (x_u - z) / (1.0 - t).clamp_min(self.t_eps)

        lo, hi = self.cfg_interval
        in_interval = (t < hi) & ((lo == 0) | (t > lo))
        scale = torch.where(in_interval, torch.tensor(self.cfg_scale, device=t.device), torch.ones_like(t))
        return v_u + scale * (v_c - v_u)

    @torch.no_grad()
    def _euler_step(self, z, t, t_next, cond_emb, uncond_emb, cond_patches, uncond_patches):
        v = self._forward_sample(z, t, cond_emb, uncond_emb, cond_patches, uncond_patches)
        return z + (t_next - t) * v

    @torch.no_grad()
    def _heun_step(self, z, t, t_next, cond_emb, uncond_emb, cond_patches, uncond_patches):
        v_t = self._forward_sample(z, t, cond_emb, uncond_emb, cond_patches, uncond_patches)
        z_e = z + (t_next - t) * v_t
        v_tn = self._forward_sample(z_e, t_next, cond_emb, uncond_emb, cond_patches, uncond_patches)
        return z + (t_next - t) * 0.5 * (v_t + v_tn)

    # ------------------------------------------------------------------
    # EMA (tracks only trainable parameters to skip frozen DINO weights)
    # ------------------------------------------------------------------

    def _trainable_params(self):
        return [p for p in self.parameters() if p.requires_grad]

    @torch.no_grad()
    def update_ema(self):
        src = self._trainable_params()
        for t, s in zip(self.ema_params1, src):
            t.detach().mul_(self.ema_decay1).add_(s, alpha=1 - self.ema_decay1)
        for t, s in zip(self.ema_params2, src):
            t.detach().mul_(self.ema_decay2).add_(s, alpha=1 - self.ema_decay2)
