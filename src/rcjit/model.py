import torch
import torch.nn as nn
import numpy as np
from typing import Optional

from rcjit.jit.model_jit import BottleneckPatchEmbed, TimestepEmbedder, JiTBlock, FinalLayer
from rcjit.jit.util.model_util import VisionRotaryEmbeddingFast, get_2d_sincos_pos_embed


class DINOv2JiT(nn.Module):
    """
    ctx_mode="cls"   : in-context tokens = CLS embedding broadcast to in_context_len copies.
    ctx_mode="patch" : in-context tokens = cross-attention over DINOv2 patch tokens.
    """

    def __init__(
        self,
        input_size: int = 256,
        patch_size: int = 16,
        in_channels: int = 3,
        hidden_size: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        bottleneck_dim: int = 128,
        in_context_len: int = 32,
        in_context_start: int = 4,
        ctx_mode: str = "cls",      # "cls" | "patch"
        dino_patch_dim: int = 768,  # only used when ctx_mode="patch"
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.in_context_len = in_context_len
        self.in_context_start = in_context_start
        self.ctx_mode = ctx_mode

        self.t_embedder = TimestepEmbedder(hidden_size)

        self.x_embedder = BottleneckPatchEmbed(
            input_size, patch_size, in_channels, bottleneck_dim, hidden_size, bias=True
        )
        num_patches = self.x_embedder.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size), requires_grad=False)

        if in_context_len > 0:
            self.in_context_posemb = nn.Parameter(torch.zeros(1, in_context_len, hidden_size))
            nn.init.normal_(self.in_context_posemb, std=0.02)

            if ctx_mode == "patch":
                self.ctx_queries    = nn.Parameter(torch.zeros(1, in_context_len, hidden_size))
                nn.init.normal_(self.ctx_queries, std=0.02)
                self.ctx_patch_proj = nn.Linear(dino_patch_dim, hidden_size)
                self.ctx_cross_attn = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)

        half_head_dim = hidden_size // num_heads // 2
        hw_seq_len = input_size // patch_size
        self.feat_rope = VisionRotaryEmbeddingFast(
            dim=half_head_dim, pt_seq_len=hw_seq_len, num_cls_token=0
        )
        self.feat_rope_incontext = VisionRotaryEmbeddingFast(
            dim=half_head_dim, pt_seq_len=hw_seq_len, num_cls_token=in_context_len
        )

        self.blocks = nn.ModuleList([
            JiTBlock(
                hidden_size, num_heads, mlp_ratio=mlp_ratio,
                attn_drop=attn_drop if (depth // 4 * 3 > i >= depth // 4) else 0.0,
                proj_drop=proj_drop if (depth // 4 * 3 > i >= depth // 4) else 0.0,
            )
            for i in range(depth)
        ])

        self.final_layer = FinalLayer(hidden_size, patch_size, in_channels)
        self._init_weights()

    def _init_weights(self):
        def _basic(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        self.apply(_basic)

        pos_embed = get_2d_sincos_pos_embed(
            self.pos_embed.shape[-1], int(self.x_embedder.num_patches ** 0.5)
        )
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        w1 = self.x_embedder.proj1.weight.data
        nn.init.xavier_uniform_(w1.view(w1.shape[0], -1))
        w2 = self.x_embedder.proj2.weight.data
        nn.init.xavier_uniform_(w2.view(w2.shape[0], -1))
        nn.init.zeros_(self.x_embedder.proj2.bias)

        if self.ctx_mode == "patch" and self.in_context_len > 0:
            nn.init.xavier_uniform_(self.ctx_patch_proj.weight)
            nn.init.zeros_(self.ctx_patch_proj.bias)

        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        for block in self.blocks:
            nn.init.zeros_(block.adaLN_modulation[-1].weight)
            nn.init.zeros_(block.adaLN_modulation[-1].bias)

        nn.init.zeros_(self.final_layer.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.final_layer.adaLN_modulation[-1].bias)
        nn.init.zeros_(self.final_layer.linear.weight)
        nn.init.zeros_(self.final_layer.linear.bias)

    def unpatchify(self, x: torch.Tensor) -> torch.Tensor:
        p = self.patch_size
        c = self.out_channels
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]
        x = x.reshape(x.shape[0], h, w, p, p, c)
        x = torch.einsum('nhwpqc->nchpwq', x)
        return x.reshape(x.shape[0], c, h * p, h * p)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        cond_emb: torch.Tensor,
        patch_tokens: Optional[torch.Tensor] = None,  # (B, N, dino_patch_dim), ctx_mode="patch" only
    ) -> torch.Tensor:
        t_emb = self.t_embedder(t)
        c = t_emb + cond_emb

        x = self.x_embedder(x) + self.pos_embed
        B = x.shape[0]

        for i, block in enumerate(self.blocks):
            if self.in_context_len > 0 and i == self.in_context_start:
                if self.ctx_mode == "patch":
                    q  = self.ctx_queries.expand(B, -1, -1)
                    kv = self.ctx_patch_proj(patch_tokens)
                    ctx, _ = self.ctx_cross_attn(q, kv, kv)
                else:  # "cls"
                    ctx = cond_emb.unsqueeze(1).expand(-1, self.in_context_len, -1)
                ctx = ctx + self.in_context_posemb
                x = torch.cat([ctx, x], dim=1)
            rope = self.feat_rope if i < self.in_context_start else self.feat_rope_incontext
            x = block(x, c, rope)

        x = x[:, self.in_context_len:]
        x = self.final_layer(x, c)
        return self.unpatchify(x)


def DINOv2JiT_B_16(**kwargs) -> DINOv2JiT:
    return DINOv2JiT(
        depth=12, hidden_size=768, num_heads=12,
        bottleneck_dim=128, in_context_len=32, in_context_start=4,
        patch_size=16, dino_patch_dim=768, **kwargs,
    )


def DINOv2JiT_B_8(**kwargs) -> DINOv2JiT:
    return DINOv2JiT(
        depth=12, hidden_size=768, num_heads=12,
        bottleneck_dim=128, in_context_len=32, in_context_start=4,
        patch_size=8, dino_patch_dim=768, **kwargs,
    )


def DINOv2JiT_S_8(**kwargs) -> DINOv2JiT:
    return DINOv2JiT(
        depth=12, hidden_size=384, num_heads=6,
        bottleneck_dim=64, in_context_len=32, in_context_start=4,
        patch_size=8, dino_patch_dim=768, **kwargs,
    )


def DINOv2JiT_S_16(**kwargs) -> DINOv2JiT:
    return DINOv2JiT(
        depth=12, hidden_size=384, num_heads=6,
        bottleneck_dim=64, in_context_len=16, in_context_start=4,
        patch_size=16, dino_patch_dim=768, **kwargs,
    )


# ---------------------------------------------------------------------------
# sanity check
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    B, t = 2, torch.rand(2, device=device)
    patches = torch.randn(B, 196, 768, device=device)

    # cls mode
    m = DINOv2JiT_B_16(input_size=256, ctx_mode="cls").to(device)
    x = torch.randn(B, 3, 256, 256, device=device)
    cond = torch.randn(B, 768, device=device)
    print("B/256 cls:", m(x, t, cond).shape)

    # patch mode
    m2 = DINOv2JiT_B_16(input_size=256, ctx_mode="patch").to(device)
    print("B/256 patch:", m2(x, t, cond, patches).shape)

    # S/128 cls
    ms = DINOv2JiT_S_8(input_size=128, ctx_mode="cls").to(device)
    print("S/128 cls:", ms(torch.randn(B, 3, 128, 128, device=device), t, torch.randn(B, 384, device=device)).shape)
