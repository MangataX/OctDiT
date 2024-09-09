"""
Adapted from: https://github.com/openai/openai/blob/55363aa496049423c37124b440e9e30366db3ed6/orc/orc/diffusion/vit.py
"""

import math
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Literal

import torch
import torch.nn as nn
from torch import Tensor
from ocnn.octree import xyz2key
from jaxtyping import Float, Int64
from einops import rearrange

from .checkpoint import checkpoint
from .util import timestep_embedding


def init_linear(l, stddev):
    nn.init.normal_(l.weight, std=stddev)
    if l.bias is not None:
        nn.init.constant_(l.bias, 0.0)


class MultiheadAttention(nn.Module):
    def __init__(
        self,
        *,
        device: torch.device,
        dtype: torch.dtype,
        n_ctx: int,
        width: int,
        heads: int,
        init_scale: float,
    ):
        super().__init__()
        self.n_ctx = n_ctx
        self.width = width
        self.heads = heads
        self.c_qkv = nn.Linear(width, width * 3, device=device, dtype=dtype)
        self.c_proj = nn.Linear(width, width, device=device, dtype=dtype)
        self.attention = QKVMultiheadAttention(device=device, dtype=dtype, heads=heads, n_ctx=n_ctx)
        init_linear(self.c_qkv, init_scale)
        init_linear(self.c_proj, init_scale)

    def forward(self, x):
        x = self.c_qkv(x)
        x = checkpoint(self.attention, (x,), (), True)
        x = self.c_proj(x)
        return x


class MLP(nn.Module):
    def __init__(self, *, device: torch.device, dtype: torch.dtype, width: int, init_scale: float):
        super().__init__()
        self.width = width
        self.c_fc = nn.Linear(width, width * 4, device=device, dtype=dtype)
        self.c_proj = nn.Linear(width * 4, width, device=device, dtype=dtype)
        self.gelu = nn.GELU()
        init_linear(self.c_fc, init_scale)
        init_linear(self.c_proj, init_scale)

    def forward(self, x):
        return self.c_proj(self.gelu(self.c_fc(x)))


class QKVMultiheadAttention(nn.Module):
    def __init__(self, *, device: torch.device, dtype: torch.dtype, heads: int, n_ctx: int):
        super().__init__()
        self.device = device
        self.dtype = dtype
        self.heads = heads
        self.n_ctx = n_ctx

    def forward(self, qkv):
        bs, n_ctx, width = qkv.shape
        attn_ch = width // self.heads // 3
        scale = 1 / math.sqrt(math.sqrt(attn_ch))
        qkv = qkv.view(bs, n_ctx, self.heads, -1) # [B, N, H, C//H]
        q, k, v = torch.split(qkv, attn_ch, dim=-1) # [B, N, H, C//3H]
        weight = torch.einsum(
            "bthc,bshc->bhts", q * scale, k * scale
        )  # More stable with f16 than dividing afterwards, [B, H, N, N]
        wdtype = weight.dtype
        weight = torch.softmax(weight.float(), dim=-1).type(wdtype)
        return torch.einsum("bhts,bshc->bthc", weight, v).reshape(bs, n_ctx, -1)


class ResidualAttentionBlock(nn.Module):
    def __init__(
        self,
        *,
        device: torch.device,
        dtype: torch.dtype,
        n_ctx: int,
        width: int,
        heads: int,
        init_scale: float = 1.0,
    ):
        super().__init__()

        self.attn = MultiheadAttention(
            device=device,
            dtype=dtype,
            n_ctx=n_ctx,
            width=width,
            heads=heads,
            init_scale=init_scale,
        )
        self.ln_1 = nn.LayerNorm(width, device=device, dtype=dtype)
        self.mlp = MLP(device=device, dtype=dtype, width=width, init_scale=init_scale)
        self.ln_2 = nn.LayerNorm(width, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class Transformer(nn.Module):
    def __init__(
        self,
        *,
        device: torch.device,
        dtype: torch.dtype,
        n_ctx: int,
        width: int,
        layers: int,
        heads: int,
        init_scale: float = 0.25,
    ):
        super().__init__()
        self.n_ctx = n_ctx
        self.width = width
        self.layers = layers
        init_scale = init_scale * math.sqrt(1.0 / width)
        self.resblocks = nn.ModuleList(
            [
                ResidualAttentionBlock(
                    device=device,
                    dtype=dtype,
                    n_ctx=n_ctx,
                    width=width,
                    heads=heads,
                    init_scale=init_scale,
                )
                for _ in range(layers)
            ]
        )

    def forward(self, x: torch.Tensor):
        for block in self.resblocks:
            x = block(x)
        return x

class PointDiffusionTransformer(nn.Module):
    def __init__(
        self,
        *,
        device: torch.device,
        dtype: torch.dtype,
        input_channels: int = 3,
        output_channels: int = 6,
        n_ctx: int = 1024,
        width: int = 512,
        layers: int = 12,
        heads: int = 8,
        init_scale: float = 0.25,
        time_token_cond: bool = False,
    ):
        super().__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.n_ctx = n_ctx
        self.time_token_cond = time_token_cond
        self.time_embed = MLP(
            device=device, dtype=dtype, width=width, init_scale=init_scale * math.sqrt(1.0 / width)
        )
        self.ln_pre = nn.LayerNorm(width, device=device, dtype=dtype)
        self.backbone = Transformer(
            device=device,
            dtype=dtype,
            n_ctx=n_ctx + int(time_token_cond),
            width=width,
            layers=layers,
            heads=heads,
            init_scale=init_scale,
        )
        self.ln_post = nn.LayerNorm(width, device=device, dtype=dtype)
        self.input_proj = nn.Linear(input_channels, width, device=device, dtype=dtype)
        self.output_proj = nn.Linear(width, output_channels, device=device, dtype=dtype)
        with torch.no_grad():
            self.output_proj.weight.zero_()
            self.output_proj.bias.zero_()

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        """
        :param x: an [N x C x T] tensor.
        :param t: an [N] tensor.
        :return: an [N x C' x T] tensor.
        """
        assert x.shape[-1] == self.n_ctx
        t_embed = self.time_embed(timestep_embedding(t, self.backbone.width))
        return self._forward_with_cond(x, [(t_embed, self.time_token_cond)])

    def _forward_with_cond(
        self, x: torch.Tensor, cond_as_token: List[Tuple[torch.Tensor, bool]]
    ) -> torch.Tensor:
        h = self.input_proj(x.permute(0, 2, 1))  # NCL -> NLC
        for emb, as_token in cond_as_token:
            if not as_token:
                h = h + emb[:, None]
        extra_tokens = [
            (emb[:, None] if len(emb.shape) == 2 else emb)
            for emb, as_token in cond_as_token
            if as_token
        ]
        if len(extra_tokens):
            h = torch.cat(extra_tokens + [h], dim=1)

        h = self.ln_pre(h)
        h = self.backbone(h)
        h = self.ln_post(h)
        if len(extra_tokens):
            h = h[:, sum(h.shape[1] for h in extra_tokens) :]
        h = self.output_proj(h)
        return h.permute(0, 2, 1)
    
class ShuffledKeyTransformer(nn.Module):
    '''
    t as condition (similar to DiT), not simply add to x
    '''
    def __init__(
        self,
        *,
        device: torch.device,
        dtype: torch.dtype,
        input_channels: int = 3,
        output_channels: int = 6,
        window_size: int | None = None,
        n_ctx: int = 1024,
        # width: int = 512,
        width: int = 384,
        layers: int = 12,
        heads: int = 8,
    ):
        super().__init__()
        self.device = device
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.n_ctx = n_ctx
        self.width = width
        self.time_embedder = TimestepEmbedder(width).to(device)
        self.ln_pre = nn.LayerNorm(width, device=device, dtype=dtype)
        self.blocks = nn.ModuleList([
            ConditionBlock(
                hidden_size=width,
                num_heads=heads,
                window_size=window_size,
                mlp_ratio=4.0,
            ).to(device) for _ in range(layers)
        ])
        self.ln_post = nn.LayerNorm(width, device=device, dtype=dtype)
        self.input_proj = nn.Linear(input_channels, width, device=device, dtype=dtype)
        self.output_proj = nn.Linear(width, output_channels, device=device, dtype=dtype)
        self.window_size = window_size
        self.decoder_dx = Mlp(
            in_features=width,
            out_features=3,
        ).to(device)
        self._initialize_weights()
        with torch.no_grad():
            self.output_proj.weight.zero_()
            self.output_proj.bias.zero_()

    def _initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        nn.init.normal_(self.time_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.time_embedder.mlp[2].weight, std=0.02)

        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
        
    def forward(self, x: Float[Tensor, "B C N"], t: Int64[Tensor, "B"]) -> Float[Tensor, "B OC N"]:
        """
        :param x: an [N x C x T] tensor.
        :param t: an [N] tensor.
        :return: an [N x C' x T] tensor.
        """
        assert x.shape[-1] == self.n_ctx
        t_embed = self.time_embedder(t)
        return self._forward_with_cond(x, [t_embed])

    def _forward_with_cond(
        self, x: torch.Tensor, cond_as_token: List[torch.Tensor]
    ) -> torch.Tensor:
        x = x.permute(0, 2, 1).contiguous()  # BCN -> BNC
        x = (x / 4.).clamp(-1., 1.)
        r = 4096
        keys = xyz2key(
            x[..., 0] * r + r,
            x[..., 1] * r + r,
            x[..., 2] * r + r,
        ) # [B, N]
        pre_indices = keys.sort().indices.unsqueeze(-1) # [B, N, 1]
        assert pre_indices.shape == (x.shape[0], x.shape[1], 1)
        x = x.gather(dim=1, index=pre_indices.expand_as(x)) # [B, N, 3]
        pos_emb = get_positional_embed(
            x=x[..., 0],
            y=x[..., 1],
            z=x[..., 2],
            condition_dim=self.width,
            device=self.device,
        ) # [B, N, CD]
        h = pos_emb
        condition = cond_as_token[0]
        for i, block in enumerate(self.blocks):
            # h = block(h, condition)
            h = pos_emb + block(h, condition)
            if (i + 1) % 3 == 0:
                dx = self.decoder_dx(h).clamp(-1., 1.)
                x = x + dx # [B, N, 3]
                keys = xyz2key(
                    x[..., 0] * r + r,
                    x[..., 1] * r + r,
                    x[..., 2] * r + r,
                ) # [B, N]
                indices = keys.sort().indices.unsqueeze(-1) # [B, N, 1]
                assert indices.shape == (x.shape[0], x.shape[1], 1)
                x = x.gather(dim=1, index=indices.expand_as(x)) # [B, N, 3]
                h = h.gather(dim=1, index=indices.expand_as(h)) # [B, N, CD]
                pos_emb = pos_emb.gather(dim=1, index=indices.expand_as(pos_emb))
                pre_indices = pre_indices.gather(dim=1, index=indices.expand_as(pre_indices))

        h: Tensor = self.output_proj(h)
        h = torch.empty_like(h).scatter(dim=1, index=pre_indices.expand_as(h), src=h) # [B, N, 2C]
        return h.permute(0, 2, 1)
    
class CrossAttention(nn.Module):
    def __init__(
        self, 
        query_dim: int, 
        context_dim: int, 
        heads: int = 8, 
        dim_head: int = 64, 
        dropout: float = 0.,
        window_size: int | None = None,
    ):
        super().__init__()
        inner_dim = dim_head * heads

        self.scale = dim_head ** -0.5
        self.heads = heads
        self.window_size = window_size

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(dropout)
        )

    def forward(self, query: Tensor, context: Tensor, mask=None):
        h = self.heads
        batch_size = query.shape[0]

        if self.window_size is not None:
            query = query.view(batch_size, -1, self.window_size, query.shape[-1])
            query = query.view(-1, self.window_size, query.shape[-1])
            context = context.view(batch_size, -1, self.window_size, context.shape[-1])
            context = query.view(-1, self.window_size, context.shape[-1])

        q = self.to_q(query)

        k = self.to_k(context)
        v = self.to_v(context)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))

        sim: Tensor = torch.einsum('b i d, b j d -> b i j', q, k) * self.scale

        # attention, what we cannot get enough of
        attn = sim.softmax(dim=-1)

        out = torch.einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)

        if self.window_size is not None:
            out = out.view(batch_size, -1, out.shape[-1])

        return self.to_out(out)
    
    
class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """
    def __init__(self, hidden_size):
        super().__init__()
        self.embedding_table = nn.Embedding(1, hidden_size)

    def forward(self, labels):
        embeddings = self.embedding_table(labels)
        return embeddings
    
class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb

class ConditionBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    def __init__(self, hidden_size, num_heads, window_size: int | None = None, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        # approx_gelu = lambda: nn.GELU(approximate="tanh")
        approx_gelu = lambda: nn.GELU() # for torch 1.7.1
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )
        self.window_size = window_size

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = \
        self.adaLN_modulation(c).chunk(6, dim=1)
                
        shortcut = x
        x = self.norm1(x)
        B = x.shape[0]
        N = x.shape[1]
        CD = x.shape[-1]
        x: Tensor = modulate(x.reshape(B, -1, CD), shift_msa, scale_msa)
        if self.window_size is not None:
            assert N % self.window_size == 0
            x = x.view(B, N // self.window_size, self.window_size, CD)
            x = x.view(-1, self.window_size, CD)
            x = self.attn(x)
            x = x.view(B, -1, self.window_size, CD)
            x = x.view(B, N, CD)
        else:
            x = self.attn(x)
        
        x = shortcut + gate_msa.unsqueeze(1) * x
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        
        return x
    
def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(
            self,
            in_features,
            hidden_features=None,
            out_features=None,
            act_layer=nn.GELU,
            norm_layer=None,
            bias=True,
            drop=0.,
            use_conv=False,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        # bias = to_2tuple(bias)
        # drop_probs = to_2tuple(drop)
        bias = (bias, bias)
        drop_probs = (drop, drop)
        from functools import partial
        linear_layer = partial(nn.Conv2d, kernel_size=1) if use_conv else nn.Linear

        self.fc1 = linear_layer(in_features, hidden_features, bias=bias[0])
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.norm = norm_layer(hidden_features) if norm_layer is not None else nn.Identity()
        self.fc2 = linear_layer(hidden_features, out_features, bias=bias[1])
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.norm(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x
    
class Attention(nn.Module):
    fused_attn: torch.jit.Final[bool]

    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            attn_drop: float = 0.,
            proj_drop: float = 0.,
            norm_layer: nn.Module = nn.LayerNorm,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        # self.fused_attn = use_fused_attn()

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        # if self.fused_attn:
        #     x = F.scaled_dot_product_attention(
        #         q, k, v,
        #         dropout_p=self.attn_drop.p if self.training else 0.,
        #     )
        if False:
            pass
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    
def get_positional_embed(
    x: Float[Tensor, "B N"], 
    y: Float[Tensor, "B N"], 
    z: Float[Tensor, "B N"],
    condition_dim: int,
    device: torch.device,
) -> Float[Tensor, "B N CD"]:
    assert condition_dim % 3 == 0
    emb_x = _get_1d_pos_emb(x, condition_dim // 3) # [B, N, CD/3]
    emb_y = _get_1d_pos_emb(y, condition_dim // 3) # [B, N, CD/3]
    emb_z = _get_1d_pos_emb(z, condition_dim // 3) # [B, N, CD/3]
    
    emb = torch.cat([emb_x, emb_y, emb_z], dim=-1) # [B, N, CD]
    return emb

def _get_1d_pos_emb(pos: Float[Tensor, "B N"], dim: int) -> Float[Tensor, "B N D"]:
    assert dim % 2 == 0
    omega = torch.arange(0, dim // 2, dtype=torch.float32, device=pos.device) # [D/2]
    omega /= dim // 2 # [D/2]
    omega = 1. / 10000**omega # [D/2]

    out = pos.unsqueeze(-1) @ omega.unsqueeze(0) # [B, N, D/2]
    emb_sin = torch.sin(out) # [B, N, D/2]
    emb_cos = torch.cos(out) # [B, N, D/2]
    
    emb = torch.cat([emb_sin, emb_cos], dim=-1) # [B, N, D]
    return emb
    
    
