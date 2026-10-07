"""Attention blocks: full self-attention (SAB) and induced attention (ISAB).

ISAB (Set Transformer, SPECS.md ref [13]) reduces attention cost from O(N^2) to
O(N*m) with ``m`` inducing points — the mitigation for ~8k-token wafers (§4). All
blocks honor a per-token validity mask so padded tokens are ignored.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class MAB(nn.Module):
    """Multihead Attention Block: pre-attention queries Q attend to keys K."""

    def __init__(self, dim: int, n_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        self.mha = nn.MultiheadAttention(dim, n_heads, dropout=dropout, batch_first=True)
        self.ln0 = nn.LayerNorm(dim)
        self.ln1 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.ffn = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, key_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        # key_mask: True = valid. MultiheadAttention wants True = ignore.
        kpm = (~key_mask) if key_mask is not None else None
        attn, _ = self.mha(q, k, k, key_padding_mask=kpm, need_weights=False)
        h = self.ln0(q + attn)
        return self.ln1(h + self.ffn(h))


class SAB(nn.Module):
    """Self-Attention Block — full O(N^2) attention over the tokens."""

    def __init__(self, dim: int, n_heads: int, **kw):
        super().__init__()
        self.mab = MAB(dim, n_heads, **kw)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        return self.mab(x, x, mask)


class ISAB(nn.Module):
    """Induced Set-Attention Block — O(N*m) via ``m`` learned inducing points."""

    def __init__(self, dim: int, n_heads: int, n_inducing: int = 32, **kw):
        super().__init__()
        self.inducing = nn.Parameter(torch.randn(1, n_inducing, dim) * 0.02)
        self.mab0 = MAB(dim, n_heads, **kw)  # inducing points attend to tokens
        self.mab1 = MAB(dim, n_heads, **kw)  # tokens attend to inducing points

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        b = x.shape[0]
        ind = self.inducing.expand(b, -1, -1)
        h = self.mab0(ind, x, mask)  # (B, m, dim) — tokens masked here
        return self.mab1(x, h, None)  # (B, L, dim) — inducing points as keys


class PMA(nn.Module):
    """Pooling by Multihead Attention — one seed query pools the token set."""

    def __init__(self, dim: int, n_heads: int, **kw):
        super().__init__()
        self.seed = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        self.mab = MAB(dim, n_heads, **kw)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        b = x.shape[0]
        pooled = self.mab(self.seed.expand(b, -1, -1), x, mask)  # (B, 1, dim)
        return pooled.squeeze(1)
