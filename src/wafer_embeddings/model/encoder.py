"""Per-die ViT encoder (patch=1): multi-scheme tokens -> 384-d embedding.

A patch=1 per-die transformer is a Set Transformer over dies (SPECS.md §4). Position
comes from center/radius-normalized coordinates via fixed Fourier features + a learned
projection (generalizes to unseen grid sizes, §5); die state(s) come from a learned
embedding table per binning scheme, fused by sum (§5). Readout is masked-mean or PMA,
then L2-normalized. Output dim fixed at 384 (§3 N1).
"""

from __future__ import annotations

from typing import cast

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint

from wafer_embeddings.model.attention import ISAB, PMA, SAB
from wafer_embeddings.tokenize.tokenizer import N_STATES


class CoordPositionalEncoding(nn.Module):
    """Fourier features of (u, v) + raw (u, v, r), projected to ``dim``.

    Fixed sinusoidal bands (no learned frequencies) so the encoding extends to grid
    sizes / positions unseen in training (SPECS.md §5).
    """

    def __init__(self, dim: int, n_bands: int = 6):
        super().__init__()
        freqs = torch.pi * (2.0 ** torch.arange(n_bands))  # (n_bands,)
        self.register_buffer("freqs", freqs, persistent=False)
        in_dim = 4 * n_bands + 3  # sin/cos of u and v per band, plus raw (u,v,r)
        self.proj = nn.Linear(in_dim, dim)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        u = coords[..., 0:1]
        v = coords[..., 1:2]
        freqs = cast(torch.Tensor, self.freqs)
        f = freqs.view(*([1] * (coords.dim() - 1)), -1)  # broadcast
        feats = torch.cat(
            [
                torch.sin(u * f),
                torch.cos(u * f),
                torch.sin(v * f),
                torch.cos(v * f),
                coords,
            ],
            dim=-1,
        )
        return self.proj(feats)


class MultiSchemeEmbedding(nn.Module):
    """One embedding table per binning scheme, fused by sum (SPECS.md §5).

    ``vocab_sizes`` lists each scheme's cardinality; a trailing "scheme-absent" index
    (= vocab) supports missing schemes / modality dropout. For MixedWM38 there is one
    scheme (pass/fail), the degenerate case.
    """

    def __init__(self, dim: int, vocab_sizes: tuple[int, ...] = (N_STATES,)):
        super().__init__()
        self.vocab_sizes = vocab_sizes
        self.tables = nn.ModuleList(
            nn.Embedding(v + 1, dim) for v in vocab_sizes  # +1 = scheme-absent
        )

    def forward(self, state_ids: torch.Tensor) -> torch.Tensor:
        # Accept (B, L) single-scheme or (B, L, S) multi-scheme.
        if state_ids.dim() == 2:
            state_ids = state_ids.unsqueeze(-1)
        out: torch.Tensor = self.tables[0](state_ids[..., 0])
        for s in range(1, len(self.tables)):
            out = out + self.tables[s](state_ids[..., s])
        return out


class PerDieViT(nn.Module):
    def __init__(
        self,
        embed_dim: int = 384,
        width: int = 384,
        depth: int = 12,
        n_heads: int = 6,
        attention: str = "full",  # "full" (SAB) | "isab"
        n_inducing: int = 32,
        readout: str = "mean",  # "mean" | "pma"
        vocab_sizes: tuple[int, ...] = (N_STATES,),
        n_bands: int = 6,
        grad_checkpoint: bool = False,
        pre_norm: bool = False,
    ):
        super().__init__()
        if attention not in ("full", "isab"):
            raise ValueError(f"attention must be 'full' or 'isab', got {attention}")
        if readout not in ("mean", "pma"):
            raise ValueError(f"readout must be 'mean' or 'pma', got {readout}")
        self.attention = attention
        self.readout = readout
        self.grad_checkpoint = grad_checkpoint
        self.state_emb = MultiSchemeEmbedding(width, vocab_sizes)
        self.pos_enc = CoordPositionalEncoding(width, n_bands)
        self.blocks = nn.ModuleList(
            (
                SAB(width, n_heads, pre_norm=pre_norm)
                if attention == "full"
                else ISAB(width, n_heads, n_inducing=n_inducing, pre_norm=pre_norm)
            )
            for _ in range(depth)
        )
        # pre-LN blocks leave the residual stream un-normalized -> final LayerNorm (as ViT)
        self.final_norm = nn.LayerNorm(width) if pre_norm else None
        self.pma = PMA(width, n_heads, pre_norm=pre_norm) if readout == "pma" else None
        self.proj = nn.Linear(width, embed_dim)

    def forward(
        self,
        coords: torch.Tensor,  # (B, L, 3)
        state_ids: torch.Tensor,  # (B, L) or (B, L, S)
        mask: torch.Tensor,  # (B, L) bool, True = valid
    ) -> torch.Tensor:
        x = self.state_emb(state_ids) + self.pos_enc(coords)
        # Zero out padded tokens so they never leak through residual/FFN paths.
        x = x * mask.unsqueeze(-1)
        for blk in self.blocks:
            if self.grad_checkpoint and self.training:
                x = torch.utils.checkpoint.checkpoint(blk, x, mask, use_reentrant=False)
            else:
                x = blk(x, mask)
            x = x * mask.unsqueeze(-1)
        if self.final_norm is not None:
            x = self.final_norm(x) * mask.unsqueeze(-1)
        if self.readout == "mean":
            denom = mask.sum(dim=1, keepdim=True).clamp(min=1)
            pooled = (x * mask.unsqueeze(-1)).sum(dim=1) / denom
        else:
            assert self.pma is not None  # guaranteed when readout == "pma"
            pooled = self.pma(x, mask)
        return F.normalize(self.proj(pooled), dim=-1)
