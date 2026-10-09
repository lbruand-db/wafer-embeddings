"""Load a trained encoder checkpoint and embed raw wafer maps (SPECS.md §10, §11.4).

One code path for every consumer: the MLflow pyfunc (real-time serving), the batch
embedding job, and evaluation. Pure torch/numpy (no MLflow), so it is unit-tested in CI.

A checkpoint is what the training job saves: ``{"state_dict": ..., "config": vars(args)}``
for the **student** encoder; ``config`` holds the architecture and the token cap.
"""

from __future__ import annotations

import base64
from typing import Any

import numpy as np

from wafer_embeddings.tokenize.tokenizer import cap_tokens, collate, tokenize

# Architecture keys read from the training config (train job argparse names).
_ARCH_DEFAULTS: dict[str, Any] = {
    "embed_dim": 384,
    "depth": 12,
    "heads": 6,
    "attention": "isab",
    "pre_norm": False,
    "max_tokens": 4096,
}


def build_encoder(config: dict):
    """A ``PerDieViT`` with the architecture recorded in a training config."""
    from wafer_embeddings.model.encoder import PerDieViT

    c = _ARCH_DEFAULTS | {k: config[k] for k in _ARCH_DEFAULTS if k in config}
    return PerDieViT(
        embed_dim=int(c["embed_dim"]),
        width=int(c["embed_dim"]),
        depth=int(c["depth"]),
        n_heads=int(c["heads"]),
        attention=str(c["attention"]),
        pre_norm=bool(c["pre_norm"]),
    )


class WaferEncoder:
    """Embed raw wafer maps with a trained encoder -> (N, D) float32, L2-normalized.

    Each map gets its **own** seeded RNG for the token cap, so a wafer's embedding does
    not depend on which other wafers share its batch (serving and batch jobs agree).
    """

    def __init__(self, state_dict: dict, config: dict, device: str = "cpu", seed: int = 0):
        self.config = dict(config)
        self.max_tokens = int(self.config.get("max_tokens") or 4096)
        self.seed = seed
        self.device = device
        self.model = build_encoder(self.config)
        self.model.load_state_dict(state_dict)
        self.model.to(device).eval()
        self.dim = int(self.config.get("embed_dim") or 384)
        for p in self.model.parameters():  # inference-only, without touching global grad mode
            p.requires_grad_(False)

    @property
    def n_params(self) -> int:
        """Total parameter count of the served encoder (frozen at inference, so all)."""
        from wafer_embeddings.model.encoder import count_parameters

        return count_parameters(self.model)

    @classmethod
    def from_checkpoint(cls, path: str, device: str = "cpu") -> WaferEncoder:
        import torch

        ck = torch.load(path, map_location=device, weights_only=False)
        return cls(ck["state_dict"], ck["config"], device=device)

    def tokenize_map(self, wafer_map: np.ndarray):
        tw = tokenize(np.asarray(wafer_map, dtype=np.uint8))
        return cap_tokens(tw, self.max_tokens, np.random.default_rng(self.seed))

    def embed_maps(self, maps: list[np.ndarray], batch_size: int = 128) -> np.ndarray:
        import torch

        out = []
        for i in range(0, len(maps), batch_size):
            coords, states, mask = collate([self.tokenize_map(m) for m in maps[i : i + batch_size]])
            with torch.no_grad():
                emb = self.model(
                    coords.to(self.device), states.to(self.device), mask.to(self.device)
                )
            out.append(emb.cpu().numpy().astype(np.float32))
        return np.concatenate(out) if out else np.zeros((0, self.dim), np.float32)

    def embed_rows(self, heights, widths, flat_maps, batch_size: int = 128) -> np.ndarray:
        """Embed rows stored as (height, width, flattened map) — the Delta/parquet layout."""
        maps = [
            np.asarray(f, dtype=np.uint8).reshape(int(h), int(w))
            for h, w, f in zip(heights, widths, flat_maps)
        ]
        return self.embed_maps(maps, batch_size=batch_size)


def encode_map(wafer_map: np.ndarray) -> str:
    """Pack a {0,1,2} wafer map into a compact base64 string (2 bits per cell).

    Format: ``"<height>x<width>:<base64>"``. ~16x smaller than a JSON int list, so the
    search table can carry the map for display without bloating Lakebase.
    """
    m = np.asarray(wafer_map, dtype=np.uint8)
    if m.ndim != 2 or m.size == 0 or int(np.max(m)) > 3:
        raise ValueError("wafer map must be a non-empty 2-D array with cells in {0..3}")
    flat = m.reshape(-1)
    pad = (-len(flat)) % 4
    q = np.concatenate([flat, np.zeros(pad, np.uint8)]).reshape(-1, 4)
    packed = (q[:, 0] | (q[:, 1] << 2) | (q[:, 2] << 4) | (q[:, 3] << 6)).astype(np.uint8)
    return f"{m.shape[0]}x{m.shape[1]}:" + base64.b64encode(packed.tobytes()).decode("ascii")


def decode_map(s: str) -> np.ndarray:
    """Inverse of :func:`encode_map`."""
    shape, b64 = s.split(":", 1)
    h, w = (int(x) for x in shape.split("x"))
    packed = np.frombuffer(base64.b64decode(b64), dtype=np.uint8)
    cells = np.stack([(packed >> s) & 3 for s in (0, 2, 4, 6)], axis=1).reshape(-1)
    return cells[: h * w].reshape(h, w).astype(np.uint8)
