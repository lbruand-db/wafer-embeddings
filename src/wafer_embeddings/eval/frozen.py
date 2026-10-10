"""Frozen off-the-shelf DINOv2 on rasterized wafer maps (SPECS.md §16 item 1, GAPS 1.4).

The question it answers: does a natural-image foundation model, used as-is, already
embed wafer maps well? Each wafer is drawn as an RGB image (no-die white, pass grey,
fail red, as in the app) on a fixed grid over the unit disk, so maps of any die-grid size
become the same image size, then embedded by DINOv2's CLS token. Weights come from
``torch.hub`` (``facebookresearch/dinov2``), so this needs internet access; the train job
only runs it with ``--dinov2-baseline``.
"""

from __future__ import annotations

import numpy as np

from wafer_embeddings.tokenize.tokenizer import STATE_FAIL, TokenizedWafer

# no-die, pass, fail (the app's palette, wafer_search.PALETTE), as 0-1 floats
PALETTE = np.array([[255, 255, 255], [205, 205, 205], [214, 39, 40]], dtype=np.float32) / 255
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def wafer_raster(w: TokenizedWafer, size: int = 56) -> np.ndarray:
    """(size, size) cell states over the unit disk: 0 no die, 1 pass, 2 fail.

    A cell holding any failing die is fail; a cell with dies but no fail is pass.
    """
    out = np.zeros((size, size), dtype=np.int64)
    if w.n_tokens == 0:
        return out
    c = np.clip(w.coords[:, :2].astype(np.float64), -1.0, 1.0)
    ij = np.minimum(((c + 1.0) / 2.0 * size).astype(np.int64), size - 1)
    out[ij[:, 1], ij[:, 0]] = 1
    fail = w.state_ids == STATE_FAIL
    out[ij[fail, 1], ij[fail, 0]] = 2
    return out


def wafer_image(w: TokenizedWafer, size: int = 224, cells: int = 56) -> np.ndarray:
    """(3, size, size) ImageNet-normalized float32 image of a wafer (nearest upsampling)."""
    rgb = PALETTE[wafer_raster(w, cells)]  # (cells, cells, 3)
    rep = size // cells
    rgb = np.repeat(np.repeat(rgb, rep, axis=0), rep, axis=1)
    return ((rgb - IMAGENET_MEAN) / IMAGENET_STD).transpose(2, 0, 1).astype(np.float32)


def dinov2_embedding(
    wafers: list[TokenizedWafer],
    device: str = "cpu",
    model: str = "dinov2_vits14",
    batch_size: int = 64,
) -> np.ndarray:  # pragma: no cover - downloads weights
    """L2-normalized frozen DINOv2 CLS embeddings (N, 384 for ViT-S/14) of the wafers."""
    import torch

    net = torch.hub.load("facebookresearch/dinov2", model).to(device).eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(wafers), batch_size):
            x = torch.from_numpy(np.stack([wafer_image(w) for w in wafers[i : i + batch_size]]))
            out.append(net(x.to(device)).float().cpu().numpy())
    z = np.concatenate(out)
    return z / (np.linalg.norm(z, axis=1, keepdims=True) + 1e-12)
