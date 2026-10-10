"""Training-free embedding baselines a learned encoder must beat (SPECS.md §8).

``polar_fail_embedding`` is a handcrafted defect descriptor: the FAIL rate of each cell in
a polar grid around the wafer center (radial rings x angular sectors), made
rotation-invariant by taking the magnitude of the FFT over the angular axis, plus the
overall fail fraction. It sees only where the failing dies are, never the map size, so
it is also the reference for the cross-device (``xgroup_*``) metrics.
"""

from __future__ import annotations

import numpy as np

from wafer_embeddings.tokenize.tokenizer import STATE_FAIL, TokenizedWafer


def polar_fail_features(w: TokenizedWafer, n_r: int = 6, n_a: int = 12) -> np.ndarray:
    """Rotation-invariant polar FAIL-density features of one wafer.

    Returns ``n_r * (n_a // 2 + 1) + 1`` values: per ring, |rFFT| over the ``n_a``
    angular sectors of the per-cell fail rate, then the overall fail fraction.
    """
    n_feat = n_r * (n_a // 2 + 1) + 1
    if w.n_tokens == 0:
        return np.zeros(n_feat)
    fail = (w.state_ids == STATE_FAIL).astype(np.float64)
    c = w.coords.astype(np.float64)  # float32 would round the clip below back to 1.0
    u, v = c[:, 0], c[:, 1]
    rb = np.minimum((np.clip(c[:, 2], 0.0, 1.0) * n_r).astype(np.int64), n_r - 1)
    ab = ((np.arctan2(v, u) + np.pi) / (2 * np.pi) * n_a).astype(np.int64) % n_a
    total = np.zeros((n_r, n_a))
    bad = np.zeros((n_r, n_a))
    np.add.at(total, (rb, ab), 1.0)
    np.add.at(bad, (rb, ab), fail)
    rate = bad / np.maximum(total, 1.0)  # empty cells -> rate 0
    rot_inv = np.abs(np.fft.rfft(rate, axis=1))  # a rotation is a circular shift in angle
    return np.concatenate([rot_inv.ravel(), [fail.mean()]])


def polar_fail_embedding(wafers: list[TokenizedWafer], **kw) -> np.ndarray:
    """Polar FAIL features for a set of wafers, z-scored over the set and L2-normalized.

    The z-scoring uses only the wafers passed in (no labels), so the result is an
    unsupervised (N, D) embedding comparable to an encoder's output.
    """
    f = np.stack([polar_fail_features(w, **kw) for w in wafers])
    f = (f - f.mean(axis=0)) / (f.std(axis=0) + 1e-6)
    return f / (np.linalg.norm(f, axis=1, keepdims=True) + 1e-12)


def pixel_grid_features(w: TokenizedWafer, size: int = 32) -> np.ndarray:
    """FAIL rate per cell of a ``size`` x ``size`` Cartesian grid over the unit disk.

    The "flattened pixel" view of a wafer (SPECS.md §16 item 24): a fixed-size raster of
    where the failing dies are, independent of the die-grid size. Not rotation-invariant.
    """
    if w.n_tokens == 0:
        return np.zeros(size * size)
    fail = (w.state_ids == STATE_FAIL).astype(np.float64)
    c = np.clip(w.coords[:, :2].astype(np.float64), -1.0, 1.0)
    ij = np.minimum(((c + 1.0) / 2.0 * size).astype(np.int64), size - 1)
    total = np.zeros((size, size))
    bad = np.zeros((size, size))
    np.add.at(total, (ij[:, 1], ij[:, 0]), 1.0)
    np.add.at(bad, (ij[:, 1], ij[:, 0]), fail)
    return (bad / np.maximum(total, 1.0)).ravel()


def pixel_pca_embedding(wafers: list[TokenizedWafer], dim: int = 64, size: int = 32) -> np.ndarray:
    """Flattened-pixel FAIL rasters reduced by PCA to ``dim``, L2-normalized (no training).

    PCA is fit on the wafers passed in (no labels), like the polar baseline's z-scoring,
    so the result is an unsupervised (N, min(dim, N, size**2)) embedding.
    """
    x = np.stack([pixel_grid_features(w, size) for w in wafers]).astype(np.float32)
    x -= x.mean(axis=0)
    # top components from the (size^2 x size^2) covariance: cheap for 100k+ wafers
    evals, evecs = np.linalg.eigh(x.T.astype(np.float64) @ x)
    top = evecs[:, np.argsort(evals)[::-1][: min(dim, len(wafers))]].astype(np.float32)
    z = x @ top
    return z / (np.linalg.norm(z, axis=1, keepdims=True) + 1e-12)
