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
