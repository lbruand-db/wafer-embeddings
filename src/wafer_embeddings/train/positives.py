"""Descriptor-guided positives: label-free cross-wafer positives for DINO (GAPS 1.1).

DINO alone only learns invariance between views of *one* wafer. NNCLR-style training adds
positives from *other* wafers that are close in some space; here that space is the
training-free polar FAIL-density descriptor (``eval.baselines``), which already
retrieves defect patterns well across devices. Each training wafer gets its ``k``
nearest neighbours by descriptor (cosine), computed once before training; at each step a
random neighbour's global view joins the student views and must match the anchor's
teacher (``trainer.train_step``). No labels are used anywhere.
"""

from __future__ import annotations

import numpy as np


def descriptor_neighbours(
    features: np.ndarray, k: int, chunk: int = 4096, device: str = "cpu"
) -> np.ndarray:
    """(N, k) indices of each row's ``k`` nearest other rows by cosine similarity.

    ``features`` are L2-normalized rows (e.g. ``polar_fail_embedding``). Exact search in
    row chunks (N x chunk similarities at a time), on ``device`` via torch. A row is
    never its own neighbour.
    """
    import torch

    n = len(features)
    if not 1 <= k < n:
        raise ValueError(f"need 1 <= k < N, got k={k}, N={n}")
    x = torch.as_tensor(np.ascontiguousarray(features), dtype=torch.float32, device=device)
    out = np.empty((n, k), dtype=np.int64)
    for start in range(0, n, chunk):
        sim = x[start : start + chunk] @ x.T
        rows = torch.arange(sim.shape[0], device=device)
        sim[rows, rows + start] = -float("inf")  # exclude self
        out[start : start + chunk] = torch.topk(sim, k, dim=1).indices.cpu().numpy()
    return out


def sample_neighbours(
    neighbours: np.ndarray, idx: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """One random neighbour (of the ``k`` stored) per batch index."""
    pick = rng.integers(0, neighbours.shape[1], size=len(idx))
    return neighbours[idx, pick]
