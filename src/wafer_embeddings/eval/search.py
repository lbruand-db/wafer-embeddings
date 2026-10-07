"""Exact cosine search + weighted kNN (SPECS.md §8.1/§8.3).

Representation quality is measured with **exact** search (no ANN index); embeddings
are assumed L2-normalized, so cosine == dot product. ANN recall is a separate,
later concern (§8.6, Lakebase).
"""

from __future__ import annotations

import numpy as np


def topk_cosine(
    query: np.ndarray, db: np.ndarray, k: int, exclude_diagonal: bool = False
) -> tuple[np.ndarray, np.ndarray]:
    """Return (indices (Q,k), sims (Q,k)) of the top-k db rows per query.

    ``exclude_diagonal`` masks self-matches when ``query is db`` (query row i == db
    row i), as in leave-one-out retrieval over a single corpus.
    """
    sims = query @ db.T
    if exclude_diagonal:
        n = min(sims.shape)
        sims[np.arange(n), np.arange(n)] = -np.inf
    k = min(k, db.shape[0])
    part = np.argpartition(-sims, kth=k - 1, axis=1)[:, :k]
    rows = np.arange(sims.shape[0])[:, None]
    order = np.argsort(-sims[rows, part], axis=1)
    idx = part[rows, order]
    return idx, sims[rows, idx]


def weighted_knn_predict(
    train_emb: np.ndarray,
    train_y: np.ndarray,
    query_emb: np.ndarray,
    k: int = 20,
    tau: float = 0.07,
) -> np.ndarray:
    """DINO-style weighted kNN over single-label class ids (SPECS.md §8.3)."""
    sims = query_emb @ train_emb.T
    k = min(k, train_emb.shape[0])
    nn = np.argpartition(-sims, kth=k - 1, axis=1)[:, :k]
    rows = np.arange(sims.shape[0])[:, None]
    w = np.exp(sims[rows, nn] / tau)
    classes = np.unique(train_y)
    scores = np.zeros((query_emb.shape[0], classes.size))
    nn_y = train_y[nn]
    for ci, c in enumerate(classes):
        scores[:, ci] = (w * (nn_y == c)).sum(axis=1)
    return classes[scores.argmax(axis=1)]
