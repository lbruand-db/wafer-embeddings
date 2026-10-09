"""Retrieval, clustering, and embedding-health metrics (SPECS.md §8).

Retrieval relevance for the multi-label defect sets is Jaccard overlap >= tau (or
exact set match); clustering uses ARI/NMI/silhouette; health uses effective rank
(dimensional collapse, ref [25]) and alignment/uniformity (ref [24]).
"""

from __future__ import annotations

import numpy as np

from wafer_embeddings.eval.search import topk_cosine


# ----- retrieval ----------------------------------------------------------------
def pairwise_relevant(
    query_labels: np.ndarray,
    db_labels: np.ndarray,
    mode: str = "jaccard",
    tau: float = 0.5,
) -> np.ndarray:
    """(Q, N) boolean relevance from multi-hot defect labels.

    Both-empty label sets (Normal) count as identical (jaccard 1).
    """
    q = np.asarray(query_labels, dtype=np.float64)
    d = np.asarray(db_labels, dtype=np.float64)
    inter = q @ d.T
    union = q.sum(1)[:, None] + d.sum(1)[None, :] - inter
    with np.errstate(divide="ignore", invalid="ignore"):
        jac = np.where(union > 0, inter / np.maximum(union, 1e-12), 1.0)
    if mode == "exact":
        return np.isclose(jac, 1.0)
    if mode == "jaccard":
        return jac >= tau
    raise ValueError(f"mode must be 'jaccard' or 'exact', got {mode}")


def _ap_at_k(rel: np.ndarray, n_rel: int, k: int) -> float:
    if n_rel == 0:
        return 0.0
    rel_k = rel[:k].astype(np.float64)  # may be shorter than k if corpus < k
    prec_at_i = np.cumsum(rel_k) / (np.arange(len(rel_k)) + 1)
    return float((prec_at_i * rel_k).sum() / min(n_rel, k))


def _ndcg_at_k(rel: np.ndarray, n_rel: int, k: int) -> float:
    if n_rel == 0:
        return 0.0
    rel_k = rel[:k].astype(np.float64)  # may be shorter than k if corpus < k
    dcg = (rel_k / np.log2(np.arange(2, len(rel_k) + 2))).sum()
    idcg = (1.0 / np.log2(np.arange(2, min(n_rel, k) + 2))).sum()
    return float(dcg / idcg) if idcg > 0 else 0.0


def retrieval_metrics(
    query_emb: np.ndarray,
    db_emb: np.ndarray,
    query_labels: np.ndarray,
    db_labels: np.ndarray,
    ks: tuple[int, ...] = (1, 5, 10),
    mode: str = "jaccard",
    tau: float = 0.5,
    exclude_diagonal: bool = True,
) -> dict[str, float]:
    """Mean precision/recall/mAP/nDCG@k and MRR over queries (SPECS.md §8.1)."""
    kmax = min(max(ks), db_emb.shape[0])
    idx, _ = topk_cosine(query_emb, db_emb, kmax, exclude_diagonal)
    rel_full = pairwise_relevant(query_labels, db_labels, mode, tau)
    if exclude_diagonal:
        n = min(rel_full.shape)
        rel_full[np.arange(n), np.arange(n)] = False
    n_rel = rel_full.sum(axis=1)
    ranked = np.take_along_axis(rel_full, idx, axis=1)  # (Q, kmax)

    out: dict[str, float] = {}
    for k in ks:
        rk = ranked[:, :k]
        out[f"precision@{k}"] = float(rk.mean(axis=1).mean())
        with np.errstate(divide="ignore", invalid="ignore"):
            rec = np.where(n_rel > 0, rk.sum(axis=1) / np.maximum(n_rel, 1), 0.0)
        out[f"recall@{k}"] = float(rec.mean())
        out[f"map@{k}"] = float(
            np.mean([_ap_at_k(ranked[i], int(n_rel[i]), k) for i in range(ranked.shape[0])])
        )
        out[f"ndcg@{k}"] = float(
            np.mean([_ndcg_at_k(ranked[i], int(n_rel[i]), k) for i in range(ranked.shape[0])])
        )
    first = np.argmax(ranked, axis=1)
    has = ranked.any(axis=1)
    rr = np.where(has, 1.0 / (first + 1), 0.0)
    out["mrr"] = float(rr.mean())
    return out


# ----- clustering ---------------------------------------------------------------
def clustering_metrics(
    labels_true: np.ndarray, cluster_labels: np.ndarray, embeddings: np.ndarray | None = None
) -> dict[str, float]:
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

    out = {
        "ari": float(adjusted_rand_score(labels_true, cluster_labels)),
        "nmi": float(normalized_mutual_info_score(labels_true, cluster_labels)),
    }
    if embeddings is not None and len(np.unique(cluster_labels)) > 1:
        from sklearn.metrics import silhouette_score

        out["silhouette"] = float(silhouette_score(embeddings, cluster_labels))
    return out


# ----- embedding health ---------------------------------------------------------
def effective_rank(embeddings: np.ndarray, eps: float = 1e-12) -> float:
    """exp(entropy of the normalized covariance eigenspectrum) (ref [25]).

    ~1 under dimensional collapse, up to D for an isotropic spectrum.
    """
    x = embeddings - embeddings.mean(axis=0, keepdims=True)
    cov = (x.T @ x) / max(x.shape[0] - 1, 1)
    eig = np.clip(np.linalg.eigvalsh(cov), 0, None)
    total = eig.sum()
    if total <= eps:
        return 1.0
    p = eig / total
    p = p[p > eps]
    return float(np.exp(-(p * np.log(p)).sum()))


def rankme(embeddings: np.ndarray, eps: float = 1e-7) -> float:
    """RankMe: exp(entropy of the normalized singular values) (Garrido et al., ICML 2023).

    Label-free effective rank of an (N, D) embedding matrix, bounded by min(N, D). Unlike
    ``effective_rank`` it uses singular values, not covariance eigenvalues (their squares),
    so a few dominant directions don't swamp the entropy; RankMe was shown to track
    downstream quality across SSL runs. Not centered, as in the paper.
    """
    s = np.linalg.svd(np.asarray(embeddings, dtype=np.float64), compute_uv=False)
    total = s.sum()
    if total <= 0:
        return 1.0
    p = s / total + eps
    return float(np.exp(-(p * np.log(p)).sum()))


def alignment(z1: np.ndarray, z2: np.ndarray) -> float:
    """Mean squared distance between paired augmented views (ref [24]); lower = tighter."""
    return float(((z1 - z2) ** 2).sum(axis=1).mean())


def uniformity(z: np.ndarray, t: float = 2.0) -> float:
    """log E[exp(-t ||zi - zj||^2)] over distinct pairs (ref [24]); lower = more spread."""
    sq = np.sum((z[:, None, :] - z[None, :, :]) ** 2, axis=-1)
    iu = np.triu_indices(z.shape[0], k=1)
    return float(np.log(np.exp(-t * sq[iu]).mean()))
