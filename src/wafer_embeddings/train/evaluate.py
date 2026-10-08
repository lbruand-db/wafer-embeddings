"""Gate-G1 evaluation over computed embeddings (SPECS.md §8, §8.9).

All exact-search (no Lakebase). Uses the labeled subset only; unlabeled maps
(label_id < 0) are ignored for metrics.
"""

from __future__ import annotations

import numpy as np

from wafer_embeddings.eval.metrics import clustering_metrics, retrieval_metrics
from wafer_embeddings.eval.search import weighted_knn_predict


def _one_hot(ids: np.ndarray, n_classes: int) -> np.ndarray:
    oh = np.zeros((len(ids), n_classes), dtype=np.float64)
    oh[np.arange(len(ids)), ids] = 1.0
    return oh


def g1_metrics(
    embeddings: np.ndarray,
    label_ids: np.ndarray,
    splits: np.ndarray,
    n_classes: int,
    knn_k: int = 20,
) -> dict[str, float]:
    """kNN-probe accuracy, clustering ARI/NMI, and retrieval mAP@k on labeled data."""
    labeled = label_ids >= 0
    tr = labeled & (splits == "train")
    te = labeled & np.isin(splits, ["val", "test"])
    out: dict[str, float] = {
        "n_labeled_train": float(tr.sum()),
        "n_labeled_eval": float(te.sum()),
    }

    if tr.sum() > 0 and te.sum() > 0:
        pred = weighted_knn_predict(embeddings[tr], label_ids[tr], embeddings[te], k=knn_k)
        out["knn_acc"] = float((pred == label_ids[te]).mean())

    uniq = np.unique(label_ids[te]) if te.sum() else np.array([])
    if te.sum() > len(uniq) >= 2:
        from sklearn.cluster import KMeans

        k = min(n_classes, len(uniq))
        km = KMeans(n_clusters=k, n_init=10, random_state=0).fit(embeddings[te])
        for name, val in clustering_metrics(label_ids[te], km.labels_, embeddings[te]).items():
            out[f"cluster_{name}"] = val

    if te.sum() > 1:
        oh = _one_hot(label_ids[te], n_classes)
        r = retrieval_metrics(embeddings[te], embeddings[te], oh, oh, ks=(1, 10), mode="exact")
        out["map@10"] = r["map@10"]
        out["recall@10"] = r["recall@10"]
    return out
