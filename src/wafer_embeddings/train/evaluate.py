"""Gate-G1 evaluation over computed embeddings (SPECS.md §8, §8.9).

All exact-search (no Lakebase). Uses the labeled subset only; unlabeled maps
(label_id < 0) are ignored for metrics.
"""

from __future__ import annotations

import numpy as np

from wafer_embeddings.eval.metrics import clustering_metrics, retrieval_metrics
from wafer_embeddings.eval.search import weighted_knn_predict


def stratified_indices(
    label_ids: np.ndarray, n: int, rng: np.random.Generator, balanced: bool = True
) -> np.ndarray:
    """Seeded, class-stratified sample of up to ``n`` positions into ``label_ids``.

    Replaces "first n rows in scan order", whose class mix depends on file layout.
    ``balanced=True`` gives each class an equal quota, water-filled (a class with fewer
    items contributes all of them and the leftover is shared by the rest), so rare
    patterns are not swamped by the dominant class (SPECS.md §8). ``balanced=False``
    samples proportionally to class frequency. Returns sorted positions.
    """
    label_ids = np.asarray(label_ids)
    if n >= len(label_ids):
        return np.arange(len(label_ids))
    classes, counts = np.unique(label_ids, return_counts=True)
    quota = np.zeros(len(classes), dtype=np.int64)
    if balanced:
        remaining = n
        open_ = np.ones(len(classes), dtype=bool)
        while remaining > 0 and open_.any():
            share = max(remaining // int(open_.sum()), 1)
            for i in np.nonzero(open_)[0]:
                take = min(share, int(counts[i] - quota[i]), remaining)
                quota[i] += take
                remaining -= take
                if quota[i] == counts[i]:
                    open_[i] = False
                if remaining == 0:
                    break
    else:
        exact = counts * n / counts.sum()
        quota = np.floor(exact).astype(np.int64)
        for i in np.argsort(-(exact - quota))[: n - int(quota.sum())]:
            quota[i] += 1  # hand the rounding remainder to the largest fractions
    picks = [
        rng.choice(np.nonzero(label_ids == c)[0], size=int(q), replace=False)
        for c, q in zip(classes, quota)
        if q > 0
    ]
    return np.sort(np.concatenate(picks)) if picks else np.zeros(0, dtype=np.int64)


def selection_score(
    embeddings: np.ndarray,
    label_ids: np.ndarray,
    n_classes: int,
    groups: np.ndarray | None = None,
) -> float:
    """Checkpoint-selection score from labeled **train** maps only.

    Alternate items form a kNN bank and a query set; returns the queries' kNN macro
    recall. Choosing a checkpoint on the reported val/test queries would leak them into
    the G1 numbers, so the selection set is drawn from the train split instead. With
    ``groups``, the cross-group recall is used, so selection can't reward a model for
    recognizing the device instead of the defect.
    """
    half = np.where(np.arange(len(label_ids)) % 2 == 0, "train", "val").astype(object)
    m = g1_metrics(embeddings, np.asarray(label_ids), half, n_classes, groups=groups)
    return m.get("xgroup_knn_macro_recall" if groups is not None else "knn_macro_recall", 0.0)


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
    class_names: tuple[str, ...] | None = None,
    groups: np.ndarray | None = None,
    clustering: bool = True,
) -> dict[str, float]:
    """kNN-probe accuracy, clustering ARI/NMI, and retrieval mAP@k on labeled data.

    Also reports per-class kNN recall (``knn_recall_<class>``, SPECS.md §8 "report
    per-pattern") so rare-pattern failures aren't hidden by the averages.
    ``clustering=False`` skips the KMeans fit (the ``cluster_*`` metrics), the costliest
    part, for frequent in-training tracking.

    ``groups`` (e.g. map shape as a device proxy) adds ``xgroup_*`` metrics in which a
    query may not use same-group neighbours. Splits are per-map, so wafers from one
    lot/device sit on both sides and share labels far more often than chance; the
    cross-group numbers measure defect similarity without that shortcut.
    """
    labeled = label_ids >= 0
    tr = labeled & (splits == "train")
    te = labeled & np.isin(splits, ["val", "test"])
    out: dict[str, float] = {
        "n_labeled_train": float(tr.sum()),
        "n_labeled_eval": float(te.sum()),
    }

    if te.sum() > 0:
        # knn_acc is only meaningful against the majority-class rate of the same queries
        te_counts = np.bincount(label_ids[te])
        out["majority_baseline"] = float(np.max(te_counts)) / float(np.sum(te_counts))

    if tr.sum() > 0 and te.sum() > 0:
        pred = weighted_knn_predict(embeddings[tr], label_ids[tr], embeddings[te], k=knn_k)
        out["knn_acc"] = float((pred == label_ids[te]).mean())
        # balanced accuracy = mean per-class recall (imbalance-aware, SPECS.md §8.3)
        recalls = {}
        for c in np.unique(label_ids[te]):
            name = class_names[c] if class_names is not None else str(c)
            recalls[name] = float((pred[label_ids[te] == c] == c).mean())
            out[f"knn_recall_{name}"] = recalls[name]
        out["knn_macro_recall"] = float(np.mean(list(recalls.values())))

    uniq = np.unique(label_ids[te]) if te.sum() else np.array([])
    if clustering and te.sum() > len(uniq) >= 2:
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

    if groups is not None and tr.sum() > 0 and te.sum() > 1:
        groups = np.asarray(groups, dtype=object)
        yq, gq, eq = label_ids[te], groups[te], embeddings[te]
        excl = gq[:, None] == groups[tr][None, :]
        pred = weighted_knn_predict(embeddings[tr], label_ids[tr], eq, k=knn_k, exclude=excl)
        out["xgroup_knn_acc"] = float((pred == yq).mean())
        out["xgroup_knn_macro_recall"] = float(
            np.mean([(pred[yq == c] == c).mean() for c in np.unique(yq)])
        )
        # precision@10 among the queries, same-group items (and self) excluded
        sims = np.where(gq[:, None] == gq[None, :], -np.inf, eq @ eq.T)
        top = np.argsort(-sims, axis=1)[:, :10]
        valid = np.isfinite(np.take_along_axis(sims, top, axis=1))
        hit = (yq[top] == yq[:, None]) & valid
        out["xgroup_precision@10"] = float((hit.sum(1) / np.maximum(valid.sum(1), 1)).mean())
    return out
