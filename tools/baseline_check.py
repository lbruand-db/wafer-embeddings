"""Two no-training checks on the eval sample (CPU, seconds; PLAN.md P1 "The eval was leaky").

1. **Shape leak**: how often two maps with the same (height, width) share a label, vs any
   two maps, over the query-vs-bank pairs a kNN probe actually uses. If same shape ->
   same label is common, an encoder that clusters by map shape gets class "accuracy" for
   free. A shape one-hot "encoder" is scored with the G1 metrics to show it.
2. **Handcrafted bar**: the training-free polar FAIL-density descriptor
   (``eval.baselines.polar_fail_embedding``), scored with the same G1 metrics.

Run: uv run --extra jobs python tools/baseline_check.py <wafer_maps_parquet> [--eval-cap 1000]
"""

from __future__ import annotations

import argparse
import json

import numpy as np

KEYS = (
    "knn_acc",
    "knn_macro_recall",
    "map@10",
    "cluster_nmi",
    "xgroup_knn_acc",
    "xgroup_knn_macro_recall",
    "xgroup_precision@10",
)


def shape_leak(shapes, labels, splits, lots) -> dict:
    """Label agreement given same map shape, over query (val/test) x bank (train) pairs."""
    shapes, labels = np.asarray(shapes, dtype=object), np.asarray(labels)
    splits, lots = np.asarray(splits, dtype=object), np.asarray(lots, dtype=object)
    same_shape = shapes[:, None] == shapes[None, :]
    same_label = labels[:, None] == labels[None, :]
    cross = np.isin(splits, ["val", "test"])[:, None] & (splits == "train")[None, :]
    known = (lots[:, None] != None) & (lots[None, :] != None)  # noqa: E711 - elementwise
    return {
        "p_same_label": round(float(same_label[cross].mean()), 4),
        "p_same_label_given_same_shape": round(float(same_label[cross & same_shape].mean()), 4),
        "frac_pairs_same_shape": round(float(same_shape[cross].mean()), 4),
        "pairs_same_lot": int((cross & known & (lots[:, None] == lots[None, :])).sum()),
    }


def shape_onehot(shapes) -> np.ndarray:
    """A one-hot of the map shape: the pure nuisance "encoder"."""
    uniq = {k: i for i, k in enumerate(sorted(set(shapes)))}
    oh = np.zeros((len(shapes), len(uniq)))
    oh[np.arange(len(shapes)), [uniq[k] for k in shapes]] = 1.0
    return oh


def main(argv=None) -> None:  # pragma: no cover - needs the parquet export
    import pyarrow.dataset as ds

    from wafer_embeddings.data.wm811k import CLASS_NAMES
    from wafer_embeddings.eval.baselines import polar_fail_embedding
    from wafer_embeddings.jobs.train import load_eval_table
    from wafer_embeddings.train import g1_metrics, wafers_from_rows

    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("parquet")
    p.add_argument("--eval-cap", type=int, default=1000)
    p.add_argument("--split", default="val", choices=["val", "test"])
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args(argv)
    dset = ds.dataset(a.parquet, format="parquet")
    rows = load_eval_table(dset, a.eval_cap, a.seed, query_split=a.split).to_pylist()
    wafers, y, s = wafers_from_rows(rows, max_tokens=512, rng=np.random.default_rng(a.seed))
    shapes = np.array([f"{r['height']}x{r['width']}" for r in rows], dtype=object)
    m_shape = g1_metrics(shape_onehot(shapes), y, s, len(CLASS_NAMES), groups=shapes)
    m_polar = g1_metrics(
        polar_fail_embedding(wafers), y, s, len(CLASS_NAMES), class_names=CLASS_NAMES, groups=shapes
    )
    out = {
        "leak": shape_leak(shapes, y, s, [r.get("lot") for r in rows]),
        "shape_onehot": {k: round(m_shape[k], 4) for k in KEYS},
        "polar": {k: round(m_polar[k], 4) for k in KEYS}
        | {k: round(v, 3) for k, v in m_polar.items() if k.startswith("knn_recall_")},
    }
    print(json.dumps(out, indent=1))


if __name__ == "__main__":  # pragma: no cover
    main()
