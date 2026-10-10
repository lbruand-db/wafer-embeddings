"""Score a checkpoint on the TEST split, once, for Gate G1 (SPECS.md §8.7: test once).

Scores the trained encoder next to the untrained encoder (same architecture, seed 0) and
the training-free polar baseline on the same class-stratified test sample, and prints
the headline metrics as JSON (``jobs/register.py --metrics-json`` takes this output).

Run: uv run --extra jobs python tools/score_test.py <encoder.pt> <wafer_maps_parquet>
"""

from __future__ import annotations

import argparse
import json


def headline(metrics: dict) -> dict:
    """Headline G1 metrics + per-class kNN recall, rounded for the report."""
    from wafer_embeddings.jobs.train import HEADLINE_KEYS

    return {k: round(metrics[k], 4) for k in HEADLINE_KEYS if k in metrics} | {
        k: round(v, 3) for k, v in metrics.items() if k.startswith("knn_recall_")
    }


def main(argv=None) -> None:  # pragma: no cover - needs a checkpoint + the parquet export
    import numpy as np
    import pyarrow.dataset as ds
    import torch

    from wafer_embeddings.data.wm811k import CLASS_NAMES
    from wafer_embeddings.eval.baselines import polar_fail_embedding
    from wafer_embeddings.jobs.train import load_eval_table
    from wafer_embeddings.serving import WaferEncoder, build_encoder
    from wafer_embeddings.train import g1_metrics

    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("checkpoint")
    p.add_argument("parquet")
    p.add_argument("--eval-cap", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args(argv)
    enc = WaferEncoder.from_checkpoint(a.checkpoint)
    dset = ds.dataset(a.parquet, format="parquet")
    rows = load_eval_table(dset, a.eval_cap, a.seed, query_split="test").to_pylist()
    maps = [np.asarray(r["wafer_map"], np.uint8).reshape(r["height"], r["width"]) for r in rows]
    y = np.array([r["label_id"] for r in rows])
    s = np.array([r["split"] for r in rows], dtype=object)
    g = np.array([f"{r['height']}x{r['width']}" for r in rows], dtype=object)
    torch.manual_seed(0)
    untrained = WaferEncoder(build_encoder(enc.config).state_dict(), enc.config)
    embs = {
        "trained": enc.embed_maps(maps),
        "untrained": untrained.embed_maps(maps),
        "polar": polar_fail_embedding([enc.tokenize_map(m) for m in maps]),
    }
    out: dict = {"split": "test", "n_queries": int((s == "test").sum())}
    for name, e in embs.items():
        m = g1_metrics(e, y, s, len(CLASS_NAMES), class_names=CLASS_NAMES, groups=g)
        out[name] = headline(m)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":  # pragma: no cover
    main()
