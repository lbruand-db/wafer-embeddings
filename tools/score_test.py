"""Score an encoder on the TEST split, once, for Gate G1 (SPECS.md §8.7: test once).

Scores the trained encoder next to the untrained encoder (same architecture, seed 0) and
the training-free polar baseline on the same class-stratified test sample, and prints
the headline metrics as JSON.

- ``--model <catalog.schema.name> --version N``: scores the **registered** checkpoint and
  records the result on that version (``test_scored_at``, ``test_*``, ``test_polar_*``
  tags). A version that already has ``test_scored_at`` is refused (GAPS 6.6).
- ``--checkpoint <encoder.pt>``: scores a local checkpoint; the JSON can be passed to
  ``jobs/register.py --metrics-json``, which tags the new version the same way.

Run: uv run --extra jobs python tools/score_test.py <wafer_maps_parquet> \\
       --model mmf_mlops_demo_catalog.wafer_embeddings.wafer_encoder --version 3
"""

from __future__ import annotations

import argparse
import datetime
import json


def headline(metrics: dict) -> dict:
    """Headline G1 metrics + per-class kNN recall, rounded for the report."""
    from wafer_embeddings.jobs.train import HEADLINE_KEYS

    return {k: round(metrics[k], 4) for k in HEADLINE_KEYS if k in metrics} | {
        k: round(v, 3) for k, v in metrics.items() if k.startswith("knn_recall_")
    }


def _args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("parquet", help="wafer_maps parquet export (local copy).")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--checkpoint", help="Local encoder checkpoint (.pt).")
    src.add_argument("--version", help="UC model version to score and tag (needs --model).")
    p.add_argument("--model", help="UC model, catalog.schema.name.")
    p.add_argument("--eval-cap", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args(argv)
    if a.version and not a.model:
        p.error("--version needs --model")
    return a


def score(checkpoint: str, parquet: str, eval_cap: int, seed: int) -> dict:  # pragma: no cover
    import numpy as np
    import pyarrow.dataset as ds
    import torch

    from wafer_embeddings.data.wm811k import CLASS_NAMES
    from wafer_embeddings.eval.baselines import polar_fail_embedding
    from wafer_embeddings.jobs.train import load_eval_table
    from wafer_embeddings.serving import WaferEncoder, build_encoder
    from wafer_embeddings.train import g1_metrics

    enc = WaferEncoder.from_checkpoint(checkpoint)
    dset = ds.dataset(parquet, format="parquet")
    rows = load_eval_table(dset, eval_cap, seed, query_split="test").to_pylist()
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
    out: dict = {
        "split": "test",
        "scored_at": datetime.date.today().isoformat(),
        "n_queries": int((s == "test").sum()),
    }
    for name, e in embs.items():
        m = g1_metrics(e, y, s, len(CLASS_NAMES), class_names=CLASS_NAMES, groups=g)
        out[name] = headline(m)
    return out


def main(argv=None) -> None:  # pragma: no cover - needs a checkpoint + the parquet export
    a = _args(argv)
    if a.checkpoint:
        print(json.dumps(score(a.checkpoint, a.parquet, a.eval_cap, a.seed), indent=1))
        return

    import glob
    import os

    import mlflow  # ty: ignore[unresolved-import]

    from wafer_embeddings.jobs.register import check_unscored, test_tags
    from wafer_embeddings.serving.pyfunc import CHECKPOINT

    mlflow.set_tracking_uri("databricks")  # not a local ./mlflow.db
    mlflow.set_registry_uri("databricks-uc")
    client = mlflow.MlflowClient()
    check_unscored(a.model, a.version, client.get_model_version(a.model, a.version).tags)
    root = mlflow.artifacts.download_artifacts(f"models:/{a.model}/{a.version}")
    ckpts = glob.glob(os.path.join(root, "artifacts", "**", "*.pt"), recursive=True)
    if len(ckpts) != 1:
        raise SystemExit(f"expected one {CHECKPOINT} .pt in the model artifacts, got {ckpts}")
    out = score(ckpts[0], a.parquet, a.eval_cap, a.seed)
    for k, v in test_tags(out, out["scored_at"]).items():
        client.set_model_version_tag(a.model, a.version, k, v)
    print(json.dumps(out | {"model": a.model, "version": a.version}, indent=1))


if __name__ == "__main__":  # pragma: no cover
    main()
