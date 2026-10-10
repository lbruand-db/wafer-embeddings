"""Does DINO learn wafer-size / die-grid features instead of defect patterns? (CPU)

Trains briefly with the real ``fit_dino``, then compares untrained vs trained embeddings on:

- ``shape@10``: fraction of the 10 nearest neighbours with the same (height, width) map
  shape (a proxy for "same device / die grid"), the nuisance signal;
- ``label@10``: fraction with the same defect class, the signal we want;
- ``logdies_r2``: kNN-regression R^2 of log(#dies) from the neighbours.

Knobs tried against it (PLAN.md P1: neither was the lever): fewer Fourier bands (can't
resolve the die pitch) and per-view coordinate jitter of about one die pitch.

Run: uv run --extra jobs python tools/nuisance_probe.py <wafer_maps_parquet> --n-bands 6
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np


def nuisance_probe(emb, labels, shapes, logdies, k: int = 10) -> dict:
    """Shape / label agreement and log(#dies) R^2 among each embedding's k nearest."""
    sim = emb @ emb.T
    np.fill_diagonal(sim, -np.inf)
    nn = np.argsort(-sim, axis=1)[:, :k]
    pred = logdies[nn].mean(axis=1)
    r2 = 1 - float(((logdies - pred) ** 2).sum() / ((logdies - logdies.mean()) ** 2).sum())
    return {
        "shape@10": round(float((shapes[nn] == shapes[:, None]).mean()), 4),
        "label@10": round(float((labels[nn] == labels[:, None]).mean()), 4),
        "logdies_r2": round(r2, 4),
    }


def jittered_view(jitter: float):
    """``random_view`` plus a uniform coordinate jitter of ``jitter`` die pitches."""
    from wafer_embeddings.tokenize.augment import random_view
    from wafer_embeddings.tokenize.tokenizer import TokenizedWafer

    def view(w, rng, **kw):
        v = random_view(w, rng, **kw)
        if jitter <= 0 or v.n_tokens == 0:
            return v
        pitch = np.sqrt(np.pi / v.n_tokens)  # ~die spacing in the unit disk
        c = v.coords.copy()
        c[:, :2] += rng.uniform(-jitter * pitch, jitter * pitch, size=(len(c), 2))
        c[:, 2] = np.sqrt(c[:, 0] ** 2 + c[:, 1] ** 2)
        return TokenizedWafer(c.astype(np.float32), v.state_ids)

    return view


def main(argv=None) -> None:  # pragma: no cover - CPU training run
    import pyarrow.dataset as ds
    import torch

    import wafer_embeddings.train.trainer as trainer_mod
    from wafer_embeddings.data.common import NO_DIE
    from wafer_embeddings.data.wm811k import CLASS_NAMES
    from wafer_embeddings.jobs.train import load_eval_table, load_train_table
    from wafer_embeddings.model.dino import DinoModel, DINOHead, DINOLoss
    from wafer_embeddings.model.encoder import PerDieViT
    from wafer_embeddings.train import embed_all, fit_dino, g1_metrics, wafers_from_rows

    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("parquet")
    p.add_argument("--n-bands", type=int, default=6)
    p.add_argument("--jitter", type=float, default=0.0, help="Coord jitter, in die pitches.")
    p.add_argument("--steps", type=int, default=400)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--max-train", type=int, default=8000)
    p.add_argument("--eval-cap", type=int, default=1000)
    p.add_argument("--max-tokens", type=int, default=512)
    p.add_argument("--depth", type=int, default=6)
    p.add_argument("--width", type=int, default=128)
    p.add_argument("--threads", type=int, default=3)
    p.add_argument("--aug", default="{}", help="JSON of extra random_view knobs.")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args(argv)
    torch.set_num_threads(a.threads)
    rng = np.random.default_rng(a.seed)
    torch.manual_seed(a.seed)
    trainer_mod.random_view = jittered_view(a.jitter)  # build_views looks it up at call time

    dset = ds.dataset(a.parquet, format="parquet")
    train_rows = load_train_table(dset, a.max_train, a.seed).to_pylist()
    train_w, _, _ = wafers_from_rows(train_rows, max_tokens=a.max_tokens, rng=rng)
    rows = load_eval_table(dset, a.eval_cap, a.seed).to_pylist()  # val queries (protocol)
    eval_w, eval_y, eval_s = wafers_from_rows(rows, max_tokens=a.max_tokens, rng=rng)
    shapes = np.array([f"{r['height']}x{r['width']}" for r in rows], dtype=object)
    logdies = np.log([max(sum(1 for c in r["wafer_map"] if c != NO_DIE), 1) for r in rows])

    enc = PerDieViT(
        embed_dim=a.width,
        width=a.width,
        depth=a.depth,
        n_heads=4,
        attention="isab",
        n_bands=a.n_bands,
    )
    dino = DinoModel(enc, DINOHead(a.width, out_dim=1024))
    params = list(dino.student_enc.parameters()) + list(dino.student_head.parameters())
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0)

    def report() -> dict:
        emb = embed_all(dino, eval_w)
        m = g1_metrics(
            emb, eval_y, eval_s, len(CLASS_NAMES), class_names=CLASS_NAMES, groups=shapes
        )
        keys = ("knn_macro_recall", "map@10", "xgroup_knn_macro_recall", "xgroup_precision@10")
        return nuisance_probe(emb, eval_y, shapes, logdies) | {k: round(m[k], 4) for k in keys}

    t0 = time.time()
    before = report()
    fit_dino(
        dino,
        DINOLoss(out_dim=1024),
        opt,
        train_w,
        steps=a.steps,
        batch_size=16,
        rng=rng,
        aug=json.loads(a.aug),
        log=lambda s: print(s, flush=True),
        log_every=50,
    )
    chance = float(np.mean([(shapes == s).mean() for s in shapes]))
    out = {
        "n_bands": a.n_bands,
        "jitter": a.jitter,
        "aug": a.aug,
        "secs": round(time.time() - t0),
        "shape_chance": round(chance, 4),
        "untrained": before,
        "trained": report(),
    }
    print(json.dumps(out), flush=True)


if __name__ == "__main__":  # pragma: no cover
    main()
