"""What does a trained encoder's embedding contain? Two label-free probes (CPU).

**A. Descriptor probe**: is the pixel-PCA descriptor (the strongest training-free bar) in
the embedding? A ridge regression from embedding to descriptor is fit on unlabeled train
maps and scored (R^2) on the val sample; a kNN-regression R^2 (each map's descriptor
predicted from its 10 nearest maps *in the embedding*) says whether the embedding's own
neighbourhoods carry it. The val retrieval metrics of the ridge-projected embedding say
whether a linear readout of the embedding would retrieve like the descriptor.

**B. Nuisance probe**: ``shape@10`` (neighbours with the same map shape, a die-grid /
device proxy) vs chance, ``label@10``, ``logdies_r2`` (``tools/nuisance_probe.py``).

Run: uv run --extra jobs python tools/probe_embedding.py <wafer_maps_parquet> \\
       --checkpoint a.pt b.pt --names seed-s0 nnpix-k5
"""

from __future__ import annotations

import argparse
import json

import numpy as np


def ridge_fit(x: np.ndarray, y: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    """Ridge weights (D+1, T) with an unpenalized bias, closed form."""
    xb = np.hstack([x, np.ones((len(x), 1))])
    reg = alpha * np.eye(xb.shape[1])
    reg[-1, -1] = 0.0
    return np.linalg.solve(xb.T @ xb + reg, xb.T @ y)


def ridge_predict(w: np.ndarray, x: np.ndarray) -> np.ndarray:
    return np.hstack([x, np.ones((len(x), 1))]) @ w


def r2(y: np.ndarray, pred: np.ndarray) -> float:
    """Variance-weighted R^2 over all target columns."""
    return 1.0 - float(((y - pred) ** 2).sum() / ((y - y.mean(axis=0)) ** 2).sum())


def knn_regress(emb: np.ndarray, y: np.ndarray, k: int = 10) -> np.ndarray:
    """Each row's target predicted as the mean of its k nearest other rows (cosine)."""
    sim = emb @ emb.T
    np.fill_diagonal(sim, -np.inf)
    nn = np.argsort(-sim, axis=1)[:, :k]
    return y[nn].mean(axis=1)


def _unit(x: np.ndarray) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)


def main(argv=None) -> None:  # pragma: no cover - needs checkpoints + the parquet export
    import pyarrow.dataset as ds
    import torch

    from nuisance_probe import nuisance_probe  # tools/ sibling (run from tools/)
    from wafer_embeddings.data.common import NO_DIE
    from wafer_embeddings.data.rows import rows_to_maps, shape_groups
    from wafer_embeddings.data.wm811k import CLASS_NAMES
    from wafer_embeddings.eval.baselines import PixelPCA, polar_fail_embedding
    from wafer_embeddings.jobs.train import load_eval_table, load_train_table
    from wafer_embeddings.serving import WaferEncoder, build_encoder
    from wafer_embeddings.train import g1_metrics

    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("parquet")
    p.add_argument("--checkpoint", nargs="+", required=True)
    p.add_argument("--names", nargs="+", required=True)
    p.add_argument("--fit-maps", type=int, default=5000, help="Unlabeled train maps for ridge.")
    p.add_argument("--dim", type=int, default=64, help="Pixel-PCA components (the target).")
    p.add_argument("--eval-cap", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args(argv)
    if len(a.names) != len(a.checkpoint):
        p.error("--names must match --checkpoint")
    torch.set_num_threads(6)
    dset = ds.dataset(a.parquet, format="parquet")

    fit_maps = rows_to_maps(load_train_table(dset, a.fit_maps, a.seed + 7).to_pylist())
    rows = load_eval_table(dset, a.eval_cap, a.seed).to_pylist()
    maps = rows_to_maps(rows)
    y = np.array([r["label_id"] for r in rows])
    s = np.array([r["split"] for r in rows], dtype=object)
    shapes = shape_groups(rows)
    logdies = np.log([max(int((m != NO_DIE).sum()), 1) for m in maps])

    encoders = {n: WaferEncoder.from_checkpoint(c) for n, c in zip(a.names, a.checkpoint)}
    ref = next(iter(encoders.values()))
    torch.manual_seed(0)
    encoders = {"untrained": WaferEncoder(build_encoder(ref.config).state_dict(), ref.config)} | (
        encoders
    )
    # descriptor target on the same capped tokens the encoders see
    fit_w = [ref.tokenize_map(m) for m in fit_maps]
    val_w = [ref.tokenize_map(m) for m in maps]
    pca = PixelPCA(a.dim).fit(fit_w)
    t_fit = pca.transform(fit_w, normalize=False)
    t_val = pca.transform(val_w, normalize=False)

    keys = ("xgroup_knn_macro_recall", "xgroup_precision@10", "map@10")
    chance = float(np.mean([(shapes == x).mean() for x in shapes]))
    out: dict = {"shape_chance@10": round(chance, 4)}

    def retrieval(emb):
        m = g1_metrics(emb, y, s, len(CLASS_NAMES), class_names=CLASS_NAMES, groups=shapes)
        return {k: round(m[k], 4) for k in keys}

    for name, emb in (("pixel_pca", _unit(t_val)), ("polar", polar_fail_embedding(val_w))):
        out[name] = {
            "knn_r2": round(r2(t_val, knn_regress(emb, t_val)), 3),
            **nuisance_probe(emb, y, shapes, logdies),
            **retrieval(emb),
        }
    for name, enc in encoders.items():
        e_fit, e_val = enc.embed_maps(fit_maps), enc.embed_maps(maps)
        w = ridge_fit(e_fit, t_fit, alpha=1e-2)
        proj = ridge_predict(w, e_val)
        out[name] = {
            "linear_r2": round(r2(t_val, proj), 3),
            "knn_r2": round(r2(t_val, knn_regress(e_val, t_val)), 3),
            **nuisance_probe(e_val, y, shapes, logdies),
            **retrieval(e_val),
            "ridge_projected": retrieval(_unit(proj)),
        }
    print(json.dumps(out, indent=1))


if __name__ == "__main__":  # pragma: no cover
    main()
