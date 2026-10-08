"""DINO pretraining + Gate-G1 eval on WM-811K (SPECS.md §4/§7/§8, PLAN P1).

Runs on AI Runtime serverless GPU (no Spark session), so it reads the parquet export
from the UC volume. Pretrains the per-die ViT with DINO (ISAB attention by default:
WM-811K maps vary widely in size, so dies are token-capped — SPECS.md §4), embeds the
labeled subset with exact search, logs loss / collapse / G1 metrics to MLflow, and
saves the encoder weights as an artifact. UC registration + serving are later (P3/P4).

The learning logic is the unit-tested ``train`` package; this file is the thin
Databricks/MLflow/GPU wiring.
"""

from __future__ import annotations

import argparse

import numpy as np

from wafer_embeddings.data.wm811k import CLASS_NAMES
from wafer_embeddings.obs import get_logger, stage
from wafer_embeddings.train import (
    embed_all,
    fit_dino,
    g1_metrics,
    stratified_indices,
    wafers_from_rows,
)

N_CLASSES = len(CLASS_NAMES)


def _args(argv=None):
    p = argparse.ArgumentParser(description="DINO pretrain + G1 eval (WM-811K).")
    p.add_argument("--catalog", required=True)
    p.add_argument("--schema", required=True)
    p.add_argument("--table", default="wafer_maps")
    p.add_argument("--volume", default="raw", help="UC volume holding wafer_maps_parquet.")
    p.add_argument("--experiment", default=None, help="MLflow experiment (default: AI Runtime's).")
    p.add_argument("--max-train", type=int, default=50000, help="Train maps to sample (0=all).")
    p.add_argument(
        "--eval-cap",
        type=int,
        default=5000,
        help="Labeled maps per eval side: val/test queries, and train maps for the kNN bank.",
    )
    p.add_argument(
        "--eval-sampling",
        default="balanced",
        choices=["balanced", "proportional"],
        help="Class-stratified eval sample: equal per class, or proportional to frequency.",
    )
    p.add_argument("--max-tokens", type=int, default=4096, help="Cap dies/wafer (GPU mem).")
    p.add_argument(
        "--grad-checkpoint",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Gradient-checkpoint encoder blocks (fits depth-12 in A10 memory).",
    )
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--embed-dim", type=int, default=384)
    p.add_argument("--depth", type=int, default=12)
    p.add_argument("--heads", type=int, default=6)
    p.add_argument("--attention", default="full", choices=["full", "isab"])
    p.add_argument("--out-dim", type=int, default=1024, help="DINO prototype count.")
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--weight-decay", type=float, default=0.0, help="AdamW weight decay (0 = off).")
    p.add_argument(
        "--freeze-last-frac",
        type=float,
        default=0.1,
        help="Freeze the DINO prototype layer for this fraction of steps (stabilizer).",
    )
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args(argv)


def load_train_table(dset, max_train: int, seed: int):
    """Seeded uniform random sample of ``max_train`` train-split maps (0 = all).

    Replaces "first N rows in scan order", which drew the whole training set from the
    first parquet file (one slice of lots). Train maps are mostly unlabeled, so the
    sample is uniform rather than class-stratified. Only the id column is read to pick
    the sample; wafer maps are fetched just for the chosen ids. Uses an RNG stream
    separate from the eval sample's, so changing one never shifts the other.
    """
    import pyarrow.dataset as ds

    train = ds.field("split") == "train"
    if not max_train:
        return dset.to_table(filter=train)
    ids = dset.to_table(columns=["id"], filter=train).column("id").to_numpy()
    if max_train < len(ids):
        ids = np.random.default_rng([seed, 1]).choice(ids, size=max_train, replace=False)
    return dset.to_table(filter=train & ds.field("id").isin(ids.tolist()))


def load_eval_table(dset, eval_cap: int, seed: int, balanced: bool = True):
    """Seeded, class-stratified labeled eval sample from a pyarrow dataset.

    Queries come from labeled val/test maps, the kNN bank from labeled train maps,
    ``eval_cap`` of each. Only the small id/label/split columns are read for the full
    labeled set; wafer maps are fetched just for the sampled ids. The dedicated RNG
    keeps the eval set identical across runs with the same seed, so runs are comparable.
    """
    import pyarrow.dataset as ds

    meta = dset.to_table(columns=["id", "label_id", "split"], filter=ds.field("label_id") >= 0)
    ids = meta.column("id").to_numpy()
    labels = meta.column("label_id").to_numpy()
    splits = np.array(meta.column("split").to_pylist(), dtype=object)
    rng = np.random.default_rng(seed)
    picked = []
    for side in (np.isin(splits, ["val", "test"]), splits == "train"):
        pos = stratified_indices(labels[side], eval_cap, rng, balanced=balanced)
        picked.append(ids[side][pos])
    want = np.concatenate(picked).tolist()
    return dset.to_table(filter=ds.field("id").isin(want))


def main(argv=None) -> None:  # pragma: no cover - needs Spark/MLflow/GPU
    import mlflow  # ty: ignore[unresolved-import]
    import pyarrow.dataset as ds
    import torch

    from wafer_embeddings.eval.metrics import effective_rank
    from wafer_embeddings.model.dino import DinoModel, DINOHead, DINOLoss
    from wafer_embeddings.model.encoder import PerDieViT

    a = _args(argv)
    log = get_logger("wafer_embeddings.train")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info(f"device={device} torch={torch.__version__}")
    rng = np.random.default_rng(a.seed)
    torch.manual_seed(a.seed)

    # AI Runtime serverless GPU has no Spark session; read the volume parquet export
    # with pyarrow (seeded samples, so we never materialize the whole corpus).
    pq = f"/Volumes/{a.catalog}/{a.schema}/{a.volume}/wafer_maps_parquet"
    dset = ds.dataset(pq, format="parquet")
    with stage(log, "load train split (seeded random sample)"):
        tbl = load_train_table(dset, a.max_train, a.seed)
        train_wafers, _, _ = wafers_from_rows(tbl.to_pylist(), max_tokens=a.max_tokens, rng=rng)
    with stage(log, f"load labeled eval ({a.eval_sampling} stratified sample)"):
        tbl = load_eval_table(dset, a.eval_cap, a.seed, balanced=a.eval_sampling == "balanced")
        eval_wafers, eval_labels, eval_splits = wafers_from_rows(
            tbl.to_pylist(), max_tokens=a.max_tokens, rng=rng
        )
    q = np.isin(eval_splits, ["val", "test"])
    mix = dict(zip(*(v.tolist() for v in np.unique(eval_labels[q], return_counts=True))))
    log.info(f"train={len(train_wafers)} eval_labeled={len(eval_wafers)} query class mix={mix}")

    encoder = PerDieViT(
        embed_dim=a.embed_dim,
        width=a.embed_dim,
        depth=a.depth,
        n_heads=a.heads,
        attention=a.attention,
        grad_checkpoint=a.grad_checkpoint,
    )
    dino = DinoModel(encoder, DINOHead(a.embed_dim, out_dim=a.out_dim)).to(device)
    loss_fn = DINOLoss(out_dim=a.out_dim).to(device)
    opt = torch.optim.AdamW(
        list(dino.student_enc.parameters()) + list(dino.student_head.parameters()),
        lr=a.lr,
        weight_decay=a.weight_decay,
    )

    if a.experiment:
        mlflow.set_experiment(a.experiment)

    def _eval(prefix: str) -> dict[str, float]:
        emb = embed_all(dino, eval_wafers, device=device)
        m = g1_metrics(emb, eval_labels, eval_splits, N_CLASSES, class_names=CLASS_NAMES)
        m["effective_rank"] = effective_rank(emb)
        return {f"{prefix}{k}": v for k, v in m.items()}

    with mlflow.start_run(run_name="dino-g1"):
        mlflow.log_params(vars(a) | {"device": device, "n_classes": N_CLASSES})
        # Same eval maps, freshly initialized encoder: the bar training must beat.
        with stage(log, "embed + eval untrained encoder (baseline)"):
            baseline = _eval("untrained_")
        mlflow.log_metrics(baseline)
        log.info(f"untrained baseline: {baseline}")

        with stage(log, "train DINO"):
            losses = fit_dino(
                dino,
                loss_fn,
                opt,
                train_wafers,
                steps=a.steps,
                batch_size=a.batch_size,
                rng=rng,
                device=device,
                freeze_last_frac=a.freeze_last_frac,
                log=log.info,
                log_every=50,
            )
        for i, lv in enumerate(losses):
            if i % 50 == 0:
                mlflow.log_metric("dino_loss", lv, step=i)

        with stage(log, "embed + eval (G1)"):
            metrics = _eval("")
        mlflow.log_metrics(metrics)
        log.info(f"G1 metrics: {metrics}")
        for k in ("knn_acc", "knn_macro_recall", "map@10", "cluster_nmi"):
            log.info(f"{k}: untrained={baseline.get('untrained_' + k)} trained={metrics.get(k)}")

        path = "/tmp/wafer_encoder.pt"
        torch.save({"state_dict": encoder.state_dict(), "config": vars(a)}, path)
        mlflow.log_artifact(path, artifact_path="encoder")

        gate = metrics.get("map@10", 0) >= 0.80 and metrics.get("cluster_nmi", 0) >= 0.85
        log.info(
            f"GATE G1 {'PASS' if gate else 'NOT MET'} "
            f"(map@10={metrics.get('map@10')}, nmi={metrics.get('cluster_nmi')})"
        )


if __name__ == "__main__":  # pragma: no cover
    main()
