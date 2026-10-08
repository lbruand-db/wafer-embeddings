"""DINO pretraining + Gate-G1 eval on WM-811K (SPECS.md §4/§7/§8, PLAN P1).

Runs on AI Runtime serverless GPU. Loads the ingested Delta table, pretrains the
per-die ViT with DINO (full attention for the ~2k-token WM-811K maps), embeds the
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
from wafer_embeddings.train import embed_all, fit_dino, g1_metrics, wafers_from_rows

N_CLASSES = len(CLASS_NAMES)


def _args(argv=None):
    p = argparse.ArgumentParser(description="DINO pretrain + G1 eval (WM-811K).")
    p.add_argument("--catalog", required=True)
    p.add_argument("--schema", required=True)
    p.add_argument("--table", default="wafer_maps")
    p.add_argument("--experiment", default=None, help="MLflow experiment (default: AI Runtime's).")
    p.add_argument("--max-train", type=int, default=50000, help="Train maps to sample (0=all).")
    p.add_argument("--eval-cap", type=int, default=20000, help="Labeled maps for eval.")
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--embed-dim", type=int, default=384)
    p.add_argument("--depth", type=int, default=12)
    p.add_argument("--heads", type=int, default=6)
    p.add_argument("--attention", default="full", choices=["full", "isab"])
    p.add_argument("--out-dim", type=int, default=4096, help="DINO prototype count.")
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args(argv)


def main(argv=None) -> None:  # pragma: no cover - needs Spark/MLflow/GPU
    import mlflow  # ty: ignore[unresolved-import]
    import torch
    from databricks.sdk.runtime import spark  # type: ignore

    from wafer_embeddings.eval.metrics import effective_rank
    from wafer_embeddings.model.dino import DinoModel, DINOHead, DINOLoss
    from wafer_embeddings.model.encoder import PerDieViT

    a = _args(argv)
    log = get_logger("wafer_embeddings.train")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info(f"device={device} torch={torch.__version__}")
    rng = np.random.default_rng(a.seed)
    torch.manual_seed(a.seed)

    fqn = f"{a.catalog}.{a.schema}.{a.table}"
    t = spark.table(fqn)
    with stage(log, "load train split"):
        q = t.where("split = 'train'")
        if a.max_train:
            q = q.limit(a.max_train)
        train_rows = q.toPandas().to_dict("records")
        train_wafers, _, _ = wafers_from_rows(train_rows)
    with stage(log, "load labeled eval"):
        eval_rows = t.where("label_id >= 0").limit(a.eval_cap).toPandas().to_dict("records")
        eval_wafers, eval_labels, eval_splits = wafers_from_rows(eval_rows)
    log.info(f"train={len(train_wafers)} eval_labeled={len(eval_wafers)}")

    encoder = PerDieViT(
        embed_dim=a.embed_dim,
        width=a.embed_dim,
        depth=a.depth,
        n_heads=a.heads,
        attention=a.attention,
    )
    dino = DinoModel(encoder, DINOHead(a.embed_dim, out_dim=a.out_dim)).to(device)
    loss_fn = DINOLoss(out_dim=a.out_dim).to(device)
    opt = torch.optim.AdamW(
        list(dino.student_enc.parameters()) + list(dino.student_head.parameters()), lr=a.lr
    )

    if a.experiment:
        mlflow.set_experiment(a.experiment)
    with mlflow.start_run(run_name="dino-g1"):
        mlflow.log_params(vars(a) | {"device": device, "n_classes": N_CLASSES})
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
                log=log.info,
                log_every=50,
            )
        for i, lv in enumerate(losses):
            if i % 50 == 0:
                mlflow.log_metric("dino_loss", lv, step=i)

        with stage(log, "embed + eval (G1)"):
            emb = embed_all(dino, eval_wafers, device=device)
            metrics = g1_metrics(emb, eval_labels, eval_splits, N_CLASSES)
            metrics["effective_rank"] = effective_rank(emb)
        mlflow.log_metrics({k: v for k, v in metrics.items()})
        log.info(f"G1 metrics: {metrics}")

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
