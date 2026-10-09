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
    param_groups,
    scaled_lr,
    selection_score,
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
    p.add_argument(
        "--eval-split",
        default="val",
        choices=list(EVAL_SPLITS),
        help="Query split: 'val' for development; 'test' only for the final report.",
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
    p.add_argument("--bottleneck", type=int, default=64, help="DINO head bottleneck dim.")
    p.add_argument(
        "--pre-norm",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Pre-LN attention blocks + final LayerNorm (ViT/DINO form).",
    )
    p.add_argument("--lr", type=float, default=5e-4, help="Peak LR (0 = 5e-4*batch/256 rule).")
    p.add_argument(
        "--weight-decay",
        type=float,
        default=0.0,
        help="AdamW weight decay on weights only (biases/norms never decayed; 0 = off).",
    )
    p.add_argument(
        "--weight-decay-end",
        type=float,
        default=None,
        help="If set, cosine weight-decay schedule weight-decay -> this (DINO: 0.04 -> 0.4).",
    )
    p.add_argument("--n-local", type=int, default=0, help="DINO local crops per wafer.")
    p.add_argument(
        "--global-crop-area",
        type=float,
        nargs=2,
        default=None,
        metavar=("LO", "HI"),
        help="Random global-crop area range (DINO: 0.4 1.0); default: fixed 0.9 side.",
    )
    p.add_argument(
        "--local-crop-area", type=float, nargs=2, default=[0.05, 0.4], metavar=("LO", "HI")
    )
    p.add_argument("--die-noise", type=float, default=0.005, help="Pass<->fail flip prob.")
    p.add_argument(
        "--freeze-last-frac",
        type=float,
        default=0.1,
        help="Freeze the DINO prototype layer for this fraction of steps (stabilizer).",
    )
    p.add_argument("--clip-grad", type=float, default=0.0, help="Global grad-norm clip (0 = off).")
    p.add_argument("--warmup-frac", type=float, default=0.1, help="LR linear-warmup fraction.")
    p.add_argument(
        "--select-every",
        type=int,
        default=0,
        help="Score a checkpoint on the train-side labeled kNN bank every N steps and keep "
        "the best (0 = off: train for a fixed schedule, no labels anywhere in training).",
    )
    p.add_argument(
        "--track-every",
        type=int,
        default=0,
        help="Log student+teacher eval metrics every N steps (log-only; 0 = off).",
    )
    p.add_argument("--run-name", default="dino", help="MLflow run name.")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args(argv)
    if a.track_every < 0:
        p.error("--track-every must be >= 0")
    if a.track_every and a.eval_split == "test":
        # a training curve on test invites tuning on it; test is scored once, at the end
        p.error("--track-every is development-only; not allowed with --eval-split test")
    return a


def eval_metric_keys(split: str, model: str, metrics: dict[str, float]) -> dict[str, float]:
    """Namespace eval metrics for MLflow as ``<split>/<model>/<metric>``.

    One MLflow section per eval split (``val`` during development, ``test`` only for the
    final report), so dev and final numbers never mix. ``model`` is ``student`` /
    ``teacher`` (curves over training steps; step 0 is the untrained encoder) or
    ``polar`` (the training-free bar, logged at the first and last step so it draws a
    flat reference line next to the curves).
    """
    return {f"{split}/{model}/{k}": float(v) for k, v in metrics.items()}


def train_recipe(a) -> dict:
    """Map parsed CLI args to the training recipe (pure, unit-tested).

    Returns ``lr`` (``--lr 0`` -> DINO's ``5e-4 * batch / 256`` rule), ``aug`` (the
    ``random_view`` / multi-crop knobs for ``fit_dino``) and ``weight_decay`` (a
    ``(start, end)`` cosine schedule, or None for a constant decay).
    """
    aug: dict = {"die_noise_p": a.die_noise}
    if a.global_crop_area is not None:
        aug["crop_area"] = tuple(a.global_crop_area)
    if a.n_local:
        aug["n_local"] = a.n_local
        aug["local_crop_area"] = tuple(a.local_crop_area)
    wd = None
    if a.weight_decay_end is not None:
        wd = (a.weight_decay, a.weight_decay_end)
    return {"lr": a.lr or scaled_lr(a.batch_size), "aug": aug, "weight_decay": wd}


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


EVAL_SPLITS = ("val", "test")
# Headline metrics: tracked during training and compared in the final report.
HEADLINE_KEYS = (
    "xgroup_knn_macro_recall",
    "xgroup_precision@10",
    "knn_acc",
    "knn_macro_recall",
    "map@10",
    "cluster_nmi",
)


def load_eval_table(
    dset, eval_cap: int, seed: int, balanced: bool = True, query_split: str = "val"
):
    """Seeded, class-stratified labeled eval sample from a pyarrow dataset.

    Queries come from labeled maps of ``query_split`` only, the kNN bank from labeled
    train maps, ``eval_cap`` of each. Protocol (SPECS.md §8.7): develop on ``"val"``;
    ``"test"`` is touched once, for the final report, so test labels never steer design
    choices. Only the small id/label/split columns are read for the full
    labeled set; wafer maps are fetched just for the sampled ids. The dedicated RNG
    keeps the eval set identical across runs with the same seed, so runs are comparable.
    """
    import pyarrow.dataset as ds

    if query_split not in EVAL_SPLITS:
        raise ValueError(f"query_split must be one of {EVAL_SPLITS}, got {query_split!r}")
    meta = dset.to_table(columns=["id", "label_id", "split"], filter=ds.field("label_id") >= 0)
    ids = meta.column("id").to_numpy()
    labels = meta.column("label_id").to_numpy()
    splits = np.array(meta.column("split").to_pylist(), dtype=object)
    rng = np.random.default_rng(seed)
    picked = []
    for side in (splits == query_split, splits == "train"):
        pos = stratified_indices(labels[side], eval_cap, rng, balanced=balanced)
        picked.append(ids[side][pos])
    want = np.concatenate(picked).tolist()
    return dset.to_table(filter=ds.field("id").isin(want))


def main(argv=None) -> None:  # pragma: no cover - needs Spark/MLflow/GPU
    import mlflow  # ty: ignore[unresolved-import]
    import pyarrow.dataset as ds
    import torch

    from wafer_embeddings.eval.baselines import polar_fail_embedding
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
    if a.eval_split == "test":
        log.warning("EVAL ON TEST SPLIT: final report only - do not tune on these numbers")
    with stage(log, f"load labeled eval ({a.eval_split} queries, {a.eval_sampling} sample)"):
        tbl = load_eval_table(
            dset,
            a.eval_cap,
            a.seed,
            balanced=a.eval_sampling == "balanced",
            query_split=a.eval_split,
        )
        rows = tbl.to_pylist()
        eval_wafers, eval_labels, eval_splits = wafers_from_rows(
            rows, max_tokens=a.max_tokens, rng=rng
        )
        # map shape = device proxy: devices span lots, so same-shape neighbours share labels
        eval_groups = np.array([f"{r['height']}x{r['width']}" for r in rows], dtype=object)
        del rows
    q = eval_splits == a.eval_split
    mix = dict(zip(*(v.tolist() for v in np.unique(eval_labels[q], return_counts=True))))
    log.info(f"train={len(train_wafers)} eval_labeled={len(eval_wafers)} query class mix={mix}")

    encoder = PerDieViT(
        embed_dim=a.embed_dim,
        width=a.embed_dim,
        depth=a.depth,
        n_heads=a.heads,
        attention=a.attention,
        grad_checkpoint=a.grad_checkpoint,
        pre_norm=a.pre_norm,
    )
    head = DINOHead(a.embed_dim, out_dim=a.out_dim, bottleneck=a.bottleneck)
    dino = DinoModel(encoder, head).to(device)
    loss_fn = DINOLoss(out_dim=a.out_dim).to(device)
    recipe = train_recipe(a)
    log.info(f"recipe: {recipe}")
    # weights decayed, biases / norms not (DINO's param grouping)
    opt = torch.optim.AdamW(
        param_groups(dino.student_enc, dino.student_head),
        lr=recipe["lr"],
        weight_decay=a.weight_decay,
    )

    if a.experiment:
        mlflow.set_experiment(a.experiment)

    def _eval(
        prefix: str, emb: np.ndarray | None = None, which: str = "student"
    ) -> dict[str, float]:
        if emb is None:
            emb = embed_all(dino, eval_wafers, device=device, which=which)
        m = g1_metrics(
            emb, eval_labels, eval_splits, N_CLASSES, class_names=CLASS_NAMES, groups=eval_groups
        )
        m["effective_rank"] = effective_rank(emb)
        return {f"{prefix}{k}": v for k, v in m.items()}

    # Checkpoint selection uses only the labeled train-side bank, never the G1 queries.
    bank = np.nonzero(eval_splits == "train")[0]
    bank_wafers = [eval_wafers[i] for i in bank]

    def _select() -> float:
        emb = embed_all(dino, bank_wafers, device=device)
        return selection_score(emb, eval_labels[bank], N_CLASSES, groups=eval_groups[bank])

    def _track(step: int) -> None:
        for which in ("student", "teacher"):
            m = _eval("", which=which)
            # g1_metrics omits a metric it can't compute (e.g. too few queries): skip it
            got = {k: m[k] for k in HEADLINE_KEYS if k in m}
            mlflow.log_metrics(eval_metric_keys(a.eval_split, which, got), step=step)
            log.info(f"track step {step} {which}: " + str({k: round(v, 4) for k, v in got.items()}))

    def _monitor(step: int, stats: dict[str, float]) -> None:
        stats = {("dino_loss" if k == "loss" else k): v for k, v in stats.items()}
        mlflow.log_metrics({f"train/{k}": v for k, v in stats.items()}, step=step)

    split = a.eval_split
    with mlflow.start_run(run_name=a.run_name):
        mlflow.log_params(vars(a) | {"device": device, "n_classes": N_CLASSES})
        # AI Runtime may hand the job a pre-created run; the tag renames it either way
        mlflow.set_tags({"mlflow.runName": a.run_name, "eval_split": split, "recipe": str(recipe)})
        # Same eval maps, freshly initialized encoder: the bar training must beat. It is
        # step 0 of the student/teacher curves (both start as the same weights).
        with stage(log, "embed + eval untrained encoder (baseline)"):
            baseline = _eval("untrained_")
        untrained = {k.removeprefix("untrained_"): v for k, v in baseline.items()}
        for which in ("student", "teacher"):
            mlflow.log_metrics(eval_metric_keys(split, which, untrained), step=0)
        log.info(f"untrained baseline: {baseline}")
        # Training-free handcrafted defect descriptor: the bar a learned encoder must beat.
        with stage(log, "eval polar FAIL-histogram baseline"):
            polar = _eval("polar_", polar_fail_embedding(eval_wafers))
        polar_m = {k.removeprefix("polar_"): v for k, v in polar.items()}
        mlflow.log_metrics(eval_metric_keys(split, "polar", polar_m), step=0)
        log.info(f"polar baseline: {polar}")

        with stage(log, "train DINO"):
            fit_dino(
                dino,
                loss_fn,
                opt,
                train_wafers,
                steps=a.steps,
                batch_size=a.batch_size,
                rng=rng,
                device=device,
                freeze_last_frac=a.freeze_last_frac,
                warmup_frac=a.warmup_frac,
                clip_grad=a.clip_grad or None,
                aug=recipe["aug"],
                weight_decay=recipe["weight_decay"],
                log=log.info,
                log_every=50,
                monitor=_monitor,
                select_fn=_select if a.select_every else None,
                select_every=a.select_every,
                track_fn=_track if a.track_every else None,
                track_every=a.track_every,
            )

        with stage(log, "embed + eval (G1)"):
            metrics = _eval("")
            # DINO reports the EMA teacher (ref [1]); log it alongside the student
            teacher = _eval("teacher_", which="teacher")
        teacher_m = {k.removeprefix("teacher_"): v for k, v in teacher.items()}
        mlflow.log_metrics(eval_metric_keys(split, "student", metrics), step=a.steps)
        mlflow.log_metrics(eval_metric_keys(split, "teacher", teacher_m), step=a.steps)
        mlflow.log_metrics(eval_metric_keys(split, "polar", polar_m), step=a.steps)
        log.info(f"G1 metrics: {metrics}")
        log.info(f"teacher metrics: {teacher}")
        for k in HEADLINE_KEYS:
            log.info(
                f"{k}: polar={polar.get('polar_' + k)} "
                f"untrained={baseline.get('untrained_' + k)} trained={metrics.get(k)} "
                f"teacher={teacher.get('teacher_' + k)}"
            )

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
