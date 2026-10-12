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
from dataclasses import dataclass

import numpy as np

from wafer_embeddings.data.rows import shape_groups
from wafer_embeddings.data.wm811k import CLASS_NAMES
from wafer_embeddings.eval.baselines import DESCRIPTORS
from wafer_embeddings.obs import get_logger, stage
from wafer_embeddings.train.evaluate import G1_KEYS, HEADLINE_KEYS
from wafer_embeddings.train import (
    embed_all,
    fit_dino,
    g1_metrics,
    param_groups,
    scaled_lr,
    stratified_indices,
    wafers_from_rows,
)

N_CLASSES = len(CLASS_NAMES)


# The pre-2026-10-09 recipe (superseded: it did not beat the untrained encoder), kept
# reproducible. Defaults below are the R1 recipe behind wafer_encoder v2 (PLAN.md P1).
LEGACY_RECIPE = (
    "--embed-dim 384 --heads 6 --depth 12 --attention full --max-tokens 4096 "
    "--batch-size 128 --steps 2000 --max-train 50000 --lr 5e-4 --weight-decay 0 "
    "--weight-decay-end 0 --n-local 0 --global-crop-area 0.81 0.81 --die-noise 0.005 "
    "--no-pre-norm --out-dim 1024 --bottleneck 64 --freeze-last-frac 0.1 --clip-grad 0"
)


def _args(argv=None):
    p = argparse.ArgumentParser(description="DINO pretrain + G1 eval (WM-811K).")
    p.add_argument("--catalog", required=True)
    p.add_argument("--schema", required=True)
    p.add_argument("--table", default="wafer_maps")
    p.add_argument("--volume", default="raw", help="UC volume holding wafer_maps_parquet.")
    p.add_argument(
        "--parquet",
        default=None,
        help="wafer_maps parquet path (default: the UC volume export); e.g. a local copy "
        "for a CPU smoke run of this job.",
    )
    p.add_argument("--experiment", default=None, help="MLflow experiment (default: AI Runtime's).")
    p.add_argument("--max-train", type=int, default=100000, help="Train maps to sample (0=all).")
    p.add_argument(
        "--eval-cap",
        type=int,
        default=1000,
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
    p.add_argument("--max-tokens", type=int, default=512, help="Cap dies/wafer (GPU mem).")
    p.add_argument(
        "--grad-checkpoint",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Gradient-checkpoint encoder blocks (fits depth-12 in A10 memory).",
    )
    p.add_argument("--steps", type=int, default=10000)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--embed-dim", type=int, default=128)
    p.add_argument("--depth", type=int, default=6)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--attention", default="isab", choices=["full", "isab"])
    p.add_argument("--out-dim", type=int, default=2048, help="DINO prototype count.")
    p.add_argument("--bottleneck", type=int, default=256, help="DINO head bottleneck dim.")
    p.add_argument(
        "--pre-norm",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Pre-LN attention blocks + final LayerNorm (ViT/DINO form).",
    )
    p.add_argument("--lr", type=float, default=0.0, help="Peak LR (0 = 5e-4*batch/256 rule).")
    p.add_argument(
        "--weight-decay",
        type=float,
        default=0.04,
        help="AdamW weight decay on weights only (biases/norms never decayed; 0 = off).",
    )
    p.add_argument(
        "--weight-decay-end",
        type=float,
        default=0.4,
        help="If set, cosine weight-decay schedule weight-decay -> this (DINO: 0.04 -> 0.4).",
    )
    p.add_argument("--n-local", type=int, default=6, help="DINO local crops per wafer.")
    p.add_argument(
        "--global-crop-area",
        type=float,
        nargs=2,
        default=[0.4, 1.0],
        metavar=("LO", "HI"),
        help="Random global-crop area range (DINO 0.4 1.0; 0.81 0.81 = the old fixed 0.9 side).",
    )
    p.add_argument(
        "--local-crop-area", type=float, nargs=2, default=[0.05, 0.4], metavar=("LO", "HI")
    )
    p.add_argument(
        "--nn-positives",
        type=int,
        default=0,
        help="Descriptor-guided positives: each wafer's K nearest train maps by polar FAIL "
        "descriptor; one random neighbour per step joins the student views (0 = off; label-free).",
    )
    p.add_argument(
        "--label-positives",
        type=int,
        default=0,
        help="ABLATION ONLY (uses train labels; such a run can't be registered): K random "
        "same-class train maps as positives, SWaCo-style (SPECS.md §16 item 3). 0 = off.",
    )
    p.add_argument(
        "--nn-descriptor",
        default="pixel",
        choices=list(DESCRIPTORS),
        help="Descriptor for --nn-positives: the flattened FAIL raster + PCA (orientation-"
        "aware; better neighbours on WM-811K) or the rotation-invariant polar histogram.",
    )
    p.add_argument(
        "--orientation-invariant",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Random rotation / flip of every view. --no-orientation-invariant keeps the map "
        "orientation (WM-811K maps are aligned; the orientation-aware pixel baseline wins).",
    )
    p.add_argument(
        "--dinov2-baseline",
        action="store_true",
        help="Also score frozen off-the-shelf DINOv2 (ViT-S/14, torch.hub: needs internet) on "
        "rasterized maps (SPECS.md §16 item 1).",
    )
    p.add_argument("--die-noise", type=float, default=0.03, help="Pass<->fail flip prob.")
    p.add_argument(
        "--freeze-last-frac",
        type=float,
        default=0.01,
        help="Freeze the DINO prototype layer for this fraction of steps (stabilizer).",
    )
    p.add_argument("--clip-grad", type=float, default=3.0, help="Global grad-norm clip (0 = off).")
    p.add_argument("--warmup-frac", type=float, default=0.1, help="LR linear-warmup fraction.")
    p.add_argument(
        "--track-every",
        type=int,
        default=None,
        help="Log student+teacher eval metrics every N steps (log-only; 0 = off). "
        "Default: 1000 on val, off on test (no training curve on the test split).",
    )
    p.add_argument(
        "--track-clustering",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include the KMeans cluster_* metrics in tracking evals (the final eval always "
        "does); --no-track-clustering is the light path for large evals / frequent tracking.",
    )
    p.add_argument(
        "--probe-size",
        type=int,
        default=2048,
        help="Train maps (no labels) in the fixed RankMe probe set logged when tracking.",
    )
    p.add_argument("--run-name", default="dino", help="MLflow run name.")
    p.add_argument("--seed", type=int, default=0, help="Training seed (init, sample, views).")
    p.add_argument(
        "--eval-seed",
        type=int,
        default=None,
        help="Eval-sample seed (which labeled maps; caps use seed 0, as serving); default: --seed. "
        "Fix it while varying --seed to measure training noise alone (GAPS 1.2).",
    )
    a = p.parse_args(argv)
    if a.eval_seed is None:
        a.eval_seed = a.seed
    if a.track_every is None:
        a.track_every = 0 if a.eval_split == "test" else 1000
    if a.track_every < 0:
        p.error("--track-every must be >= 0")
    if a.label_positives and a.nn_positives:
        p.error("--label-positives and --nn-positives are exclusive")
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


def beats_polar(trained: dict[str, float], polar: dict[str, float]) -> bool:
    """Revised Gate G1 for one run (PLAN.md P1): above polar on both cross-device metrics.

    "Beyond seed noise" needs several seeds, so a single run's pass is necessary, not
    sufficient.
    """
    return all(trained.get(k, 0.0) > polar.get(k, float("inf")) for k in G1_KEYS)


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


@dataclass
class EvalSet:
    """The labeled eval sample: queries of one split + the train-side kNN bank."""

    wafers: list
    labels: np.ndarray
    splits: np.ndarray
    groups: np.ndarray  # map shape per wafer (the device proxy, ``data.rows.shape_groups``)

    @classmethod
    def load(cls, dset, a) -> EvalSet:
        """Seeded, class-stratified sample (``load_eval_table``), capped at seed 0 as in
        serving, so the eval sees the dies the served encoder sees."""
        rows = load_eval_table(
            dset,
            a.eval_cap,
            a.eval_seed,
            balanced=a.eval_sampling == "balanced",
            query_split=a.eval_split,
        ).to_pylist()
        wafers, labels, splits = wafers_from_rows(rows, max_tokens=a.max_tokens)
        return cls(wafers, labels, splits, shape_groups(rows))

    def metrics(self, emb: np.ndarray, clustering: bool = True) -> dict[str, float]:
        """G1 metrics (+ per-class recall) of an embedding of ``wafers``."""
        return g1_metrics(
            emb,
            self.labels,
            self.splits,
            N_CLASSES,
            class_names=CLASS_NAMES,
            groups=self.groups,
            clustering=clustering,
        )


def build_positives(a, train_wafers, train_labels, device: str = "cpu", log=None):
    """The ``(N, k)`` positives for ``fit_dino(neighbours=...)``, or None.

    ``--nn-positives``: descriptor neighbours (label-free); ``--label-positives``: the
    same-class ablation (uses train labels; such a run can't be registered).
    """
    from wafer_embeddings.eval.baselines import descriptor_embedding
    from wafer_embeddings.train.positives import descriptor_neighbours, label_neighbours

    if a.nn_positives:
        desc = descriptor_embedding(a.nn_descriptor, train_wafers, dim=a.embed_dim)
        return descriptor_neighbours(desc, a.nn_positives, device=device)
    if a.label_positives:
        if log is not None:
            log.warning("ABLATION: same-class positives use TRAIN LABELS (not servable)")
        return label_neighbours(train_labels, a.label_positives, np.random.default_rng(a.seed))
    return None


def build_training(a, device: str = "cpu"):
    """``(dino, loss_fn, optimizer, recipe)`` for the args: the encoder via
    ``serving.build_encoder`` (so serving rebuilds the same architecture), DINO head +
    EMA teacher, AdamW with DINO's param groups (weights decayed, biases / norms not)."""
    import torch

    from wafer_embeddings.model.dino import DinoModel, DINOHead, DINOLoss
    from wafer_embeddings.serving.encoder import build_encoder

    encoder = build_encoder(vars(a), grad_checkpoint=a.grad_checkpoint)
    head = DINOHead(a.embed_dim, out_dim=a.out_dim, bottleneck=a.bottleneck)
    dino = DinoModel(encoder, head).to(device)
    loss_fn = DINOLoss(out_dim=a.out_dim).to(device)
    recipe = train_recipe(a)
    opt = torch.optim.AdamW(
        param_groups(dino.student_enc, dino.student_head),
        lr=recipe["lr"],
        weight_decay=a.weight_decay,
    )
    return dino, loss_fn, opt, recipe


def fit_kwargs(a, recipe: dict) -> dict:
    """The ``fit_dino`` schedule / augmentation keyword arguments for the args."""
    return {
        "steps": a.steps,
        "batch_size": a.batch_size,
        "freeze_last_frac": a.freeze_last_frac,
        "warmup_frac": a.warmup_frac,
        "clip_grad": a.clip_grad or None,
        "aug": recipe["aug"],
        "weight_decay": recipe["weight_decay"],
        "orientation_invariant": a.orientation_invariant,
        "track_every": a.track_every,
    }


def baseline_embeddings(a, wafers, device: str = "cpu") -> dict[str, np.ndarray]:
    """Training-free bars on the eval wafers: polar, pixel-PCA (SPECS.md §16 item 24) and,
    with ``--dinov2-baseline``, frozen DINOv2 (§16 item 1). Logged next to the curves."""
    from wafer_embeddings.eval.baselines import pixel_pca_embedding, polar_fail_embedding

    out = {
        "polar": polar_fail_embedding(wafers),
        "pixel_pca": pixel_pca_embedding(wafers, dim=a.embed_dim),
    }
    if a.dinov2_baseline:
        from wafer_embeddings.eval.frozen import dinov2_embedding

        out["dinov2"] = dinov2_embedding(wafers, device=device)
    return out


def main(argv=None) -> None:  # pragma: no cover - needs Spark/MLflow/GPU
    import mlflow  # ty: ignore[unresolved-import]
    import pyarrow.dataset as ds
    import torch

    from wafer_embeddings.eval.metrics import effective_rank, rankme
    from wafer_embeddings.model.encoder import count_parameters

    a = _args(argv)
    log = get_logger("wafer_embeddings.train")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info(f"device={device} torch={torch.__version__}")
    rng = np.random.default_rng(a.seed)
    torch.manual_seed(a.seed)

    # AI Runtime serverless GPU has no Spark session; read the volume parquet export
    # with pyarrow (seeded samples, so we never materialize the whole corpus).
    pq = a.parquet or f"/Volumes/{a.catalog}/{a.schema}/{a.volume}/wafer_maps_parquet"
    dset = ds.dataset(pq, format="parquet")
    with stage(log, "load train split (seeded random sample)"):
        tbl = load_train_table(dset, a.max_train, a.seed)
        train_wafers, train_labels, _ = wafers_from_rows(
            tbl.to_pylist(), max_tokens=a.max_tokens, seed=a.seed
        )
    with stage(log, "positives (descriptor neighbours / ablation)"):
        neighbours = build_positives(a, train_wafers, train_labels, device=device, log=log)
    if a.eval_split == "test":
        log.warning("EVAL ON TEST SPLIT: final report only - do not tune on these numbers")
    with stage(log, f"load labeled eval ({a.eval_split} queries, {a.eval_sampling} sample)"):
        ev = EvalSet.load(dset, a)
    q = ev.splits == a.eval_split
    mix = dict(zip(*(v.tolist() for v in np.unique(ev.labels[q], return_counts=True))))
    log.info(f"train={len(train_wafers)} eval_labeled={len(ev.wafers)} query class mix={mix}")

    dino, loss_fn, opt, recipe = build_training(a, device)
    log.info(f"recipe: {recipe}")
    if a.experiment:
        mlflow.set_experiment(a.experiment)

    def _eval(emb: np.ndarray) -> dict[str, float]:
        m = ev.metrics(emb)
        m["effective_rank"] = effective_rank(emb)
        m["rankme"] = rankme(emb)
        return m

    def _embed(which: str = "student", wafers=None) -> np.ndarray:
        return embed_all(dino, ev.wafers if wafers is None else wafers, device=device, which=which)

    # Label-free health signal: RankMe on a fixed set of train maps (the train sample is
    # already a seeded random draw, so its head is a random probe set).
    probe_wafers = train_wafers[: a.probe_size]

    def _track(step: int) -> None:
        for which in ("student", "teacher"):
            r = rankme(_embed(which, probe_wafers))
            mlflow.log_metrics({f"train/{which}/rankme": r}, step=step)
            log.info(f"track step {step} {which}: train rankme={r:.2f}")
            if step >= a.steps:
                continue  # the final eval below logs this point, with every metric
            emb = _embed(which)
            m = ev.metrics(emb, clustering=a.track_clustering)
            m["rankme"] = rankme(emb)
            # g1_metrics omits a metric it can't compute (e.g. too few queries): skip it
            got = {k: m[k] for k in HEADLINE_KEYS if k in m}
            mlflow.log_metrics(eval_metric_keys(a.eval_split, which, got), step=step)
            log.info(f"track step {step} {which}: " + str({k: round(v, 4) for k, v in got.items()}))

    def _monitor(step: int, stats: dict[str, float]) -> None:
        stats = {("dino_loss" if k == "loss" else k): v for k, v in stats.items()}
        mlflow.log_metrics({f"train/{k}": v for k, v in stats.items()}, step=step)

    split = a.eval_split
    with mlflow.start_run(run_name=a.run_name):
        mlflow.log_params(
            vars(a)
            | {
                "device": device,
                "n_classes": N_CLASSES,
                # the served artifact is the encoder; the DINO head is training-only
                "n_params": count_parameters(dino.student_enc),
                "head_n_params": count_parameters(dino.student_head),
            }
        )
        # AI Runtime may hand the job a pre-created run; the tag renames it either way
        mlflow.set_tags(
            {
                "mlflow.runName": a.run_name,
                "eval_split": split,
                "recipe": str(recipe),
                # register.py refuses runs that saw training labels
                "uses_train_labels": str(bool(a.label_positives)).lower(),
            }
        )
        # Same eval maps, freshly initialized encoder: the bar training must beat. It is
        # step 0 of the student/teacher curves (both start as the same weights).
        with stage(log, "embed + eval untrained encoder (baseline)"):
            untrained = _eval(_embed())
        for which in ("student", "teacher"):
            mlflow.log_metrics(eval_metric_keys(split, which, untrained), step=0)
        log.info(f"untrained baseline: {untrained}")
        # Training-free bars, logged at the first and last step (flat reference lines).
        bars: dict[str, dict[str, float]] = {}
        for name, emb in baseline_embeddings(a, ev.wafers, device).items():
            with stage(log, f"eval {name} baseline"):
                bars[name] = _eval(emb)
            mlflow.log_metrics(eval_metric_keys(split, name, bars[name]), step=0)
            log.info(f"{name} baseline: {bars[name]}")

        with stage(log, "train DINO"):
            fit_dino(
                dino,
                loss_fn,
                opt,
                train_wafers,
                rng=rng,
                device=device,
                log=log.info,
                log_every=50,
                monitor=_monitor,
                track_fn=_track if a.track_every else None,
                neighbours=neighbours,
                **fit_kwargs(a, recipe),
            )

        with stage(log, "embed + eval (G1)"):
            metrics = _eval(_embed())
            # DINO reports the EMA teacher (ref [1]); log it alongside the student
            teacher = _eval(_embed("teacher"))
        mlflow.log_metrics(eval_metric_keys(split, "student", metrics), step=a.steps)
        mlflow.log_metrics(eval_metric_keys(split, "teacher", teacher), step=a.steps)
        for name, m in bars.items():
            mlflow.log_metrics(eval_metric_keys(split, name, m), step=a.steps)
        log.info(f"G1 metrics: {metrics}")
        log.info(f"teacher metrics: {teacher}")
        for k in HEADLINE_KEYS:
            ref = " ".join(f"{name}={m.get(k)}" for name, m in bars.items())
            log.info(
                f"{k}: {ref} untrained={untrained.get(k)} trained={metrics.get(k)} "
                f"teacher={teacher.get(k)}"
            )

        path = "/tmp/wafer_encoder.pt"
        torch.save({"state_dict": dino.student_enc.state_dict(), "config": vars(a)}, path)
        mlflow.log_artifact(path, artifact_path="encoder")

        gate = beats_polar(metrics, bars["polar"])
        mlflow.log_metric(f"{split}/beats_polar", float(gate))
        log.info(f"GATE G1 (single run; seed noise not accounted) {'PASS' if gate else 'NOT MET'}")


if __name__ == "__main__":  # pragma: no cover
    main()
