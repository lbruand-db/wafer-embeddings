"""DINO training core (SPECS.md §4/§7).

Pure of Databricks/MLflow/GPU specifics so it is unit-testable on CPU; the training
job (P3/P4) wraps this with data loading, device placement, MLflow, and the AI
Runtime serverless-GPU environment.
"""

from __future__ import annotations

import math
from typing import Callable

import numpy as np

from wafer_embeddings.tokenize.augment import random_view
from wafer_embeddings.tokenize.tokenizer import TokenizedWafer, cap_tokens, collate, tokenize
from wafer_embeddings.train.positives import sample_neighbours


def wafers_from_rows(
    rows, max_tokens: int | None = None, rng: np.random.Generator | None = None
) -> tuple[list[TokenizedWafer], np.ndarray, np.ndarray]:
    """Reconstruct + tokenize wafers from Delta/parquet row dicts.

    Each row has a flattened ``wafer_map`` plus ``height``/``width``; returns the
    tokenized wafers alongside aligned ``label_id`` and ``split`` arrays. ``max_tokens``
    caps giant wafers (bounds GPU memory, SPECS.md §4).
    """
    if rng is None:
        rng = np.random.default_rng(0)
    wafers: list[TokenizedWafer] = []
    labels: list[int] = []
    splits: list[str] = []
    for r in rows:
        h, w = int(r["height"]), int(r["width"])
        m = np.asarray(r["wafer_map"], dtype=np.uint8).reshape(h, w)
        tw = tokenize(m)
        if max_tokens:
            tw = cap_tokens(tw, max_tokens, rng)
        wafers.append(tw)
        labels.append(int(r["label_id"]))
        splits.append(str(r["split"]))
    return wafers, np.array(labels, dtype=np.int64), np.array(splits, dtype=object)


def _to_device(views: list[tuple], device: str) -> list[tuple]:
    if device == "cpu":
        return views
    return [(c.to(device), s.to(device), m.to(device)) for (c, s, m) in views]


def build_views(
    wafers: list[TokenizedWafer],
    rng: np.random.Generator,
    n_views: int = 2,
    orientation_invariant: bool = True,
    crop_scale: float = 0.9,
    n_local: int = 0,
    local_crop_area: tuple[float, float] = (0.05, 0.4),
    **aug,
) -> list[tuple]:
    """Make ``n_views`` global + ``n_local`` local augmented, collated views of a batch.

    Each view is a ``(coords, state_ids, mask)`` tensor triple. Global views come first
    (the teacher sees only those); local views are small crops (``local_crop_area`` of the
    die bounding box, DINO multi-crop, ref [1]) that the student must match to a global
    view. Extra keyword arguments (e.g. ``crop_area``, ``token_drop_p``) go to
    ``random_view``; ``crop_area`` applies to global views only.
    """
    views = []
    for i in range(n_views + n_local):
        kw = dict(aug)
        if i >= n_views:
            kw["crop_area"] = local_crop_area
        batch = [
            random_view(
                w, rng, orientation_invariant=orientation_invariant, crop_scale=crop_scale, **kw
            )
            for w in wafers
        ]
        views.append(collate(batch))
    return views


def param_groups(*modules) -> list[dict]:
    """Optimizer param groups as in DINO (ref [1] ``get_params_groups``).

    Weights get weight decay (and are tagged ``wd_schedule`` so ``fit_dino`` can follow
    the cosine weight-decay schedule); biases and 1-D params (norms) get none.
    """
    reg, noreg = [], []
    for m in modules:
        for name, p in m.named_parameters():
            if not p.requires_grad:
                continue
            (noreg if name.endswith(".bias") or p.ndim == 1 else reg).append(p)
    return [{"params": reg, "wd_schedule": True}, {"params": noreg, "weight_decay": 0.0}]


def scaled_lr(batch_size: int, base_lr: float = 5e-4) -> float:
    """DINO's linear LR scaling rule: ``base_lr * batch_size / 256`` (ref [1])."""
    return base_lr * batch_size / 256.0


def collapse_stats(dino, loss_fn, teacher_logits, view) -> dict[str, float]:
    """Collapse gauges for one step (ref [1]; SimSiam's embedding-std check).

    - ``t_entropy`` / ``t_maxp``: entropy and mean max-probability of the centered,
      sharpened teacher target. Uniform collapse = entropy ~ ln(out_dim), maxp ~ 1/out_dim;
      one-prototype collapse = entropy ~ 0, maxp ~ 1. Healthy training sits in between.
    - ``emb_std``: mean per-dimension std of the L2-normalized student encoder output
      across the batch, scaled by sqrt(dim) so a well-spread embedding is ~1 and a
      collapsed one (every wafer in the same direction) goes to 0.
    """
    import torch

    with torch.no_grad():
        t = torch.softmax((teacher_logits - loss_fn.center) / loss_fn.teacher_temp, dim=-1)
        ent = -(t * (t + 1e-12).log()).sum(dim=-1).mean()
        emb = dino.student_enc(*view)
        emb_std = (
            emb.std(dim=0).mean() * math.sqrt(emb.shape[-1]) if len(emb) > 1 else emb.new_zeros(())
        )
    return {
        "t_entropy": float(ent),
        "t_maxp": float(t.max(dim=-1).values.mean()),
        "emb_std": float(emb_std),
    }


def train_step(
    dino,
    loss_fn,
    optimizer,
    wafers: list[TokenizedWafer],
    rng: np.random.Generator,
    orientation_invariant: bool = True,
    device: str = "cpu",
    freeze_last: bool = False,
    clip_grad: float | None = None,
    stats: dict[str, float] | None = None,
    aug: dict | None = None,
    positives: list[TokenizedWafer] | None = None,
) -> float:
    """One DINO optimization step over a batch of tokenized wafers.

    ``freeze_last`` zeros the prototype (last-layer) gradient this step — DINO's
    early-training stabilizer (ref [1]): without it, real-data training is unstable and
    oscillates in and out of the uniform collapse. ``clip_grad`` clips the global grad
    norm (DINO uses 3.0). If ``stats`` is given, it is filled with ``collapse_stats`` and
    the pre-clip ``grad_norm`` (costs one extra no-grad encoder pass, so only ask on log
    steps). ``aug`` holds extra ``random_view`` keyword arguments (augmentation knobs).
    ``positives`` (one wafer per batch item, e.g. a descriptor neighbour) adds one global
    view of each as an extra student view that must match the anchor's teacher views.
    """
    import torch

    views = build_views(
        wafers, rng, n_views=2, orientation_invariant=orientation_invariant, **(aug or {})
    )
    if positives is not None:
        global_aug = {
            k: v for k, v in (aug or {}).items() if k not in ("n_local", "local_crop_area")
        }
        views += build_views(
            positives, rng, n_views=1, orientation_invariant=orientation_invariant, **global_aug
        )
    views = _to_device(views, device)
    student = dino.student_views(views)  # all views (global + local + positives)
    teacher = dino.teacher_views(views[:2])  # the anchor's global views only (ref [1])
    if stats is not None:
        stats.update(collapse_stats(dino, loss_fn, teacher[0], views[0]))
    loss = loss_fn(student, teacher)
    optimizer.zero_grad()
    loss.backward()
    if freeze_last:
        proto = getattr(getattr(dino, "student_head", None), "prototypes", None)
        if proto is not None:
            proto.weight.grad = None
    params = [p for g in optimizer.param_groups for p in g["params"] if p.grad is not None]
    if clip_grad is not None or stats is not None:
        norm = torch.nn.utils.clip_grad_norm_(
            params, clip_grad if clip_grad is not None else float("inf")
        )
        if stats is not None:
            stats["grad_norm"] = float(norm)
    optimizer.step()
    dino.update_teacher()
    return float(loss.detach())


def _cosine(start: float, end: float, t: float) -> float:
    """Cosine interpolate start -> end for t in [0, 1]."""
    t = min(max(t, 0.0), 1.0)
    return end + 0.5 * (start - end) * (1.0 + math.cos(math.pi * t))


def fit_dino(
    dino,
    loss_fn,
    optimizer,
    wafers: list[TokenizedWafer],
    *,
    steps: int,
    batch_size: int,
    rng: np.random.Generator,
    device: str = "cpu",
    orientation_invariant: bool = True,
    base_lr: float | None = None,
    lr_min: float = 1e-6,
    warmup_frac: float = 0.1,
    teacher_temp: tuple[float, float] = (0.04, 0.04),
    teacher_temp_warmup_frac: float = 0.3,
    momentum: tuple[float, float] = (0.996, 1.0),
    freeze_last_frac: float = 0.1,
    clip_grad: float | None = None,
    aug: dict | None = None,
    weight_decay: tuple[float, float] | None = None,
    log: Callable[[str], None] | None = None,
    log_every: int = 50,
    monitor: Callable[[int, dict[str, float]], None] | None = None,
    select_fn: Callable[[], float] | None = None,
    select_every: int = 0,
    track_fn: Callable[[int], None] | None = None,
    track_every: int = 0,
    neighbours: np.ndarray | None = None,
) -> list[float]:
    """Run ``steps`` DINO steps with LR / teacher-temp / momentum schedules (ref [1]).

    LR: linear warmup 0->base_lr over ``warmup_frac``, then cosine to ``lr_min``.
    Teacher temp: linear warmup over ``teacher_temp_warmup_frac``, then held. Default is a
    constant 0.04 — on real WM-811K a warmup to 0.07 left the teacher too soft to sharpen
    the small cosine-logit spread, pinning the loss at ln(out_dim) (uniform collapse).
    Teacher EMA momentum: cosine from ``momentum[0]`` to ``momentum[1]``.
    ``freeze_last_frac``: freeze the prototype layer for the first fraction of steps
    (DINO's stabilizer; prevents the early collapse-and-oscillate on real data).
    ``clip_grad``: global grad-norm clip (None = off). ``aug``: extra ``random_view``
    keyword arguments, e.g. ``{"token_drop_p": 0.5, "fail_drop_p": 0.4}``, plus
    ``n_local`` / ``local_crop_area`` for multi-crop. ``weight_decay=(start, end)``:
    cosine weight-decay schedule (DINO: 0.04 -> 0.4) on param groups tagged
    ``wd_schedule`` (see ``param_groups``); None leaves the optimizer's decay alone.

    Every ``log_every`` steps, ``collapse_stats`` + grad norm are computed and passed to
    ``monitor(step, stats)`` and the ``log`` line. If ``select_fn`` is given (higher =
    better; must not touch the reported eval queries), it is scored before training and
    every ``select_every`` steps, and the best-scoring weights are restored at the end
    (step 0 included, so a run that only degrades returns the untrained weights). The
    chosen step is reported to ``monitor`` as ``best_step`` / ``best_score``.

    ``track_fn(step)`` is a log-only hook called before training and every
    ``track_every`` steps, e.g. to record val metrics over training. Its last call (at
    ``steps``) comes after any keep-best restore, so it describes the returned weights. It
    never changes the weights, so no label it looks at can steer training.

    ``neighbours`` ((N, k) indices into ``wafers``, e.g. ``positives.descriptor_neighbours``)
    turns on descriptor-guided positives: each step, every batch wafer also brings one
    random neighbour as an extra student view (see ``train_step``).
    """
    n = len(wafers)
    bs = min(batch_size, n)
    if base_lr is None:
        base_lr = optimizer.param_groups[0]["lr"]
    warmup = max(int(steps * warmup_frac), 1)
    tt_warmup = max(int(steps * teacher_temp_warmup_frac), 1)
    freeze_steps = int(steps * freeze_last_frac)
    tt0, tt1 = teacher_temp
    m0, m1 = momentum
    losses: list[float] = []
    best: tuple[float, int, dict] | None = None

    def _select(step: int) -> None:
        nonlocal best
        score = select_fn()  # ty: ignore[call-non-callable]
        if log is not None:
            log(f"step {step}/{steps} select_score={score:.4f}")
        if best is None or score > best[0]:
            snap = {k: v.detach().clone() for k, v in dino.state_dict().items()}
            best = (score, step, snap)

    if select_fn is not None:
        _select(0)
    if track_fn is not None and track_every:
        track_fn(0)
    for step in range(1, steps + 1):
        lr = (
            base_lr * step / warmup
            if step <= warmup
            else _cosine(base_lr, lr_min, (step - warmup) / max(steps - warmup, 1))
        )
        for g in optimizer.param_groups:
            g["lr"] = lr
        if weight_decay is not None:
            wd = _cosine(weight_decay[0], weight_decay[1], step / max(steps, 1))
            for g in optimizer.param_groups:
                if g.get("wd_schedule"):
                    g["weight_decay"] = wd
        loss_fn.teacher_temp = tt0 + (tt1 - tt0) * min(step / tt_warmup, 1.0)
        dino.teacher_momentum = _cosine(m0, m1, step / max(steps, 1))

        idx = rng.integers(0, n, size=bs)
        batch = [wafers[i] for i in idx]
        positives = None
        if neighbours is not None:
            positives = [wafers[j] for j in sample_neighbours(neighbours, idx, rng)]
        log_step = step % log_every == 0 or step == steps
        stats: dict[str, float] | None = {} if log_step else None
        loss = train_step(
            dino,
            loss_fn,
            optimizer,
            batch,
            rng,
            orientation_invariant=orientation_invariant,
            device=device,
            freeze_last=step <= freeze_steps,
            clip_grad=clip_grad,
            stats=stats,
            aug=aug,
            positives=positives,
        )
        losses.append(loss)
        if stats is not None:
            stats |= {"loss": loss, "lr": lr, "teacher_temp": float(loss_fn.teacher_temp)}
            if monitor is not None:
                monitor(step, stats)
            if log is not None:
                log(
                    f"step {step}/{steps} loss={loss:.4f} lr={lr:.2e} "
                    f"ttemp={loss_fn.teacher_temp:.3f} t_ent={stats['t_entropy']:.3f} "
                    f"t_maxp={stats['t_maxp']:.4f} emb_std={stats['emb_std']:.3f} "
                    f"grad_norm={stats['grad_norm']:.3f}"
                )
        if select_fn is not None and select_every and (step % select_every == 0 or step == steps):
            _select(step)
        if track_fn is not None and track_every and step % track_every == 0 and step < steps:
            track_fn(step)
    if best is not None:
        score, best_step, snap = best
        dino.load_state_dict(snap)
        if log is not None:
            log(f"restored best weights from step {best_step} (select_score={score:.4f})")
        if monitor is not None:
            monitor(steps, {"best_step": float(best_step), "best_score": score})
    if track_fn is not None and track_every:
        track_fn(steps)  # after any restore: the last point describes the returned weights
    return losses


def embed_all(
    dino,
    wafers: list[TokenizedWafer],
    *,
    device: str = "cpu",
    batch_size: int = 256,
    which: str = "student",
) -> np.ndarray:
    """Embed all wafers with the student or EMA-teacher encoder -> (N, D) L2-normalized.

    DINO evaluates the **teacher** (ref [1]: it outperforms the student throughout
    training); ``which="teacher"`` gives that.
    """
    if which not in ("student", "teacher"):
        raise ValueError(f"which must be 'student' or 'teacher', got {which!r}")
    enc = dino.teacher_enc if which == "teacher" else dino.student_enc
    import torch

    was_training = dino.training
    dino.eval()  # disable grad-checkpointing path during inference
    out: list[np.ndarray] = []
    with torch.no_grad():
        for i in range(0, len(wafers), batch_size):
            coords, states, mask = collate(wafers[i : i + batch_size])
            if device != "cpu":
                coords, states, mask = coords.to(device), states.to(device), mask.to(device)
            emb = enc(coords, states, mask)
            out.append(emb.cpu().numpy())
    dino.train(was_training)
    return np.concatenate(out, axis=0) if out else np.zeros((0, 1), dtype=np.float32)
