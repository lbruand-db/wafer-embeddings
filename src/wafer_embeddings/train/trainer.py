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
) -> list[tuple]:
    """Make ``n_views`` augmented, collated views of a batch of wafers.

    Each view is a ``(coords, state_ids, mask)`` tensor triple (DINO multi-crop; the
    crops double as the variable-size training signal, SPECS.md §5).
    """
    views = []
    for _ in range(n_views):
        aug = [
            random_view(w, rng, orientation_invariant=orientation_invariant, crop_scale=crop_scale)
            for w in wafers
        ]
        views.append(collate(aug))
    return views


def train_step(
    dino,
    loss_fn,
    optimizer,
    wafers: list[TokenizedWafer],
    rng: np.random.Generator,
    orientation_invariant: bool = True,
    device: str = "cpu",
    freeze_last: bool = False,
) -> float:
    """One DINO optimization step over a batch of tokenized wafers.

    ``freeze_last`` zeros the prototype (last-layer) gradient this step — DINO's
    early-training stabilizer (ref [1]): without it, real-data training is unstable and
    oscillates in and out of the uniform collapse.
    """
    views = build_views(wafers, rng, n_views=2, orientation_invariant=orientation_invariant)
    views = _to_device(views, device)
    student = dino.student_views(views)
    teacher = dino.teacher_views(views)
    loss = loss_fn(student, teacher)
    optimizer.zero_grad()
    loss.backward()
    if freeze_last:
        proto = getattr(getattr(dino, "student_head", None), "prototypes", None)
        if proto is not None:
            proto.weight.grad = None
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
    log: Callable[[str], None] | None = None,
    log_every: int = 50,
) -> list[float]:
    """Run ``steps`` DINO steps with LR / teacher-temp / momentum schedules (ref [1]).

    LR: linear warmup 0->base_lr over ``warmup_frac``, then cosine to ``lr_min``.
    Teacher temp: linear warmup over ``teacher_temp_warmup_frac``, then held. Default is a
    constant 0.04 — on real WM-811K a warmup to 0.07 left the teacher too soft to sharpen
    the small cosine-logit spread, pinning the loss at ln(out_dim) (uniform collapse).
    Teacher EMA momentum: cosine from ``momentum[0]`` to ``momentum[1]``.
    ``freeze_last_frac``: freeze the prototype layer for the first fraction of steps
    (DINO's stabilizer; prevents the early collapse-and-oscillate on real data).
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
    for step in range(1, steps + 1):
        lr = (
            base_lr * step / warmup
            if step <= warmup
            else _cosine(base_lr, lr_min, (step - warmup) / max(steps - warmup, 1))
        )
        for g in optimizer.param_groups:
            g["lr"] = lr
        loss_fn.teacher_temp = tt0 + (tt1 - tt0) * min(step / tt_warmup, 1.0)
        dino.teacher_momentum = _cosine(m0, m1, step / max(steps, 1))

        idx = rng.integers(0, n, size=bs)
        batch = [wafers[i] for i in idx]
        loss = train_step(
            dino,
            loss_fn,
            optimizer,
            batch,
            rng,
            orientation_invariant=orientation_invariant,
            device=device,
            freeze_last=step <= freeze_steps,
        )
        losses.append(loss)
        if log is not None and (step % log_every == 0 or step == steps):
            log(f"step {step}/{steps} loss={loss:.4f} lr={lr:.2e} ttemp={loss_fn.teacher_temp:.3f}")
    return losses


def embed_all(
    dino, wafers: list[TokenizedWafer], *, device: str = "cpu", batch_size: int = 256
) -> np.ndarray:
    """Embed all wafers with the (frozen) student encoder -> (N, D) L2-normalized."""
    import torch

    was_training = dino.training
    dino.eval()  # disable grad-checkpointing path during inference
    out: list[np.ndarray] = []
    with torch.no_grad():
        for i in range(0, len(wafers), batch_size):
            coords, states, mask = collate(wafers[i : i + batch_size])
            if device != "cpu":
                coords, states, mask = coords.to(device), states.to(device), mask.to(device)
            emb = dino.embed(coords, states, mask)
            out.append(emb.cpu().numpy())
    dino.train(was_training)
    return np.concatenate(out, axis=0) if out else np.zeros((0, 1), dtype=np.float32)
