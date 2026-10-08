"""DINO training core (SPECS.md §4/§7).

Pure of Databricks/MLflow/GPU specifics so it is unit-testable on CPU; the training
job (P3/P4) wraps this with data loading, device placement, MLflow, and the AI
Runtime serverless-GPU environment.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from wafer_embeddings.tokenize.augment import random_view
from wafer_embeddings.tokenize.tokenizer import TokenizedWafer, collate, tokenize


def wafers_from_rows(rows) -> tuple[list[TokenizedWafer], np.ndarray, np.ndarray]:
    """Reconstruct + tokenize wafers from Delta row dicts.

    Each row has a flattened ``wafer_map`` plus ``height``/``width``; returns the
    tokenized wafers alongside aligned ``label_id`` and ``split`` arrays.
    """
    wafers: list[TokenizedWafer] = []
    labels: list[int] = []
    splits: list[str] = []
    for r in rows:
        h, w = int(r["height"]), int(r["width"])
        m = np.asarray(r["wafer_map"], dtype=np.uint8).reshape(h, w)
        wafers.append(tokenize(m))
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
) -> float:
    """One DINO optimization step over a batch of tokenized wafers."""
    views = build_views(wafers, rng, n_views=2, orientation_invariant=orientation_invariant)
    views = _to_device(views, device)
    student = dino.student_views(views)
    teacher = dino.teacher_views(views)
    loss = loss_fn(student, teacher)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    dino.update_teacher()
    return float(loss.detach())


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
    log: Callable[[str], None] | None = None,
    log_every: int = 50,
) -> list[float]:
    """Run ``steps`` DINO optimization steps, sampling random batches each step."""
    n = len(wafers)
    bs = min(batch_size, n)
    losses: list[float] = []
    for step in range(1, steps + 1):
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
        )
        losses.append(loss)
        if log is not None and (step % log_every == 0 or step == steps):
            log(f"step {step}/{steps} loss={loss:.4f}")
    return losses


def embed_all(
    dino, wafers: list[TokenizedWafer], *, device: str = "cpu", batch_size: int = 256
) -> np.ndarray:
    """Embed all wafers with the (frozen) student encoder -> (N, D) L2-normalized."""
    import torch

    out: list[np.ndarray] = []
    with torch.no_grad():
        for i in range(0, len(wafers), batch_size):
            coords, states, mask = collate(wafers[i : i + batch_size])
            if device != "cpu":
                coords, states, mask = coords.to(device), states.to(device), mask.to(device)
            emb = dino.embed(coords, states, mask)
            out.append(emb.cpu().numpy())
    return np.concatenate(out, axis=0) if out else np.zeros((0, 1), dtype=np.float32)
