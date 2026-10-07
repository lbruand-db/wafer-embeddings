"""DINO training core (SPECS.md §4/§7).

Pure of Databricks/MLflow/GPU specifics so it is unit-testable on CPU; the training
job (P3/P4) wraps this with data loading, device placement, MLflow, and the AI
Runtime serverless-GPU environment.
"""

from __future__ import annotations

import numpy as np

from wafer_embeddings.tokenize.augment import random_view
from wafer_embeddings.tokenize.tokenizer import TokenizedWafer, collate


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
) -> float:
    """One DINO optimization step over a batch of tokenized wafers."""
    views = build_views(wafers, rng, n_views=2, orientation_invariant=orientation_invariant)
    student = dino.student_views(views)
    teacher = dino.teacher_views(views)
    loss = loss_fn(student, teacher)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    dino.update_teacher()
    return float(loss.detach())
