"""Device-agnostic DINO training core (the Databricks GPU job wraps this)."""

from wafer_embeddings.train.evaluate import g1_metrics, selection_score, stratified_indices
from wafer_embeddings.train.trainer import (
    build_views,
    collapse_stats,
    embed_all,
    fit_dino,
    param_groups,
    scaled_lr,
    train_step,
    wafers_from_rows,
)

__all__ = [
    "build_views",
    "collapse_stats",
    "train_step",
    "fit_dino",
    "param_groups",
    "scaled_lr",
    "embed_all",
    "wafers_from_rows",
    "g1_metrics",
    "selection_score",
    "stratified_indices",
]
