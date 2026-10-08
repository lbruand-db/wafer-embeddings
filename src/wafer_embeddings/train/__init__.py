"""Device-agnostic DINO training core (the Databricks GPU job wraps this)."""

from wafer_embeddings.train.evaluate import g1_metrics, stratified_indices
from wafer_embeddings.train.trainer import (
    build_views,
    embed_all,
    fit_dino,
    train_step,
    wafers_from_rows,
)

__all__ = [
    "build_views",
    "train_step",
    "fit_dino",
    "embed_all",
    "wafers_from_rows",
    "g1_metrics",
    "stratified_indices",
]
