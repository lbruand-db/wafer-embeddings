"""Device-agnostic DINO training core (the Databricks GPU job wraps this)."""

from wafer_embeddings.train.trainer import build_views, train_step

__all__ = ["build_views", "train_step"]
