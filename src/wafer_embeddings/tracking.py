"""MLflow on Databricks: one place for the tracking / registry setup and experiment path.

Jobs on Databricks pick these up from the environment, but local runs (tools, a laptop
``register``) would otherwise fall back to a local ``./mlflow.db`` and the workspace
model registry; ``use_databricks`` makes every caller hit the workspace + Unity Catalog.
"""

from __future__ import annotations

EXPERIMENT_NAME = "wafer-embeddings"  # ai_runtime/*.yaml experiment_name


def use_databricks() -> None:
    """Point MLflow at the workspace tracking server and the Unity Catalog registry."""
    import mlflow  # ty: ignore[unresolved-import]

    mlflow.set_tracking_uri("databricks")
    mlflow.set_registry_uri("databricks-uc")


def experiment_path(user_name: str, name: str = EXPERIMENT_NAME) -> str:
    """The experiment AI Runtime creates for ``user_name`` (``mlflow_experiment_directory``
    is the user's home)."""
    return f"/Users/{user_name}/{name}"


def default_experiment() -> str:  # pragma: no cover - needs workspace auth
    """The current user's ``wafer-embeddings`` experiment."""
    from databricks.sdk import WorkspaceClient  # ty: ignore[unresolved-import]

    return experiment_path(WorkspaceClient().current_user.me().user_name or "")
