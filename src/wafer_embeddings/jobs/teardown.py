"""Tear down what the jobs create outside the bundle (SPECS.md §11.8 / N7, GAPS 4.5).

``databricks bundle destroy`` removes what the bundle owns (jobs, the app). The rest is
created by code — the serving endpoint (``jobs/serve.py``), the Lakebase project and
synced table (``jobs/lakebase.py``), Delta tables, volume files and the UC model — and is
removed here, in dependency order:

- ``--scope serving``: everything rebuildable from the ingested data — endpoint, synced
  table, Lakebase project, the embeddings Delta table and its parquet export. Enough to
  stop all serving / indexing cost; ``bootstrap`` + ``publish`` rebuild it.
- ``--scope all``: also the UC model, ``wafer_maps`` (+ parquet export), the volume and
  the schema. The catalog is never dropped.

**Dry run by default**: prints the plan and what exists; ``--yes`` deletes. Every step
skips targets that are already gone, so it is safe to re-run.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass


@dataclass(frozen=True)
class Step:
    kind: str  # endpoint | synced_table | lakebase_project | table | volume_dir | model | volume | schema
    target: str

    def __str__(self) -> str:
        return f"{self.kind:16s} {self.target}"


def plan(
    catalog: str,
    schema: str,
    volume: str = "raw",
    scope: str = "serving",
    endpoint: str = "wafer-encoder",
    project: str = "wafer-embeddings",
    model: str = "wafer_encoder",
) -> list[Step]:
    """Ordered teardown steps: consumers before what they read from."""
    if scope not in ("serving", "all"):
        raise ValueError("scope must be 'serving' or 'all'")
    fq = f"{catalog}.{schema}"
    vol = f"/Volumes/{catalog}/{schema}/{volume}"
    steps = [
        Step("endpoint", endpoint),
        Step("synced_table", f"{fq}.wafer_map_embeddings_pg"),
        Step("lakebase_project", project),
        Step("table", f"{fq}.wafer_map_embeddings"),
        Step("volume_dir", f"{vol}/wafer_map_embeddings_parquet"),
    ]
    if scope == "all":
        steps += [
            Step("model", f"{fq}.{model}"),
            Step("table", f"{fq}.wafer_maps"),
            Step("volume_dir", f"{vol}/wafer_maps_parquet"),
            Step("volume", f"{fq}.{volume}"),
            Step("schema", fq),
        ]
    return steps


def _args(argv=None):
    p = argparse.ArgumentParser(
        description="Tear down code-managed resources (dry run by default)."
    )
    p.add_argument("--catalog", required=True)
    p.add_argument("--schema", required=True)
    p.add_argument("--volume", default="raw")
    p.add_argument("--scope", default="serving", choices=["serving", "all"])
    p.add_argument("--endpoint", default="wafer-encoder")
    p.add_argument("--project", default="wafer-embeddings")
    p.add_argument("--model", default="wafer_encoder")
    p.add_argument("--yes", action="store_true", help="Actually delete (default: dry run).")
    return p.parse_args(argv)


def _exists_and_delete(w, step: Step, delete: bool) -> str:  # pragma: no cover - workspace
    from databricks.sdk.errors import NotFound  # ty: ignore[unresolved-import]

    k, t = step.kind, step.target
    try:
        if k == "endpoint":
            w.serving_endpoints.get(t)
            if delete:
                w.serving_endpoints.delete(t)
        elif k == "synced_table":
            w.api_client.do("GET", f"/api/2.0/postgres/synced_tables/{t}")
            if delete:
                w.api_client.do("DELETE", f"/api/2.0/postgres/synced_tables/{t}")
        elif k == "lakebase_project":
            w.api_client.do("GET", f"/api/2.0/postgres/projects/{t}")
            if delete:
                w.api_client.do("DELETE", f"/api/2.0/postgres/projects/{t}")
        elif k == "table":
            if not w.tables.exists(t).table_exists:
                return "absent"
            if delete:
                w.tables.delete(t)
        elif k == "volume_dir":
            entries = list(w.files.list_directory_contents(t))
            if delete:
                _rm_tree(w, t, entries)
        elif k == "model":
            w.registered_models.get(t)
            if delete:
                for v in w.model_versions.list(t):
                    w.model_versions.delete(t, v.version)
                w.registered_models.delete(t)
        elif k == "volume":
            w.volumes.read(t)
            if delete:
                w.volumes.delete(t)
        elif k == "schema":
            w.schemas.get(t)
            if delete:
                w.schemas.delete(t, force=True)
        else:
            raise ValueError(k)
    except NotFound:
        return "absent"
    return "deleted" if delete else "exists (would delete)"


def _rm_tree(w, path: str, entries) -> None:  # pragma: no cover - workspace
    for e in entries:
        if e.is_directory:
            _rm_tree(w, e.path, list(w.files.list_directory_contents(e.path)))
        else:
            w.files.delete(e.path)
    w.files.delete_directory(path)


def main(argv=None) -> None:  # pragma: no cover - needs a workspace
    from databricks.sdk import WorkspaceClient  # ty: ignore[unresolved-import]

    a = _args(argv)
    w = WorkspaceClient()
    steps = plan(a.catalog, a.schema, a.volume, a.scope, a.endpoint, a.project, a.model)
    print(f"teardown scope={a.scope} ({'DELETING' if a.yes else 'dry run; pass --yes to delete'})")
    for step in steps:
        print(f"  {step}  -> {_exists_and_delete(w, step, a.yes)}")
    print("then remove the bundle-owned jobs + app with:  databricks bundle destroy")


if __name__ == "__main__":  # pragma: no cover
    main()
