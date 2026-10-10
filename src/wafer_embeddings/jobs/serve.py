"""Point the Model Serving endpoint at the UC model's ``@champion`` version (SPECS.md §10).

Model Serving needs a concrete model version (no aliases), so a bundle-declared endpoint
would pin it in a hand-set variable. Instead this step resolves the alias at run time and
creates or updates the endpoint to serve exactly that version — idempotent: when the
endpoint already serves it with the same settings, nothing changes.

Decisions are pure functions (unit-tested); ``main`` is SDK glue.
"""

from __future__ import annotations

import argparse


def desired_entity(
    model: str, version: str, workload_size: str = "Small", scale_to_zero: bool = True
) -> dict:
    """The single served entity the endpoint should run."""
    return {
        "name": model.split(".")[-1].replace("_", "-"),
        "entity_name": model,
        "entity_version": str(version),
        "workload_size": workload_size,
        "scale_to_zero_enabled": scale_to_zero,
    }


def serve_action(current: list[dict] | None, desired: dict) -> str:
    """``create`` (no endpoint), ``update`` (serves something else) or ``ok``."""
    if current is None:
        return "create"
    keys = ("entity_name", "entity_version", "workload_size", "scale_to_zero_enabled")
    same = len(current) == 1 and all(str(current[0].get(k)) == str(desired[k]) for k in keys)
    return "ok" if same else "update"


def _args(argv=None):
    p = argparse.ArgumentParser(description="Serve the UC model's alias on an endpoint.")
    p.add_argument("--catalog", required=True)
    p.add_argument("--schema", required=True)
    p.add_argument("--model", default="wafer_encoder")
    p.add_argument("--alias", default="champion")
    p.add_argument("--endpoint", default="wafer-encoder")
    p.add_argument("--workload-size", default="Small", choices=["Small", "Medium", "Large"])
    p.add_argument(
        "--scale-to-zero", action=argparse.BooleanOptionalAction, default=True, help="Idle -> 0."
    )
    p.add_argument("--timeout-min", type=int, default=45)
    return p.parse_args(argv)


def main(argv=None) -> None:  # pragma: no cover - needs a workspace
    import datetime
    import json

    from databricks.sdk import WorkspaceClient  # ty: ignore[unresolved-import]
    from databricks.sdk.errors import NotFound  # ty: ignore[unresolved-import]
    from databricks.sdk.service.serving import (  # ty: ignore[unresolved-import]
        EndpointCoreConfigInput,
        ServedEntityInput,
    )

    from wafer_embeddings.obs import get_logger

    a = _args(argv)
    log = get_logger("wafer_embeddings.serve")
    w = WorkspaceClient()
    fqn = f"{a.catalog}.{a.schema}.{a.model}"
    version = w.model_versions.get_by_alias(fqn, a.alias).version
    want = desired_entity(fqn, str(version), a.workload_size, a.scale_to_zero)
    try:
        ep = w.serving_endpoints.get(a.endpoint)
        cfg = ep.config or ep.pending_config
        current = [e.as_dict() for e in (cfg.served_entities or [])] if cfg else []
    except NotFound:
        current = None
    action = serve_action(current, want)
    log.info(f"{fqn}@{a.alias} = v{version}; endpoint {a.endpoint}: {action}")
    timeout = datetime.timedelta(minutes=a.timeout_min)
    entity = ServedEntityInput(**want)
    if action == "create":
        w.serving_endpoints.create_and_wait(
            a.endpoint,
            config=EndpointCoreConfigInput(name=a.endpoint, served_entities=[entity]),
            timeout=timeout,
        )
    elif action == "update":
        w.serving_endpoints.update_config_and_wait(
            a.endpoint, served_entities=[entity], timeout=timeout
        )
    state = w.serving_endpoints.get(a.endpoint).state
    log.info(
        "serve: "
        + json.dumps(
            {"endpoint": a.endpoint, "version": str(version), "action": action, "state": str(state)}
        )
    )


if __name__ == "__main__":  # pragma: no cover
    main()
