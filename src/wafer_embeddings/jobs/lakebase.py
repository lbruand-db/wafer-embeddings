"""Fully automatic, idempotent Lakebase Search bootstrap (SPECS.md §9, §11.6, §11.8 / N7).

Every step checks current state first and only acts when needed, so the job is safe to
re-run at any point of the pipeline:

1. **Project** (Lakebase Autoscaling, Postgres >= 16) — create if missing.
2. **Lakebase Search** — ``POST /api/2.0/postgres/projects/{p}/search-extensions`` (the
   call behind the UI's *Enable Lakebase Search*; idempotent — re-posting returns done).
   It preloads ``lakebase_vector`` (and ``lakebase_text``) on the project's computes.
3. **Database** — create the Postgres database if missing (owned by the job's identity).
4. **Extension** — ``CREATE EXTENSION lakebase_vector CASCADE``. Never pgvector directly:
   ``CASCADE`` pulls in the ``vector`` *type* as lakebase_vector's own dependency, and all
   similarity search goes through the ``lakebase_ann`` index.
5. **Synced table** — UC Delta embeddings -> Postgres ``vector(dim)``; create if missing,
   recreate if failed, wait until online. Skipped (re-run later) if the source table
   does not exist yet, so this can run as part of bootstrap before any embedding.
6. **ANN index** — cosine ``lakebase_ann`` index + ``ANALYZE``.
7. **App grants** — read access for the search app's service principal, if the app exists.
8. **Verify** — row count, index info, and an EXPLAIN that must use the ANN index.

Decisions and SQL are pure functions (unit-tested); ``main`` is REST / Postgres glue.
"""

from __future__ import annotations

import argparse
import re
import time
from dataclasses import dataclass, replace

_IDENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*)?$")
_ROLE = re.compile(r"^[A-Za-z0-9@._+-]+$")


def _check_ident(name: str) -> str:
    if not _IDENT.match(name):
        raise ValueError(f"unsafe SQL identifier: {name!r}")
    return name


def _check_ident_3(name: str) -> str:
    """A UC ``catalog.schema.table`` name."""
    parts = name.split(".")
    if len(parts) != 3 or not all(_IDENT.match(x) for x in parts):
        raise ValueError(f"unsafe UC table name: {name!r}")
    return name


def _quote_role(role: str) -> str:
    """Double-quoted Postgres role (user e-mails / service-principal UUIDs)."""
    if not _ROLE.match(role):
        raise ValueError(f"unsafe Postgres role: {role!r}")
    return f'"{role}"'


@dataclass(frozen=True)
class LakebaseConfig:
    project: str = "wafer-embeddings"
    branch: str = "production"
    database_id: str = "wafer-embeddings"
    pg_database: str = "wafer_embeddings"
    pg_version: int = 17
    source_table: str = "mmf_mlops_demo_catalog.wafer_embeddings.wafer_map_embeddings"
    synced_table: str = "mmf_mlops_demo_catalog.wafer_embeddings.wafer_map_embeddings_pg"
    dim: int = 0  # embedding width D; 0 = detect from the source Delta table
    index: str = "wafer_map_embeddings_ann"
    build_mode: str = "standard"
    app_name: str = "wafer-search"
    refresh: bool = False  # re-snapshot an online synced table (after re-embedding)
    sync_timeout_s: int = 3600  # give up waiting for the synced table after this
    suspend_timeout_s: int = 3600  # scale computes to zero after 1 h idle (default is 24 h)

    @property
    def branch_name(self) -> str:
        return f"projects/{self.project}/branches/{self.branch}"

    @property
    def pg_table(self) -> str:
        """Postgres name of the synced table: ``<uc schema>.<uc table>``."""
        _, schema, table = self.synced_table.split(".")
        return _check_ident(f"{schema}.{table}")

    @property
    def pg_schema(self) -> str:
        return self.pg_table.split(".")[0]


def project_spec(cfg: LakebaseConfig) -> dict:
    return {
        "spec": {
            "display_name": cfg.project,
            "pg_version": cfg.pg_version,
            "default_endpoint_settings": {"suspend_timeout_duration": f"{cfg.suspend_timeout_s}s"},
        }
    }


def duration_seconds(d: str | None) -> int | None:
    """Parse a protobuf duration string like ``"3600s"`` (None if absent)."""
    if not d:
        return None
    return int(float(d.rstrip("s")))


def suspend_updates(project: dict, endpoints: list[dict], seconds: int) -> list[tuple]:
    """``(path, update_mask, body)`` PATCHes so the project default and every compute
    suspend after ``seconds`` idle. Empty when already set (idempotent)."""
    want = f"{seconds}s"
    out = []
    cur = ((project.get("status") or {}).get("default_endpoint_settings") or {}).get(
        "suspend_timeout_duration"
    )
    if duration_seconds(cur) != seconds:
        out.append(
            (
                project["name"],
                "spec.default_endpoint_settings.suspend_timeout_duration",
                {"spec": {"default_endpoint_settings": {"suspend_timeout_duration": want}}},
            )
        )
    for ep in endpoints:
        cur = (ep.get("status") or {}).get("suspend_timeout_duration")
        if duration_seconds(cur) != seconds:
            out.append(
                (
                    ep["name"],
                    "spec.suspend_timeout_duration",
                    {"spec": {"suspend_timeout_duration": want}},
                )
            )
    return out


def synced_table_spec(cfg: LakebaseConfig) -> dict:
    """UC synced table: Delta embeddings -> Postgres, embedding typed ``vector(dim)``."""
    catalog, schema, _ = cfg.source_table.split(".")
    return {
        "spec": {
            "source_table_full_name": cfg.source_table,
            "primary_key_columns": ["id"],
            "scheduling_policy": "SNAPSHOT",
            "branch": cfg.branch_name,
            "postgres_database": cfg.pg_database,
            "create_database_objects_if_missing": True,
            "type_overrides": [
                {"column_name": "embedding", "pg_type": "PG_SPECIFIC_TYPE_VECTOR", "size": cfg.dim}
            ],
            "new_pipeline_spec": {"storage_catalog": catalog, "storage_schema": schema},
        }
    }


def synced_table_action(state: str | None, created: bool = False) -> str:
    """What to do given the synced table's ``detailed_state`` (None = not found).

    Any ``SYNCED_TABLE_ONLINE*`` state is serving data and is never deleted — including
    ``ONLINE_PIPELINE_FAILED`` (only its last refresh failed). Only offline failures are
    recreated. ``created``: we already POSTed a create this run, so "not found" means
    "not visible yet" (wait), not "create again".
    """
    if state is None:
        return "wait" if created else "create"
    if state.startswith("SYNCED_TABLE_ONLINE"):
        return "ok"
    if "FAILED" in state or "ERROR" in state:
        return "recreate"
    return "wait"


def drop_table_sql(table: str) -> list[str]:
    """Remove a leftover Postgres table before recreating a failed synced table."""
    return [f"DROP TABLE IF EXISTS {_check_ident(table)} CASCADE"]


def is_preload_not_ready(err: Exception) -> bool:
    """The one transient error worth retrying: Search preload not active yet (restart)."""
    return "shared_preload_libraries" in str(err) or "the database system is" in str(err)


def extension_sql() -> list[str]:
    # CASCADE installs lakebase_vector's dependencies (the `vector` type) — no direct pgvector
    return ["CREATE EXTENSION IF NOT EXISTS lakebase_vector CASCADE"]


def index_sql(table: str, index: str, build_mode: str = "standard") -> list[str]:
    """The cosine ``lakebase_ann`` index (§9) + statistics."""
    _check_ident(table)
    _check_ident(index)
    if build_mode not in ("standard", "quality"):
        raise ValueError("build_mode must be 'standard' or 'quality'")
    return [
        f"CREATE INDEX IF NOT EXISTS {index} ON {table} USING lakebase_ann "
        f"(embedding vector_cosine_ops) WITH (build_mode = '{build_mode}')",
        f"ANALYZE {table}",
    ]


def grant_sql(schema: str, role: str) -> list[str]:
    """Read-only access for a role (the app's service principal) on a schema."""
    _check_ident(schema)
    r = _quote_role(role)
    return [
        f"GRANT USAGE ON SCHEMA {schema} TO {r}",
        f"GRANT SELECT ON ALL TABLES IN SCHEMA {schema} TO {r}",
        f"ALTER DEFAULT PRIVILEGES IN SCHEMA {schema} GRANT SELECT ON TABLES TO {r}",
    ]


def verify_sql(table: str, index: str) -> list[str]:
    """Checks: row count, index metadata, and the plan of a top-k query."""
    _check_ident(table)
    _check_ident(index)
    # an index lives in its table's schema; regclass needs it qualified
    qualified = f"{table.rsplit('.', 1)[0]}.{index}" if "." in table else index
    probe = f"(SELECT embedding FROM {table} ORDER BY id LIMIT 1)"
    return [
        f"SELECT count(*) FROM {table}",
        f"SELECT lakebase_ann_index_info('{qualified}'::regclass)",
        f"EXPLAIN SELECT id FROM {table} ORDER BY embedding <=> {probe} LIMIT 10",
    ]


def dim_sql(table: str) -> str:
    """Embedding widths present in a Delta table (Spark SQL)."""
    return (
        f"SELECT min(size(embedding)) AS lo, max(size(embedding)) AS hi "
        f"FROM {_check_ident_3(table)}"
    )


def check_dim(lo, hi) -> int:
    """The single embedding width D of a table, or an error if absent / mixed."""
    if lo is None or hi is None or int(lo) <= 0:
        raise ValueError("no embeddings to infer the vector width from")
    if int(lo) != int(hi):
        raise ValueError(f"mixed embedding widths {lo}..{hi}: re-embed with one model")
    return int(lo)


def pg_dim_sql(table: str) -> str:
    """Width of the synced Postgres ``vector`` column (one row is enough)."""
    return f"SELECT vector_dims(embedding) FROM {_check_ident(table)} LIMIT 1"


def plan_uses_index(plan_lines: list[str], index: str) -> bool:
    """True if the top-k query is an ANN index scan ordered by ``embedding <=>``.

    Synced tables are partitioned, so the plan names the per-partition child index
    (``partition_<n>_embedding_idx``) rather than ``index``; accept either.
    """
    for i, line in enumerate(plan_lines):
        if "Index Scan" in line and (index in line or "embedding_idx" in line):
            nxt = plan_lines[i + 1] if i + 1 < len(plan_lines) else ""
            if "Order By" in nxt and "embedding <=>" in nxt:
                return True
    return False


def owner_role(roles: list[dict], user_name: str) -> str | None:
    """Resource name of the Postgres role mapped to ``user_name`` (the job identity)."""
    for r in roles:
        st = r.get("status") or {}
        if user_name in (st.get("postgres_role"), r.get("role_id"), st.get("identity")):
            return r["name"]
    return None


# --------------------------------------------------------------------------- glue
class _Rest:  # pragma: no cover - needs a workspace
    """Minimal Lakebase (``/api/2.0/postgres``) REST client with LRO waiting."""

    def __init__(self, w, log):
        self.w, self.log = w, log

    def call(self, method: str, path: str, body: dict | None = None, query: dict | None = None):
        return self.w.api_client.do(method, f"/api/2.0/postgres/{path}", body=body, query=query)

    def get_or_none(self, path: str):
        from databricks.sdk.errors import NotFound  # ty: ignore[unresolved-import]

        try:
            return self.call("GET", path)
        except NotFound:
            return None

    def wait(self, op: dict, what: str, timeout_s: int = 1800) -> dict:
        t0 = time.time()
        if not isinstance(op, dict) or "name" not in op:
            return op or {}  # a plain response, not a long-running operation
        while not op.get("done"):
            if time.time() - t0 > timeout_s:
                raise TimeoutError(f"{what}: operation {op.get('name')} not done")
            time.sleep(5)
            op = self.call("GET", op["name"])
        if op.get("error"):
            raise RuntimeError(f"{what} failed: {op['error']}")
        return op


def _spark():  # pragma: no cover - Databricks runtime only
    try:
        from databricks.sdk.runtime import spark  # type: ignore

        return spark
    except Exception as e:  # noqa: BLE001
        raise RuntimeError("width detection needs Spark: run as a job or pass --dim") from e


def _psql(w, cfg: LakebaseConfig):  # pragma: no cover
    import psycopg  # ty: ignore[unresolved-import]

    eps = list(w.postgres.list_endpoints(cfg.branch_name))
    ep = next(e for e in eps if (e.status and e.status.hosts))
    token = w.postgres.generate_database_credential(endpoint=ep.name).token
    return psycopg.connect(
        host=ep.status.hosts.host,
        dbname=cfg.pg_database,
        user=w.current_user.me().user_name,
        password=token,
        sslmode="require",
        autocommit=True,
    )


def _run(conn, stmts: list[str], log) -> list[list[tuple]]:  # pragma: no cover
    out = []
    for s in stmts:
        cur = conn.execute(s)
        rows = cur.fetchall() if cur.description else []
        log(f"SQL ok: {s[:110]}")
        out.append(rows)
    return out


def bootstrap(w, cfg: LakebaseConfig, log) -> dict:  # pragma: no cover - needs Lakebase
    rest = _Rest(w, log)
    p = f"projects/{cfg.project}"
    report: dict = {}

    # 1. project
    if rest.get_or_none(p) is None:
        op = rest.call(
            "POST", "projects", body=project_spec(cfg), query={"project_id": cfg.project}
        )
        rest.wait(op, "create project")
        report["project"] = "created"
    else:
        report["project"] = "exists"
    log(f"project {cfg.project}: {report['project']}")

    # 1b. compute: scale to zero after cfg.suspend_timeout_s idle (project default + computes)
    eps = rest.call("GET", f"{cfg.branch_name}/endpoints").get("endpoints", [])
    updates = suspend_updates(rest.call("GET", p), eps, cfg.suspend_timeout_s)
    for path, mask, body in updates:
        rest.wait(
            rest.call("PATCH", path, body=body, query={"update_mask": mask}), f"update {path}"
        )
        log(f"set {mask} = {cfg.suspend_timeout_s}s on {path}")
    report["suspend_timeout"] = f"{cfg.suspend_timeout_s}s ({len(updates)} updated)"

    # 2. Lakebase Search (idempotent; restarts computes only the first time)
    rest.wait(rest.call("POST", f"{p}/search-extensions", body={}), "enable Lakebase Search")
    report["search"] = "enabled"

    # 3. database
    dbs = rest.call("GET", f"{cfg.branch_name}/databases").get("databases", [])
    if not any((d.get("status") or {}).get("postgres_database") == cfg.pg_database for d in dbs):
        roles = rest.call("GET", f"{cfg.branch_name}/roles").get("roles", [])
        role = owner_role(roles, w.current_user.me().user_name)
        if role is None:
            raise RuntimeError("no Postgres role for the job identity on the branch")
        op = rest.call(
            "POST",
            f"{cfg.branch_name}/databases",
            body={"spec": {"postgres_database": cfg.pg_database, "role": role}},
            query={"database_id": cfg.database_id},
        )
        rest.wait(op, "create database")
        report["database"] = "created"
    else:
        report["database"] = "exists"
    log(f"database {cfg.pg_database}: {report['database']}")

    # 4. extension (needs the Search preload; retry while computes restart)
    for attempt in range(12):
        try:
            with _psql(w, cfg) as conn:
                _run(conn, extension_sql(), log)
            break
        except Exception as e:  # only the restart-after-enable window is transient
            if attempt == 11 or not is_preload_not_ready(e):
                raise
            log(f"Lakebase Search preload not active yet ({e}); retrying")
            time.sleep(10)
    report["extension"] = "lakebase_vector"

    # 5. synced table (skip until the source embeddings exist)
    if not w.tables.exists(cfg.source_table).table_exists:
        report["synced_table"] = "skipped: source table missing (re-run after embedding)"
        log(report["synced_table"])
        return report
    if not cfg.dim:  # width follows the embeddings, not a hand-set variable
        lo, hi = _spark().sql(dim_sql(cfg.source_table)).collect()[0]
        cfg = replace(cfg, dim=check_dim(lo, hi))
        log(f"embedding width D = {cfg.dim} (from {cfg.source_table})")
    report["dim"] = cfg.dim
    st_path = f"synced_tables/{cfg.synced_table}"
    created, recreated, deadline = False, False, time.time() + cfg.sync_timeout_s
    while True:
        if time.time() > deadline:
            raise TimeoutError(f"synced table {cfg.synced_table} not online in time")
        st = rest.get_or_none(st_path)
        state = ((st or {}).get("status") or {}).get("detailed_state") if st else None
        action = synced_table_action(state, created=created)
        if action == "ok" and not created:  # a width change (e.g. 128 -> 384) needs a rebuild
            with _psql(w, cfg) as conn:
                pg_dim = (conn.execute(pg_dim_sql(cfg.pg_table)).fetchone() or [None])[0]
            if pg_dim is not None and int(pg_dim) != cfg.dim:
                log(f"synced table is vector({pg_dim}) but embeddings are {cfg.dim}-d: rebuild")
                action = "recreate"
        log(f"synced table {cfg.synced_table}: state={state} -> {action}")
        if action == "ok":
            break
        if action == "recreate":
            if recreated:  # failed again after a clean rebuild: not transient, stop
                msg = ((st or {}).get("status") or {}).get("message")
                raise RuntimeError(f"synced table failed again after recreate: {msg}")
            rest.wait(rest.call("DELETE", st_path), "delete failed synced table")
            with _psql(w, cfg) as conn:  # a leftover Postgres table would block the rebuild
                _run(conn, drop_table_sql(cfg.pg_table), log)
            recreated, created = True, False
            continue
        if action == "create":
            rest.call(
                "POST",
                "synced_tables",
                body=synced_table_spec(cfg),
                query={"synced_table_id": cfg.synced_table},
            )
            created = True
        time.sleep(30)
    report["synced_table"] = "online"
    if cfg.refresh and created:
        log("synced table was just created: its initial snapshot is fresh, skipping refresh")
    elif cfg.refresh:
        pid = ((st or {}).get("status") or {}).get("pipeline_id")
        if not pid:
            raise RuntimeError("synced table has no pipeline_id to refresh")
        upd = w.pipelines.start_update(pid).update_id
        log(f"refreshing synced table: pipeline {pid} update {upd}")
        while True:
            state = str(w.pipelines.get_update(pid, upd).update.state)
            if state.endswith(("COMPLETED", "FAILED", "CANCELED")):
                break
            time.sleep(20)
        if not state.endswith("COMPLETED"):
            raise RuntimeError(f"synced table refresh {upd} ended {state}")
        report["synced_table"] = f"online (refreshed, update {upd})"

    with _psql(w, cfg) as conn:
        # 6. ANN index
        _run(conn, index_sql(cfg.pg_table, cfg.index, cfg.build_mode), log)
        report["index"] = cfg.index
        # 7. app grants
        from databricks.sdk.errors import NotFound  # ty: ignore[unresolved-import]

        try:
            sp = w.apps.get(cfg.app_name).service_principal_client_id
        except NotFound:  # no app (yet): nothing to grant; other errors must surface
            sp = None
            log(f"app {cfg.app_name} not found: skipping grants")
        if sp:
            _run(conn, grant_sql(cfg.pg_schema, sp), log)
            report["grants"] = f"SELECT on {cfg.pg_schema} for app {cfg.app_name}"
        # 8. verify
        count, info, plan = _run(conn, verify_sql(cfg.pg_table, cfg.index), log)
        lines = [r[0] for r in plan]
        report["rows"] = count[0][0]
        pg_dim = conn.execute(pg_dim_sql(cfg.pg_table)).fetchone()[0]
        if int(pg_dim) != cfg.dim:
            raise RuntimeError(f"synced vector({pg_dim}) != embedding width {cfg.dim}")
        report["index_info"] = info[0][0] if info else None
        report["plan_uses_index"] = plan_uses_index(lines, cfg.index)
        if not report["plan_uses_index"]:
            raise RuntimeError(
                "top-k query does not use the lakebase_ann index:\n" + "\n".join(lines)
            )
    return report


def _args(argv=None):
    d = LakebaseConfig()
    p = argparse.ArgumentParser(description="Idempotent Lakebase Search bootstrap.")
    p.add_argument("--project", default=d.project)
    p.add_argument("--branch", default=d.branch)
    p.add_argument("--database-id", default=d.database_id)
    p.add_argument("--pg-database", default=d.pg_database)
    p.add_argument("--source-table", default=d.source_table)
    p.add_argument("--synced-table", default=d.synced_table)
    p.add_argument(
        "--dim", type=int, default=d.dim, help="Embedding width (0 = detect from the source)."
    )
    p.add_argument("--index", default=d.index)
    p.add_argument("--build-mode", default=d.build_mode, choices=["standard", "quality"])
    p.add_argument("--app-name", default=d.app_name)
    p.add_argument(
        "--suspend-timeout-s",
        type=int,
        default=d.suspend_timeout_s,
        help="Scale Lakebase computes to zero after this many idle seconds.",
    )
    p.add_argument(
        "--refresh",
        action="store_true",
        help="Re-snapshot an already-online synced table (use after re-embedding).",
    )
    a = p.parse_args(argv)
    return LakebaseConfig(
        project=a.project,
        branch=a.branch,
        database_id=a.database_id,
        pg_database=a.pg_database,
        source_table=a.source_table,
        synced_table=a.synced_table,
        dim=a.dim,
        index=a.index,
        build_mode=a.build_mode,
        app_name=a.app_name,
        refresh=a.refresh,
        suspend_timeout_s=a.suspend_timeout_s,
    )


def main(argv=None) -> None:  # pragma: no cover - needs a workspace
    import json

    from databricks.sdk import WorkspaceClient  # ty: ignore[unresolved-import]

    from wafer_embeddings.obs import get_logger

    log = get_logger("wafer_embeddings.lakebase")
    report = bootstrap(WorkspaceClient(), _args(argv), log.info)
    log.info("lakebase bootstrap: " + json.dumps(report, default=str))


if __name__ == "__main__":  # pragma: no cover
    main()
