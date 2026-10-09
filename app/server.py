"""Wafer-map similarity search — FastAPI backend for the Vue frontend (Databricks App).

JSON API over Lakebase Search (``lakebase_ann`` cosine index) and the ``wafer-encoder``
Model Serving endpoint, plus the built Vue app (``frontend/dist``) as static files. The
browser never talks to Postgres or the endpoint: credentials stay server-side.

Database and endpoint access are injected (``create_app(db, embed)``), so the API is
tested in CI with fakes; ``main`` wires the real Lakebase / serving clients.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Callable, Protocol

from fastapi import FastAPI, HTTPException, Query
from fastapi.staticfiles import StaticFiles

import wafer_search as ws

CLASSES = [
    "Center",
    "Donut",
    "Edge-Loc",
    "Edge-Ring",
    "Loc",
    "Near-full",
    "Random",
    "Scratch",
    "none",
]
DIST = Path(__file__).resolve().parent / "frontend" / "dist"
_WAFER_COLS = ("id", "label", "split", "lot", "height", "width", "map_code")


class Db(Protocol):
    def query(self, sql: str, params: dict) -> list[tuple]: ...


def _wafer(row: tuple) -> dict:
    return dict(zip(_WAFER_COLS, row[: len(_WAFER_COLS)]))


def create_app(
    db: Db,
    embed: Callable[[list[list[int]]], list[float]] | None,
    table: str = ws.TABLE,
    endpoint_name: str = "",
    static_dir: Path | None = DIST,
) -> FastAPI:
    app = FastAPI(title="Wafer-map search", docs_url="/api/docs", openapi_url="/api/openapi.json")

    def _query_wafer(wafer_id: int | None, label: str | None) -> tuple:
        if wafer_id is not None:
            rows = db.query(ws.by_id_sql(table), {"qid": wafer_id})
        else:
            rows = db.query(ws.sample_sql(table, with_label=label is not None), {"label": label})
        if not rows:
            raise HTTPException(404, "no wafer found")
        return rows[0]

    @app.get("/api/health")
    def health() -> dict:
        return {"status": "ok"}

    @app.get("/api/config")
    def config() -> dict:
        return {"classes": CLASSES, "endpoint": endpoint_name, "live_embedding": embed is not None}

    @app.get("/api/wafer")
    def wafer(id: int | None = None, label: str | None = None) -> dict:
        if label is not None and label not in CLASSES:
            raise HTTPException(422, f"unknown class {label!r}")
        return _wafer(_query_wafer(id, label))

    @app.get("/api/search")
    def search(
        id: int,
        k: int = Query(12, ge=1, le=48),
        cross_lot: bool = True,
        live: bool = False,
    ) -> dict:
        row = _query_wafer(id, None)
        q = _wafer(row)
        if live:
            if embed is None:
                raise HTTPException(400, "live embedding is not configured")
            vec = embed(ws.decode_map(q["map_code"]).tolist())
            source = "live"
        else:
            vec = ws.parse_vector(row[len(_WAFER_COLS)])
            source = "stored"
        hits = db.query(
            ws.neighbours_sql(table, exclude_same_lot=cross_lot),
            {"q": ws.vector_literal(vec), "qid": q["id"], "lot": q["lot"], "k": k},
        )
        neighbours = [_wafer(h) | {"similarity": float(h[len(_WAFER_COLS)])} for h in hits]
        labeled = [n for n in neighbours if n["label"]]  # most of WM-811K is unlabeled
        same = sum(1 for n in labeled if q["label"] and n["label"] == q["label"])
        return {
            "query": q,
            "embedding": source,
            "cross_lot": cross_lot,
            "same_class": same,
            "labeled_neighbours": len(labeled),
            "neighbours": neighbours,
        }

    if static_dir is not None and static_dir.is_dir():
        app.mount("/", StaticFiles(directory=str(static_dir), html=True), name="frontend")
    return app


class LakebaseDb:  # pragma: no cover - needs a Lakebase project
    """Postgres access with a fresh OAuth credential per connection (tokens expire)."""

    def __init__(self, w, endpoint: str):
        self.w, self.endpoint = w, endpoint

    def query(self, sql: str, params: dict) -> list[tuple]:
        import psycopg  # ty: ignore[unresolved-import]

        password = self.w.postgres.generate_database_credential(endpoint=self.endpoint).token
        with psycopg.connect(
            host=os.environ["PGHOST"],
            port=int(os.environ.get("PGPORT", "5432")),
            dbname=os.environ.get("PGDATABASE", "wafer_embeddings"),
            user=os.environ.get("PGUSER") or self.w.current_user.me().user_name,
            password=password,
            sslmode=os.environ.get("PGSSLMODE", "require"),
            autocommit=True,
        ) as conn:
            return conn.execute(sql, params).fetchall()


def main() -> None:  # pragma: no cover - runs inside the Databricks App
    import uvicorn  # ty: ignore[unresolved-import]
    from databricks.sdk import WorkspaceClient  # ty: ignore[unresolved-import]

    w = WorkspaceClient()
    endpoint = os.environ.get("SERVING_ENDPOINT", "")

    def embed(wafer_map: list[list[int]]) -> list[float]:
        import numpy as np

        body = ws.endpoint_request(np.asarray(wafer_map, dtype=np.uint8))
        pred = w.api_client.do("POST", f"/serving-endpoints/{endpoint}/invocations", body=body)
        p = pred["predictions"][0]
        return p["embedding"] if isinstance(p, dict) else p

    db = LakebaseDb(w, os.environ["LAKEBASE_ENDPOINT"])
    app = create_app(
        db,
        embed if endpoint else None,
        table=os.environ.get("PG_TABLE", ws.TABLE),
        endpoint_name=endpoint,
    )
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("DATABRICKS_APP_PORT", "8000")))


if __name__ == "__main__":  # pragma: no cover
    main()
