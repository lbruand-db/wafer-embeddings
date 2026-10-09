"""The Vue app's FastAPI backend (app/server.py) with a fake Lakebase and endpoint."""

import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

APP = Path(__file__).parents[1] / "app"
sys.path.insert(0, str(APP))
import server  # noqa: E402
import wafer_search as ws  # noqa: E402

from wafer_embeddings.serving import encode_map  # noqa: E402

MAP = np.array([[0, 1, 1], [1, 2, 1], [0, 1, 0]], dtype=np.uint8)
CODE = encode_map(MAP)
QUERY = (7, "Center", "test", "lotA", 3, 3, CODE, "[1,0,0]")
HITS = [
    (8, "Center", "train", "lotB", 3, 3, CODE, 0.99),
    (9, "Donut", "train", "lotC", 3, 3, CODE, 0.80),
    (10, None, "train", "lotD", 3, 3, CODE, 0.75),
]


class FakeDb:
    def __init__(self):
        self.calls: list[tuple[str, dict]] = []

    def query(self, sql, params):
        self.calls.append((sql, params))
        if "ORDER BY random()" in sql:
            return [QUERY] if params.get("label") in (None, "Center") else []
        if "WHERE id = %(qid)s" in sql:
            return [QUERY] if params["qid"] == 7 else []
        return HITS[: params["k"]]


def _client(embed=None):
    db = FakeDb()
    app = server.create_app(db, embed, table="s.t", endpoint_name="ep", static_dir=None)
    return TestClient(app), db


def test_health_and_config():
    c, _ = _client()
    assert c.get("/api/health").json() == {"status": "ok"}
    cfg = c.get("/api/config").json()
    assert "Center" in cfg["classes"] and cfg["endpoint"] == "ep" and cfg["live_embedding"] is False


def test_random_and_by_id_wafer():
    c, db = _client()
    assert c.get("/api/wafer", params={"label": "Center"}).json()["id"] == 7
    assert c.get("/api/wafer", params={"id": 7}).json()["map_code"] == CODE
    assert c.get("/api/wafer", params={"id": 99}).status_code == 404
    assert c.get("/api/wafer", params={"label": "Bogus"}).status_code == 422
    assert all("FROM s.t" in sql for sql, _ in db.calls)  # configured table, not a default


def test_search_uses_stored_vector_and_cross_lot():
    c, db = _client()
    r = c.get("/api/search", params={"id": 7, "k": 3, "cross_lot": True}).json()
    assert r["query"]["id"] == 7 and r["embedding"] == "stored"
    assert [n["id"] for n in r["neighbours"]] == [8, 9, 10]
    assert r["neighbours"][0]["similarity"] == pytest.approx(0.99)
    assert r["same_class"] == 1
    sql, params = db.calls[-1]
    assert "IS DISTINCT FROM %(lot)s" in sql and params["lot"] == "lotA"
    assert params["q"] == ws.vector_literal([1.0, 0.0, 0.0]) and params["k"] == 3


def test_search_live_embedding_goes_through_the_endpoint():
    seen = []

    def embed(m):
        seen.append(m)
        return [0.0, 1.0, 0.0]

    c, db = _client(embed)
    r = c.get("/api/search", params={"id": 7, "live": True, "cross_lot": False}).json()
    assert r["embedding"] == "live" and seen == [MAP.tolist()]  # decoded map sent
    sql, params = db.calls[-1]
    assert params["q"] == ws.vector_literal([0.0, 1.0, 0.0]) and "IS DISTINCT FROM" not in sql


def test_search_validation():
    c, _ = _client()
    assert c.get("/api/search", params={"id": 7, "live": True}).status_code == 400  # no endpoint
    assert c.get("/api/search", params={"id": 7, "k": 0}).status_code == 422
    assert c.get("/api/search", params={"id": 99}).status_code == 404


def test_serves_the_built_frontend(tmp_path):
    (tmp_path / "index.html").write_text("<div id=app></div>")
    app = server.create_app(FakeDb(), None, static_dir=tmp_path)
    c = TestClient(app)
    assert "id=app" in c.get("/").text
    assert c.get("/api/health").status_code == 200  # API routes still win over static
