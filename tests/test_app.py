"""The app's pure helpers (app/wafer_search.py) — rendering, codec parity, SQL."""

import importlib.util
import pathlib

import numpy as np

from wafer_embeddings.serving import encode_map

_spec = importlib.util.spec_from_file_location(
    "wafer_search", pathlib.Path(__file__).parents[1] / "app" / "wafer_search.py"
)
assert _spec is not None and _spec.loader is not None
ws = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ws)


def test_app_decoder_matches_package_encoder():
    rng = np.random.default_rng(0)
    for shape in [(1, 1), (5, 7), (40, 33)]:
        m = rng.integers(0, 3, size=shape).astype(np.uint8)
        np.testing.assert_array_equal(ws.decode_map(encode_map(m)), m)


def test_render_colours_and_scale():
    m = np.array([[0, 1], [2, 1]], dtype=np.uint8)
    img = ws.render(m, scale=3)
    assert img.shape == (6, 6, 3) and img.dtype == np.uint8
    assert (img[0, 0] == ws.PALETTE[0]).all() and (img[3, 0] == ws.PALETTE[2]).all()
    assert ws.auto_scale(np.zeros((40, 20))) == 4 and ws.auto_scale(np.zeros((500, 500))) == 1


def test_vector_literal_round_trip():
    v = [0.5, -1.25e-3, 3.0]
    assert ws.parse_vector(ws.vector_literal(v)) == v


def test_neighbours_sql_uses_the_ann_operator_and_binds():
    sql = ws.neighbours_sql()
    assert "ORDER BY embedding <=> %(q)s::vector" in sql and "LIMIT %(k)s" in sql
    assert "id <> %(qid)s" in sql and "lot" not in sql.split("WHERE")[1].split("ORDER")[0]
    assert "IS DISTINCT FROM %(lot)s" in ws.neighbours_sql(exclude_same_lot=True)


def test_sample_and_by_id_sql():
    assert "label = %(label)s" in ws.sample_sql(with_label=True)
    assert "label IS NOT NULL" in ws.sample_sql()
    assert "id = %(qid)s" in ws.by_id_sql()


def test_endpoint_request_matches_pyfunc_layout():
    m = np.array([[0, 1, 2], [1, 1, 0]], dtype=np.uint8)
    body = ws.endpoint_request(m)["dataframe_split"]
    assert body["columns"] == ["height", "width", "wafer_map"]
    assert body["data"] == [[2, 3, [0, 1, 2, 1, 1, 0]]]
