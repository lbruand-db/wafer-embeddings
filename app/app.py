"""Wafer-map similarity search (Databricks App).

Pick a wafer, see its map, and retrieve its nearest neighbours from Lakebase Search
(``lakebase_ann`` cosine index over the encoder's embeddings). Optionally re-embed the
query live through the ``wafer-encoder`` Model Serving endpoint.
"""

from __future__ import annotations

import os

import numpy as np
import psycopg
import streamlit as st
from databricks.sdk import WorkspaceClient

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
ENDPOINT = os.environ.get("SERVING_ENDPOINT", "dev_lucas_bruand_wafer-encoder")
LAKEBASE_ENDPOINT = os.environ.get(
    "LAKEBASE_ENDPOINT", "projects/wafer-embeddings/branches/production/endpoints/primary"
)

st.set_page_config(page_title="Wafer-map search", layout="wide")


@st.cache_resource
def client() -> WorkspaceClient:
    return WorkspaceClient()


def connect():
    """Postgres connection with a fresh OAuth credential (tokens are short-lived)."""
    w = client()
    try:
        password = w.postgres.generate_database_credential(endpoint=LAKEBASE_ENDPOINT).token
    except Exception:  # older SDKs: the workspace OAuth token also works for Lakebase
        password = w.config.oauth_token().access_token
    return psycopg.connect(
        host=os.environ["PGHOST"],
        port=int(os.environ.get("PGPORT", "5432")),
        dbname=os.environ.get("PGDATABASE", "wafer_embeddings"),
        user=os.environ.get("PGUSER") or w.current_user.me().user_name,
        password=password,
        sslmode=os.environ.get("PGSSLMODE", "require"),
        autocommit=True,
    )


def query(sql: str, params: dict) -> list[tuple]:
    with connect() as conn, conn.cursor() as cur:
        cur.execute("SET lakebase_ann.probes = 'auto'")
        cur.execute(sql, params)
        return cur.fetchall()


def embed_live(wafer_map: np.ndarray) -> list[float]:
    resp = client().api_client.do(
        "POST", f"/serving-endpoints/{ENDPOINT}/invocations", body=ws.endpoint_request(wafer_map)
    )
    pred = resp["predictions"][0]
    return pred["embedding"] if isinstance(pred, dict) else pred


st.title("Wafer-map similarity search")
st.caption("WM-811K · per-die ViT + DINO embeddings · Lakebase Search (lakebase_ann, cosine)")

with st.sidebar:
    mode = st.radio("Query wafer", ["Random wafer", "By wafer id"])
    label = st.selectbox("Class (random mode)", ["any"] + CLASSES)
    wafer_id = st.number_input("Wafer id", min_value=0, value=0, step=1)
    k = st.slider("Neighbours", 4, 24, 12, step=4)
    cross_lot = st.checkbox("Exclude the query's own lot", value=True)
    live = st.checkbox(f"Re-embed query via endpoint `{ENDPOINT}`", value=False)
    go = st.button("Search", type="primary")

if go or "q" not in st.session_state:
    if mode == "Random wafer":
        rows = query(ws.sample_sql(with_label=label != "any"), {"label": label})
    else:
        rows = query(ws.by_id_sql(), {"qid": int(wafer_id)})
    if not rows:
        st.warning("No wafer found.")
        st.stop()
    st.session_state.q = rows[0]

qid, qlabel, qsplit, qlot, _, _, qcode, qemb = st.session_state.q
qmap = ws.decode_map(qcode)
vec = embed_live(qmap) if live else ws.parse_vector(qemb)
hits = query(
    ws.neighbours_sql(exclude_same_lot=cross_lot),
    {"q": ws.vector_literal(vec), "qid": qid, "lot": qlot, "k": int(k)},
)

left, right = st.columns([1, 3])
with left:
    st.subheader("Query")
    st.image(ws.render(qmap, ws.auto_scale(qmap)), caption=f"#{qid} · {qlabel or 'unlabeled'}")
    st.write(f"split **{qsplit}** · lot `{qlot}` · {qmap.shape[0]}×{qmap.shape[1]}")
    st.write("embedding: " + ("live from the endpoint" if live else "stored (batch job)"))
with right:
    st.subheader(f"Top {len(hits)} neighbours" + (" (other lots)" if cross_lot else ""))
    same = sum(1 for h in hits if h[1] == qlabel and qlabel)
    if qlabel:
        st.write(f"{same}/{len(hits)} share the query's class **{qlabel}**")
    cols = st.columns(4)
    for i, (hid, hlabel, hsplit, hlot, _, _, hcode, sim) in enumerate(hits):
        m = ws.decode_map(hcode)
        with cols[i % 4]:
            st.image(
                ws.render(m, ws.auto_scale(m, 120)),
                caption=f"#{hid} · {hlabel or 'unlabeled'} · cos {sim:.3f}",
            )
