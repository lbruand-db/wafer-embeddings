"""Run the app's query path (``app/wafer_search.py`` SQL) against the live Lakebase table.

For a few defect classes: pick a random labeled query wafer, fetch its top-k neighbours
from other lots through the ``lakebase_ann`` index, and report how many share its label,
the round-trip latency (client side: includes network) and that the plan uses the index.

Run: uv run --extra jobs python tools/app_smoke.py --profile mmf
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "app"))
import wafer_search as ws  # noqa: E402

ENDPOINT = "projects/wafer-embeddings/branches/production/endpoints/primary"
LABELS = ("Center", "Edge-Ring", "Donut", "Scratch")


def main(argv=None) -> None:  # pragma: no cover - needs a workspace + Lakebase
    import psycopg  # ty: ignore[unresolved-import]
    from databricks.sdk import WorkspaceClient  # ty: ignore[unresolved-import]

    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--profile", default=None, help="Databricks CLI profile.")
    p.add_argument("--endpoint", default=ENDPOINT, help="Lakebase endpoint resource name.")
    p.add_argument("--database", default="wafer_embeddings")
    p.add_argument("--table", default=ws.TABLE)
    p.add_argument("-k", type=int, default=12)
    a = p.parse_args(argv)
    w = WorkspaceClient(profile=a.profile)
    conn = psycopg.connect(
        host=w.postgres.get_endpoint(a.endpoint).status.hosts.host,
        dbname=a.database,
        user=w.current_user.me().user_name,
        password=w.postgres.generate_database_credential(endpoint=a.endpoint).token,
        sslmode="require",
        autocommit=True,
    )
    qvec = None
    for label in LABELS:
        q = conn.execute(ws.sample_sql(a.table, with_label=True), {"label": label}).fetchone()
        if q is None:
            print(f"{label:9s} no labeled wafer")
            continue
        qid, qlabel, _, qlot, _, _, _, qemb = q
        qvec = ws.vector_literal(ws.parse_vector(qemb))
        t0 = time.time()
        params = {"q": qvec, "qid": qid, "lot": qlot, "k": a.k}
        hits = conn.execute(ws.neighbours_sql(a.table, exclude_same_lot=True), params).fetchall()
        ms = (time.time() - t0) * 1000
        labels = [h[1] for h in hits]
        print(
            f"{label:9s} #{qid}: {sum(lb == qlabel for lb in labels)}/{len(hits)} same class, "
            f"top sim {hits[0][7]:.3f}, {ms:.0f} ms, "
            f"other lots only: {all(h[3] != qlot for h in hits)}; labels={labels[:6]}"
        )
    if qvec is not None:
        params = {"q": qvec, "qid": 0, "lot": "", "k": a.k}
        plan = conn.execute("EXPLAIN " + ws.neighbours_sql(a.table), params).fetchall()
        print("plan:", " | ".join(r[0] for r in plan[:4]))


if __name__ == "__main__":  # pragma: no cover
    main()
