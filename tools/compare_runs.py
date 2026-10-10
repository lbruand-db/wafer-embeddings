"""Compare training runs' final val metrics, with a seed band (GAPS 1.1, 1.2).

Prints, per MLflow run name, the final ``<split>/<model>/<metric>`` values for the
trained student and teacher and the training-free bars (polar, pixel-PCA), and for a
``--band`` group of run names (e.g. the same recipe at several seeds) the mean ± std,
so a candidate can be judged "beyond seed noise". The newest FINISHED run of each name
is used.

Run: uv run --extra jobs python tools/compare_runs.py --profile mmf \\
       --band seed-s0 seed-s1 seed-s2 --runs nn-k5 nnpix-k5 orient-off
"""

from __future__ import annotations

import argparse
import math

KEYS = ("xgroup_knn_macro_recall", "xgroup_precision@10", "map@10", "cluster_nmi")
MODELS = ("student", "teacher", "polar", "pixel_pca")
SHORT = {
    "xgroup_knn_macro_recall": "x-recall",
    "xgroup_precision@10": "x-p@10",
    "map@10": "map@10",
    "cluster_nmi": "nmi",
}


def band(values: list[float]) -> tuple[float, float]:
    """Mean and sample std (0 for a single value)."""
    n = len(values)
    if n == 0:
        raise ValueError("no values")
    mean = sum(values) / n
    var = sum((v - mean) ** 2 for v in values) / (n - 1) if n > 1 else 0.0
    return mean, math.sqrt(var)


def final_metrics(metrics: dict[str, float], split: str = "val") -> dict[str, dict[str, float]]:
    """``{model: {metric: value}}`` from a run's flat ``<split>/<model>/<metric>`` dict."""
    out: dict[str, dict[str, float]] = {}
    for key, v in metrics.items():
        parts = key.split("/")
        if len(parts) == 3 and parts[0] == split and parts[1] in MODELS and parts[2] in KEYS:
            out.setdefault(parts[1], {})[parts[2]] = v
    return out


W = 15  # column width: fits "0.4050±0.0123"


def _row(label: str, m: dict[str, float]) -> str:
    return f"{label:28s}" + "".join(f"{m.get(k, float('nan')):>{W}.4f}" for k in KEYS)


def main(argv=None) -> None:  # pragma: no cover - needs a workspace
    import mlflow  # ty: ignore[unresolved-import]

    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--profile", default=None)
    p.add_argument("--experiment", default="/Users/lucas.bruand@databricks.com/wafer-embeddings")
    p.add_argument("--split", default="val")
    p.add_argument("--band", nargs="*", default=[], help="Run names of one recipe, many seeds.")
    p.add_argument("--runs", nargs="*", default=[], help="Other run names to compare.")
    a = p.parse_args(argv)
    if a.profile:
        import os

        os.environ["DATABRICKS_CONFIG_PROFILE"] = a.profile
    mlflow.set_tracking_uri("databricks")
    exp = mlflow.get_experiment_by_name(a.experiment)
    if exp is None:
        raise SystemExit(f"experiment {a.experiment} not found")

    def fetch(name: str) -> dict[str, dict[str, float]]:
        runs = mlflow.search_runs(
            [exp.experiment_id],
            filter_string=f"attributes.run_name = '{name}' AND attributes.status = 'FINISHED'",
            order_by=["attributes.start_time DESC"],
            max_results=1,
            output_format="list",
        )
        return final_metrics(runs[0].data.metrics, a.split) if runs else {}

    print(f"{'':28s}" + "".join(f"{SHORT[k]:>{W}s}" for k in KEYS))
    got = {name: fetch(name) for name in [*a.band, *a.runs]}
    bars = next((m for m in got.values() if "polar" in m), {})
    for model in ("polar", "pixel_pca"):
        if model in bars:
            print(_row(f"[{model}]", bars[model]))
    if a.band:
        for model in ("student", "teacher"):
            vals = {
                k: [got[n][model][k] for n in a.band if k in got[n].get(model, {})] for k in KEYS
            }
            stats = {k: band(v) for k, v in vals.items() if v}
            n = max((len(v) for v in vals.values()), default=0)
            print(
                f"{'band ' + model + f' (n={n})':28s}"
                + "".join(f"{f'{stats[k][0]:.4f}±{stats[k][1]:.4f}':>{W}s}" for k in stats)
            )
    for name in [*a.band, *a.runs]:
        for model in ("student", "teacher"):
            if model in got[name]:
                print(_row(f"{name} {model}", got[name][model]))
            elif model == "student":
                print(f"{name:28s} (no finished run with {a.split}/<model>/<metric> metrics)")


if __name__ == "__main__":  # pragma: no cover
    main()
