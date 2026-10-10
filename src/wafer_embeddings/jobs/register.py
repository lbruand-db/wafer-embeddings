"""Package a trained checkpoint as an MLflow pyfunc and register it in Unity Catalog.

Pulls ``encoder/wafer_encoder.pt`` from a training run (or takes a local file), logs the
pyfunc with a signature + input example, registers it as ``<catalog>.<schema>.<name>``,
sets an alias (default ``champion``) and records gate metrics as version tags
(SPECS.md §11.4). Runs anywhere with workspace auth (CPU is enough: the endpoint is CPU).
"""

from __future__ import annotations

import argparse
import json
import os
import sys


def pip_requirements(torch_version: str, numpy_version: str) -> list[str]:
    """CPU torch wheels (no CUDA, much smaller image) + the runtime deps of the encoder."""
    return [
        "--extra-index-url https://download.pytorch.org/whl/cpu",
        f"torch=={torch_version.split('+')[0]}",
        f"numpy=={numpy_version}",
        "pandas",
    ]


def run_filter(run_name: str, since_ms: int = 0) -> str:
    """MLflow search filter: finished runs with this name, started at/after ``since_ms``."""
    if "'" in run_name:
        raise ValueError(f"unsafe run name: {run_name!r}")
    f = f"attributes.run_name = '{run_name}' AND attributes.status = 'FINISHED'"
    return f + (f" AND attributes.start_time >= {int(since_ms)}" if since_ms else "")


def pick_run_id(run_ids: list[str], run_name: str, since_ms: int = 0) -> str:
    """The newest matching run (search is ordered newest first), or a clear error."""
    if not run_ids:
        raise SystemExit(f"no FINISHED run named {run_name!r} started at/after {since_ms}")
    return run_ids[0]


def sample_input():
    """Two tiny maps in the Delta row layout, for the signature + input example."""
    import pandas as pd

    return pd.DataFrame(
        {
            "height": [3, 2],
            "width": [3, 4],
            "wafer_map": [[0, 1, 0, 1, 2, 1, 0, 1, 0], [1, 1, 2, 1, 1, 2, 1, 1]],
        }
    )


def _args(argv=None):
    p = argparse.ArgumentParser(description="Register the wafer encoder pyfunc in UC.")
    p.add_argument("--catalog", required=True)
    p.add_argument("--schema", required=True)
    p.add_argument("--name", default="wafer_encoder")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--run-id", help="Training run whose encoder artifact to package.")
    src.add_argument("--checkpoint", help="Local checkpoint .pt instead of a run.")
    src.add_argument(
        "--latest-run-name",
        help="Package the newest FINISHED run with this name in --experiment (the pipeline's "
        "training run), optionally started after --since-ms.",
    )
    p.add_argument("--since-ms", type=int, default=0, help="Only runs started at/after this.")
    p.add_argument("--experiment", default=None, help="MLflow experiment (search + log run).")
    p.add_argument("--alias", default="champion")
    p.add_argument("--metrics-json", default=None, help="Gate metrics to tag the version with.")
    return p.parse_args(argv)


def main(argv=None) -> None:  # pragma: no cover - needs a workspace
    import mlflow  # ty: ignore[unresolved-import]
    import numpy
    import torch
    from mlflow.models import infer_signature  # ty: ignore[unresolved-import]

    from wafer_embeddings.serving.encoder import WaferEncoder
    from wafer_embeddings.serving.pyfunc import CHECKPOINT, WaferEncoderModel

    a = _args(argv)
    mlflow.set_registry_uri("databricks-uc")
    if a.experiment:
        mlflow.set_experiment(a.experiment)
    if a.latest_run_name:
        if not a.experiment:
            raise SystemExit("--latest-run-name needs --experiment")
        exp = mlflow.get_experiment_by_name(a.experiment)
        if exp is None:
            raise SystemExit(f"experiment {a.experiment} not found")
        runs = mlflow.search_runs(
            [exp.experiment_id],
            filter_string=run_filter(a.latest_run_name, a.since_ms),
            order_by=["attributes.start_time DESC"],
            max_results=1,
            output_format="list",
        )
        a.run_id = pick_run_id([r.info.run_id for r in runs], a.latest_run_name, a.since_ms)
        print(f"registering run {a.run_id} (latest finished {a.latest_run_name!r})")
    ckpt = a.checkpoint or mlflow.artifacts.download_artifacts(
        run_id=a.run_id, artifact_path="encoder/wafer_encoder.pt"
    )
    enc = WaferEncoder.from_checkpoint(ckpt)
    x = sample_input()
    y = WaferEncoderModel()
    y.encoder = enc
    signature = infer_signature(x, y.predict(None, x))
    fqn = f"{a.catalog}.{a.schema}.{a.name}"
    pkg = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # .../wafer_embeddings
    metrics = json.load(open(a.metrics_json)) if a.metrics_json else {}
    with mlflow.start_run(run_name=f"register-{a.name}") as run:
        mlflow.log_params(
            {
                "source_run_id": a.run_id or "",
                "embed_dim": enc.dim,
                "max_tokens": enc.max_tokens,
                "n_params": enc.n_params,
            }
            | {
                f"arch_{k}": v
                for k, v in enc.config.items()
                if k in ("depth", "heads", "attention", "pre_norm")
            }
        )
        info = mlflow.pyfunc.log_model(
            name="model",
            python_model=WaferEncoderModel(),
            artifacts={CHECKPOINT: ckpt},
            code_paths=[pkg],
            signature=signature,
            input_example=x,
            pip_requirements=pip_requirements(torch.__version__, numpy.__version__),
            registered_model_name=fqn,
        )
        for split_name, by_model in metrics.items() if isinstance(metrics, dict) else []:
            if isinstance(by_model, dict):
                mlflow.log_metrics(
                    {
                        f"{split_name}/{k}": v
                        for k, v in by_model.items()
                        if isinstance(v, (int, float))
                    }
                )
    client = mlflow.MlflowClient()
    version = info.registered_model_version
    client.set_registered_model_alias(fqn, a.alias, version)
    tags = {
        "source_run_id": a.run_id or "",
        "embed_dim": str(enc.dim),
        "n_params": str(enc.n_params),
        "gate": "G1-provisional",
    }
    trained = metrics.get("trained", {}) if isinstance(metrics, dict) else {}
    for k in ("xgroup_knn_macro_recall", "xgroup_precision@10", "map@10", "cluster_nmi"):
        if k in trained:
            tags[f"test_{k.replace('@', '_at_')}"] = str(trained[k])
    for k, v in tags.items():
        client.set_model_version_tag(fqn, version, k, v)
    print(
        json.dumps(
            {
                "model": fqn,
                "version": version,
                "alias": a.alias,
                "log_run_id": run.info.run_id,
                "model_uri": f"models:/{fqn}@{a.alias}",
            }
        )
    )


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
