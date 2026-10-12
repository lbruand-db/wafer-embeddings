"""CPU run of a DINO recipe, tracking student and teacher on val (PLAN.md P1 "The recipe was").

Recipes are ``jobs/train.py`` command-line flags on top of its defaults (the R1 reference
recipe), parsed by the job's own parser so a CPU run trains exactly what the GPU job would:

- ``R0``: the legacy recipe (``LEGACY_RECIPE``) at the small model size;
- ``R1``: the job defaults (lr = 5e-4 * bs / 256, WD 0.04 -> 0.4, global crop area 0.4-1,
  6 local crops, 3% die noise, pre-LN, head 2048/256, freeze 1%, clip 3);
- ``R2``: R1 with post-LN (isolates pre-LN);
- ``R3``: R1 without local crops (isolates multi-crop).

``--flags`` appends any further train-job flags. Lot-split data, val-only queries,
cross-device metrics (SPECS.md §8.7).

Run: uv run --extra jobs python tools/ref_recipe.py <wafer_maps_parquet> --recipe R1
"""

from __future__ import annotations

import argparse
import json
import shlex
import time

import numpy as np

from wafer_embeddings.jobs.train import LEGACY_RECIPE, _args, train_recipe
from wafer_embeddings.train.evaluate import REPORT_KEYS

SMALL = "--embed-dim 128 --heads 4 --depth 6 --attention isab --max-tokens 512 --batch-size 32"
RECIPES = {
    "R0": f"{LEGACY_RECIPE} {SMALL}",
    "R1": "",
    "R2": "--no-pre-norm",
    "R3": "--n-local 0",
}
# CPU budget (the GPU job: 10k steps on 100k maps)
CPU = "--steps 3000 --max-train 20000 --track-every 500"


def recipe_args(name: str, flags: str = ""):
    """The train job's parsed args for a named recipe (+ extra flags; later flags win)."""
    if name not in RECIPES:
        raise ValueError(f"recipe must be one of {sorted(RECIPES)}, got {name!r}")
    argv = ["--catalog", "-", "--schema", "-", *shlex.split(f"{CPU} {RECIPES[name]} {flags}")]
    return _args(argv)


def main(argv=None) -> None:  # pragma: no cover - CPU training run
    import pyarrow.dataset as ds
    import torch

    from wafer_embeddings.jobs.train import (
        EvalSet,
        build_positives,
        build_training,
        fit_kwargs,
        load_train_table,
    )
    from wafer_embeddings.train import embed_all, fit_dino, wafers_from_rows

    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("parquet")
    p.add_argument("--recipe", required=True, choices=sorted(RECIPES))
    p.add_argument("--flags", default="", help="Extra jobs/train.py flags, e.g. '--steps 500'.")
    p.add_argument("--threads", type=int, default=3)
    c = p.parse_args(argv)
    a = recipe_args(c.recipe, c.flags)
    torch.set_num_threads(c.threads)
    rng = np.random.default_rng(a.seed)
    torch.manual_seed(a.seed)

    # the train job's own pieces (jobs/train.py), on CPU and a local parquet copy
    dset = ds.dataset(c.parquet, format="parquet")
    train_rows = load_train_table(dset, a.max_train, a.seed).to_pylist()
    train_w, train_y, _ = wafers_from_rows(train_rows, max_tokens=a.max_tokens, seed=a.seed)
    ev = EvalSet.load(dset, a)  # val queries (protocol)
    neighbours = build_positives(a, train_w, train_y)
    dino, loss_fn, opt, recipe = build_training(a)
    print(f"recipe {c.recipe}: {recipe} train={len(train_w)} eval={len(ev.wafers)}", flush=True)

    def score(which: str) -> dict:
        m = ev.metrics(embed_all(dino, ev.wafers, which=which))
        return {k: round(m[k], 4) for k in REPORT_KEYS}

    def track(step: int) -> None:
        line = {"step": step, "student": score("student"), "teacher": score("teacher")}
        print("TRACK " + json.dumps(line), flush=True)

    t0 = time.time()
    fit_dino(
        dino,
        loss_fn,
        opt,
        train_w,
        rng=rng,
        log=lambda s: print(s, flush=True),
        log_every=100,
        track_fn=track,
        neighbours=neighbours,
        **fit_kwargs(a, recipe),
    )
    print(f"DONE {c.recipe} secs={round(time.time() - t0)}", flush=True)


if __name__ == "__main__":  # pragma: no cover
    main()
