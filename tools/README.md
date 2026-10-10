# tools/

Diagnostics behind the findings in [`SPEC/PLAN.md`](../SPEC/PLAN.md) (P1 "Status"). They
are run by hand, not by the bundle. Their pure helpers are unit-tested in
`tests/test_tools.py`, so the scripts break in CI rather than silently when the package
changes.

Most of them read a local copy of the `wafer_maps_parquet` export (UC volume
`/Volumes/<catalog>/wafer_embeddings/raw/wafer_maps_parquet`), for example:

```bash
databricks fs cp -r --profile mmf \
  dbfs:/Volumes/mmf_mlops_demo_catalog/wafer_embeddings/raw/wafer_maps_parquet data/wafer_maps_parquet
```

`data/` is gitignored. All runs are CPU-only and free.

| Tool | What it shows | Run |
|---|---|---|
| `baseline_check.py` | The **shape leak** (same map shape → same label, over query × bank pairs), a shape one-hot "encoder", and the polar baseline, all with the G1 metrics. | `uv run --extra jobs python tools/baseline_check.py data/wafer_maps_parquet` |
| `nuisance_probe.py` | Whether DINO learns die-grid / wafer-size features: `shape@10`, `label@10`, `log(#dies)` R², before vs after a short training run; knobs `--n-bands`, `--jitter`, `--aug`. | `uv run --extra jobs python tools/nuisance_probe.py data/wafer_maps_parquet --n-bands 6` |
| `ref_recipe.py` | A CPU run of a DINO recipe (R0 legacy, R1 job defaults, R2 post-LN, R3 no local crops), parsed by `jobs/train.py`'s own parser, with student and teacher tracked on val. | `uv run --extra jobs python tools/ref_recipe.py data/wafer_maps_parquet --recipe R1` |
| `score_test.py` | The one-time **test-split** score against the untrained encoder and polar (SPECS §8.7). With `--model/--version` it scores the registered checkpoint and tags the version (`test_scored_at`, `test_*`, `test_polar_*`), refusing one already scored. With `--checkpoint`, its JSON feeds `jobs/register.py --metrics-json`. | `uv run --extra jobs python tools/score_test.py data/wafer_maps_parquet --model mmf_mlops_demo_catalog.wafer_embeddings.wafer_encoder --version 3` |
| `app_smoke.py` | The app's query path against live Lakebase: same-class neighbours from other lots, round-trip latency, and the `lakebase_ann` plan. | `uv run --extra jobs --with psycopg[binary] python tools/app_smoke.py --profile mmf` |
| `compare_runs.py` | Final val metrics of named MLflow runs (student, teacher) next to the polar and pixel-PCA bars, and a mean ± std **seed band** for a group of runs of one recipe (GAPS 1.2). | `uv run --extra jobs python tools/compare_runs.py --profile mmf --band seed-s0 seed-s1 seed-s2 --runs nnpix-k5` |
