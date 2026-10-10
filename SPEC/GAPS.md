# Open gaps

What is still missing, as of 2026-10-10. Everything listed here is **not done**; finished
work lives in [`PLAN.md`](./PLAN.md) and the design in [`SPECS.md`](./SPECS.md).
Items marked **🧭 decision** need an owner's call before work starts.

Suggested order: **1 → 2 → 4.1–4.3 → 3 → the rest**.

---

## 1. Modelling — the real Gate G1 *(highest priority)*

The served model (`wafer_encoder` v2, 128-d) is only **provisionally** gated. On the test
split it beats the untrained encoder but not the training-free polar baseline on the
headline metric (cross-device kNN macro recall **0.389 vs 0.442**; precision@10 0.356 vs
0.359; it does beat polar on map@10, 0.346 vs 0.315).

| # | Gap | Done when |
|---|---|---|
| 1.1 | **Beat the polar baseline.** Try, in order: descriptor-guided positives (polar-histogram nearest neighbours as extra DINO / InfoNCE positives, label-free) + a plain InfoNCE baseline; then Sonata-style masked self-distillation; fallback: polar + learned concatenation. | Val cross-device macro recall and precision@10 above polar beyond seed noise (1.2), then confirmed **once** on test. |
| 1.2 | **Seed noise.** The gate says "beyond seed noise", but no run has been repeated with other seeds (training seed and eval-sample seed). | ≥ 3 seeds for the baseline recipe; a ± band on the val metrics. |
| 1.3 | **Weak classes.** Loc (test kNN recall 0.07) and Scratch (0.13) are poor for both the model and polar. | Per-class recall tracked as a target; a recipe change that moves them. |
| 1.4 | **Spec baselines not wired** (SPECS §16 items 1, 24): frozen off-the-shelf DINOv2 on rasterized maps, flattened-pixel / PCA. | Logged next to `untrained_*` / `polar_*` in every run. |
| 1.5 | **False-negative check** (SPECS §16 item 3): pure DINO vs SWaCo-style same-class positives. | Ablation result recorded. |
| 1.6 | **Capacity, revisited.** The full-size model (width 384, depth 12) matched the small one at 10k steps with an untuned LR. | Re-run with a tuned LR / longer schedule once 1.1 beats polar. |
| 1.7 | **🧭 decision — G1 definition.** Plain map@10 retrieves among val queries, so same-lot neighbours still help it. | Gate defined on cross-device metrics only, or map@10 redefined as cross-lot retrieval. |
| 1.8 | **🧭 decision — embedding width.** 128-d today vs the spec's 384-d (N1). | Width chosen for the next real G1 model. |

## 2. Lakebase Search — Gate G2 measurements

Lakebase Search works end to end at 696,599 × 128-d, but the acceptance numbers were never
measured.

| # | Gap | Done when |
|---|---|---|
| 2.1 | **ANN recall@10 vs exact** (SPECS §16 item 13). | ≥ 0.95, or `lakebase_ann.probes` / `epsilon` / `build_mode` tuned until it is. |
| 2.2 | **Server-side latency** (§16 item 14). The ~40–120 ms seen from the app includes network and credentials. | p50 / p99 measured in-database; p99 < 50 ms. |
| 2.3 | **Scale** (N3): synthetic 1M rows. | Recall + latency re-measured at 1M. |
| 2.4 | **384-d** re-check (§16 items 11, 12) if 1.8 picks 384. | Synced-table type override + index verified at `vector(384)`. |

## 3. P2 — scale & generalization *(not started)*

| # | Gap | Done when |
|---|---|---|
| 3.1 | **Attention feasibility** (§16 item 4): full vs ISAB at median (~1–2k) and capped (~4k) tokens on an A10 — wall-clock, peak memory, encode latency. | Benchmark table in PLAN.md. |
| 3.2 | **ISAB quality** (§16 item 5) and **patch 1 vs 2** (§16 item 6). | Smallest config that keeps quality chosen (N4). |
| 3.3 | **Size generalization** (§16 items 7, 8): hold out whole size groups; token-cap sensitivity. The served model was trained and evaluated at a **512-token** cap — the effect of the cap on giant maps is unmeasured. | Same pattern across sizes → nearby embeddings; cap chosen from data. |
| 3.4 | **Invariance** (§16 items 9, 10): rotation / flip view-consistency; do crops confuse Loc ↔ Edge-Loc? | Checks pass, recorded. |

## 4. Platform & automation

| # | Gap | Done when |
|---|---|---|
| 4.1 | ~~**Registration runs locally.**~~ ✅ **Done (2026-10-10):** the `pipeline` job's `register` task (serverless, `wafer-register`) packages the newest FINISHED `pipeline-train` run started after the job began (`--latest-run-name`, `--since-ms {{job.start_time.timestamp_ms}}`) under the `alias` job parameter. | — |
| 4.2 | ~~**No orchestrated flow.**~~ ✅ **Done (2026-10-10):** `resources/pipeline.yml`. **`pipeline`**: train (`ai_runtime_task`, GPU) → register → `alias == champion`? → **`publish`**: serve `@champion` → embed (GPU) → load → Lakebase sync (`--refresh`). Default `alias=candidate` registers without serving (promotion is explicit). Verified: smoke pipeline (train → register v3 → publish excluded, champion untouched) and a full publish (696,599 maps, synced table refreshed). | — |
| 4.3 | ~~**Training defaults ≠ the working recipe.**~~ ✅ **Done (2026-10-10):** `jobs/train.py` + `ai_runtime/train.yaml` default to the R1 recipe (w128 d6 ISAB pre-LN, 512 tokens, bs32 × 10k steps on 100k maps, LR rule, WD 0.04 → 0.4, 6 local crops, 3% die noise, head 2048/256); the old one is `jobs.train.LEGACY_RECIPE` (tested). Tracking defaults to every 1000 steps on val, off on test. | — |
| 4.4 | ~~**Model width / version wired by hand.**~~ ✅ **Done (2026-10-10):** the Lakebase task detects D from the embeddings table (and rebuilds a synced table whose width changed); `jobs/serve.py` resolves `wafer_encoder@champion` and creates / updates the endpoint to serve that version (idempotent). `var.embed_dim` and `var.encoder_version` are gone; the endpoint left the bundle (Model Serving can't take an alias). | — |
| 4.5 | ~~**No teardown.**~~ ✅ **Done (2026-10-10):** `wafer-teardown` (`jobs/teardown.py`) removes the code-managed resources in dependency order — `--scope serving`: endpoint, synced table, Lakebase project, embeddings table + parquet; `--scope all`: also the UC model, `wafer_maps`, the volume and the schema (never the catalog) — then `databricks bundle destroy` removes jobs + app. Dry run by default (`--yes` deletes), already-gone targets skipped. Live dry run of both scopes verified; the deleting path has not been run (it would remove the demo). | — |
| 4.6 | ~~**Dev-only target.**~~ ⊘ **Descoped (2026-10-10, owner decision).** A `prod` target exists and validates (`mode: production`) but is not deployed; no service principal will be created. | — |
| 4.7 | ~~**GPU serving tier.**~~ ⊘ **Descoped (2026-10-10, owner decision).** The CPU endpoint serves the current model in < 1 s; revisit only for larger models or high throughput. | — |
| 4.8 | **No dedicated Lakebase UC catalog** — no `CREATE CATALOG` on the metastore, so the synced table lives in `mmf_mlops_demo_catalog`. Keep the catalog as is but just do a CREATE CATALOG IF NOT EXISTS. ✅ **Done (2026-10-10):** the bootstrap checks `SHOW CATALOGS LIKE` and runs `CREATE CATALOG IF NOT EXISTS` only when the catalog is missing (UC checks the metastore privilege before existence, so the bare statement fails with PERMISSION_DENIED even as a no-op here). Live run: catalog exists → skipped. | Accepted as-is: one catalog, created if missing. |

## 5. Operations & cost

| # | Gap | Done when |
|---|---|---|
| 5.1 | **App start / stop is manual.** Standard Databricks Apps have no scale-to-zero; the app (MEDIUM, 0.5 DBU/h ≈ $12/day at list) is currently **stopped**: `databricks apps start wafer-search --profile mmf` before a demo. | Optional: a scheduled start/stop job (e.g. weekdays 08:00–19:00). |
| 5.2 | **No monitoring / alerts** for the endpoint, Lakebase or job failures. | Alerts on failed jobs and endpoint errors; a usage / cost view from `system.billing`. |

## 6. Hygiene

| # | Gap | Done when |
|---|---|---|
| 6.1 | ~~**Diagnostics live only in `scratch/`.**~~ ✅ **Done (2026-10-10):** `tools/` (`baseline_check`, `nuisance_probe`, `ref_recipe`, `score_test`, `app_smoke`), documented in [`tools/README.md`](../tools/README.md). Refreshed on the way (`nuisance_probe` imported a removed module; `baseline_check` now uses the library polar baseline; `ref_recipe` builds recipes with `jobs/train.py`'s own parser, so R1 *is* the job default). Pure helpers unit-tested (`tests/test_tools.py`); each script smoke-run on CPU. The other `scratch/` scripts were superseded one-offs and stay out. | — |
| 6.2 | ~~**Proxy-driven pins.**~~ ✅ **Done (2026-10-10):** the proxies still 403 the newest releases (vite 8.3.4 → rolldown 1.2.13, rollup 4.64.4, boto3 1.43.111, pyarrow 26, fastapi 0.143), but it is a **release-age cooldown** (~1 week), not a version problem. So the caps are gone from the committed files — `boto3>=1.34` (no `<1.43`), Vite **8.3.2** with no rollup `overrides` — and local installs resolve as of a week ago (`UV_EXCLUDE_NEWER`, `npm_config_before`; README *Local dev*). Verified: `uv sync` (boto3 1.43.108), pytest, `npm install` + `vite build` + JS tests. Not verified: an Apps build on Vite 8 (the app is stopped; it installs from public npm). | — |
| 6.3 | **`uv.lock` not committed** (it resolves through the internal PyPI proxy). | A portable lock generated off-proxy, or accepted as-is. |
| 6.4 | ~~**CI runner image.**~~ ✅ **Done (2026-10-10):** `ubuntu-latest` moves to Ubuntu 26.04 from 2026-10-19 ([runner-images#14748](https://github.com/actions/runner-images/issues/14748)); 26.04 is already GA, so CI is pinned to `runs-on: ubuntu-26.04` now (no silent image moves, like the pinned actions). | — |
| 6.5 | ~~**Deferred code-review items** (eval tracking).~~ ✅ **Done (2026-10-10):** the tracked point at the last step no longer re-runs the eval (the final eval logs that step with every metric; only the train RankMe probe is tracked there), and `--no-track-clustering` is a light tracking path that skips the KMeans fit (`g1_metrics(clustering=False)`; the final eval is always full). Default unchanged (clustering tracked), since it is cheap at the current eval size. | — |
| 6.6 | **Test split use.** Test was scored once for `wafer_encoder` v2 (provisional G1). Every future G1 candidate must likewise be scored on test **once**, after its development is frozen on val. | Recorded per registered version (version tags). |
