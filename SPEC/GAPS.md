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
| 4.1 | **Registration runs locally** (`uv run … jobs.register`). | A bundle / AI Runtime task registers and sets `@champion`. |
| 4.2 | **No orchestrated flow.** train → register → embed → load + Lakebase sync are separate commands. | One workflow, re-runnable end to end. |
| 4.3 | **Training defaults ≠ the working recipe.** `jobs/train.py` / `ai_runtime/train.yaml` still default to the old recipe (depth 12, LR 5e-4, no local crops, 4,096 tokens, head 1024/64); the R1 recipe that works is only reachable through `EXTRA_ARGS`. | The R1 recipe is the default (YAML + argparse), old one reproducible via flags. |
| 4.4 | **Model width / version wired by hand.** `encoder_version` and `embed_dim` are bundle variables that must match the registered model. | Derived from the UC model (alias → version, version tag → dim). |
| 4.5 | **No teardown** target (SPECS §11.8). | One command removes the endpoint, app, Lakebase project, synced table and Delta outputs. |
| 4.6 | **Dev-only target.** Resources carry the `dev_lucas_bruand_` prefix; no `prod` target, no service-principal `run_as`. | A `prod` target with an SP identity. |
| 4.7 | **GPU serving tier untested** (§16 item 18: log from a GPU runtime, `GPU_SMALL`). Only needed for larger models / throughput. | Endpoint healthy on GPU, or explicitly descoped. |
| 4.8 | **No dedicated Lakebase UC catalog** — no `CREATE CATALOG` on the metastore, so the synced table lives in `mmf_mlops_demo_catalog`. Keep the catalog as is but just do a CREATE CATALOG IF NOT EXISTS. ✅ **Done (2026-10-10):** the bootstrap checks `SHOW CATALOGS LIKE` and runs `CREATE CATALOG IF NOT EXISTS` only when the catalog is missing (UC checks the metastore privilege before existence, so the bare statement fails with PERMISSION_DENIED even as a no-op here). Live run: catalog exists → skipped. | Accepted as-is: one catalog, created if missing. |

## 5. Operations & cost

| # | Gap | Done when |
|---|---|---|
| 5.1 | **App start / stop is manual.** Standard Databricks Apps have no scale-to-zero; the app (MEDIUM, 0.5 DBU/h ≈ $12/day at list) is currently **stopped**: `databricks apps start wafer-search --profile mmf` before a demo. | Optional: a scheduled start/stop job (e.g. weekdays 08:00–19:00). |
| 5.2 | **No monitoring / alerts** for the endpoint, Lakebase or job failures. | Alerts on failed jobs and endpoint errors; a usage / cost view from `system.billing`. |

## 6. Hygiene

| # | Gap | Done when |
|---|---|---|
| 6.1 | **Diagnostics live only in `scratch/`** (gitignored): `baseline_check.py`, `nuisance_probe.py`, `ref_recipe.py`, `score_test.py`, `app_smoke.py`. The CPU findings in PLAN.md can't be reproduced from the repo. | Moved under e.g. `tools/`, documented. |
| 6.2 | **Proxy-driven pins** to revisit: Vite 7.3.5 + an npm `overrides` for rollup 4.59.1 (`app/package.json`), `boto3>=1.34,<1.43` (`pyproject.toml`). The internal package proxies block the newest releases. | Relaxed once the proxies allow them. |
| 6.3 | **`uv.lock` not committed** (it resolves through the internal PyPI proxy). | A portable lock generated off-proxy, or accepted as-is. |
| 6.4 | **CI runner image**: `ubuntu-latest` moves to Ubuntu 26 from 2026-10-19 (GitHub notice). | Next CI run after that date green. |
| 6.5 | **Deferred code-review items** (eval tracking): each tracking step recomputes the full G1 metrics (KMeans, N×N sort), and the final tracked point duplicates the job's final eval. Cheap at the current eval size. | A light tracking path if eval size or tracking frequency grows. |
| 6.6 | **Test split use.** Test was scored once for `wafer_encoder` v2 (provisional G1). Every future G1 candidate must likewise be scored on test **once**, after its development is frozen on val. | Recorded per registered version (version tags). |
