# Build Plan — Wafer-Map Embedding Model (WM-811K)

Companion to [`SPEC/SPECS.md`](./SPECS.md); open work is listed in
[`GAPS.md`](./GAPS.md). This is the **execution plan**: how we build
it, sequenced **risk-first**. Each early step is a cheap *spike* that tries to **kill or
confirm** a top assumption before we invest in polished infrastructure. Kill/confirm
signals come from the **§16 de-risking checklist** and the **§8.9 acceptance gates** in the
spec (section numbers below refer to `SPECS.md`).

**Progress (2026-10-09):** P0 done. **P1 in progress:** after fixing the data / eval
protocol (lot-grouped splits, val-only development, cross-device metrics, a training-free
polar baseline) and aligning the DINO recipe with the reference implementation, training
**clearly beats the untrained encoder** (val cross-device kNN macro recall 0.276 → 0.405)
but **not yet the polar baseline** (0.468); see P1 "Status". **End-to-end path built
ahead of G1** on a provisionally gated model (user decision): UC model → Model Serving →
batch embeddings → Lakebase Search → Vue/FastAPI Databricks App, all from code (see
"End-to-end path"). That covers most of P3 and P4; P2 not started. 183 Python + 3 JS unit
tests (CPU-only, seconds), CI green.

---

## Principles

- **Fail fast, cheap.** Prove the risky core on WM-811K with minimal scaffolding; harden
  into the N7 reproducible bundle only after the kill-risks are retired.
- **The one early existential risk is modeling:** *can we learn a useful embedding at all?*
  Everything else (Lakebase, serving, scale) is downstream and only matters once that holds.
- **No Lakebase in the beginning.** Retrieval *quality* is validated with **exact search**
  (FAISS inner-product on L2-normalized vectors — §8.1/§8.6, ref [19]), which needs no
  vector store. Lakebase Search (the online ANN index) is a **later serving/scale concern**,
  not a gate on whether the model works.
- **Reproducible from day one (N7, §11.8).** Even spikes land in the bundle / bootstrap as
  idempotent code (no click-ops), so nothing is thrown away.
- **Don't optimize before it learns.** Build the ISAB / scaling work only *after* the model
  is shown to learn with simple full attention.

---

## Risk ranking (attack order)

| # | Risk (kill potential) | Spec refs | Retired in |
|---|---|---|---|
| **R1** | DINO-from-scratch doesn't learn a useful embedding on WM-811K | §16 1–3; §8 | **P1** — partly retired: learns (beats untrained), not yet above the polar bar |
| **R3** | Per-die attention infeasible on giant maps / ISAB loses quality | §16 4–6; §4 | **P2** |
| **R6** | Variable-size + invariance don't generalize | §16 7–10; §5 | **P2** |
| **R4** | Serverless-GPU AI Runtime training path / env not usable | §16 16–17; §11.3 | ✅ **P3** (proven) |
| **R5** | GPU model-serving packaging / custom pyfunc won't deploy | §16 18–20; §11.5 | ✅ **CPU** (pyfunc deploys, matches local to 1e-7); GPU tier untested |
| **R2** | Lakebase Search (Public Preview) can't do 384-d ANN at p99 < 50 ms | §16 11–14; §9 | **mostly retired**: `lakebase_ann` works at 696,599 × 128-d, app top-k ~40–120 ms incl. network; still to measure: recall@10 vs exact, server-side p99, 384-d, 1M |
| **R7** | Multi-scheme binning (needs non-WM-811K data) | §16 22–23; §5 | deferred |

> R2 (Lakebase) drops down the order deliberately: it's a real risk but **not** on the path
> to proving the model, because exact search covers retrieval quality until an online store
> is actually needed.

---

## Phases

### P0 — Minimal reproducible substrate *(enabling)*
Just enough green scaffold to run spikes, built the N7 way so nothing is throwaway.
- `databricks bundle init default-python`; repo layout per §11.2; pin a **modern serverless
  `environment_version`** + uv deps (the base env is old — §16 item 17 finding).
- **Bootstrap job** (idempotent, SDK/CLI): `USE CATALOG mmf_mlops_demo_catalog`; create
  schema `wafer_embeddings` + volume `raw` if absent (§11.8). *(No Lakebase yet.)*
- **Retires:** §16 item 15 (✅ already verified), item 17 (seed), N7 seed.
- **Exit:** `bundle deploy` + bootstrap runs clean on `fevm-mmf-mlops-demo`; schema/volume
  present *via code*.

### P1 — ⚑ R1: Does the representation learn? *(make-or-break spike)*
Confirm a per-die ViT + DINO learns a useful embedding on WM-811K using **ISAB attention +
token cap** (WM-811K's giant maps make full attention OOM, so ISAB is the default) and
**exact search for eval** (no Lakebase).
1. **Ingest** WM-811K → volume → Delta + parquet: **stream** `LSWMD.pkl` (variable-size
   `waferMap` {0,1,2}, `failureType` → 9 single-label classes); clean messy labels,
   content-hash **dedup** (696,599 unique of 811,457), leakage-safe **lot-grouped** splits
   (45,345 lots; no lot straddles splits — enforced at ingest) (§16 21). ✅ done
2. **Tokenizer** (§5): on-wafer dies → (pass/fail) feature + center/radius coords; drop
   no-die; **token cap** for giant maps; padding + mask. (Pass/fail = the degenerate case
   of multi-scheme binning.) ✅ done
3. **Per-die ViT (patch=1), ISAB** + grad checkpointing; DINO self-distillation (weight-
   normed prototype head + LR/teacher-temp/momentum schedules); MLflow logging; live
   collapse diagnostics (§8.4) (§16 2). ✅ built; GPU run proven on AI Runtime
4. **Eval harness** (§8), all on **exact** search: kNN probe (+ macro and per-class
   recall), clustering ARI / NMI / silhouette, retrieval mAP@k, effective rank;
   **cross-device `xgroup_*` metrics** (same-shape neighbours excluded); every run also
   scores the **untrained encoder** and the **polar FAIL-histogram** baseline; student
   *and* teacher; MLflow as `<split>/<model>/<metric>` curves. ✅ done. Frozen DINOv2 /
   pixel-PCA baselines (§16 1, 24) still to wire.
5. **Label protocol** (SPECS §8.7): labels evaluation-only (no labels in training or
   checkpointing); develop on **val**; score **test once** for the final report. ✅ enforced
   in code (`--eval-split`, keep-best off by default, no tracking on test).
6. **False-negative check:** pure DINO vs. SWaCo same-class positives on retrieval (§16 3).

**Status (2026-10-09).** What we learned getting here:
- **Collapse → fixed.** Early GPU runs pinned at ln(K) (uniform collapse); fixed by head
  sizing, teacher temp, prototype freeze, and a **defect-preserving token cap** (keep all
  FAIL dies). Collapse gauges (teacher entropy / max-prob, embedding std, grad norm) are
  logged every 50 steps.
- **The eval was leaky.** Per-map splits put one lot on both sides (same map shape → same
  label 75% vs 22%); a shape one-hot alone matched the untrained encoder. Fixed with
  lot-grouped splits + cross-device metrics. All results before 2026-10-09 are superseded.
- **The real bar is the polar baseline**, a training-free rotation-invariant FAIL-density
  descriptor; it beats every encoder so far.
- **Ruled out as the lever:** LR / clipping / freeze timing, training length alone,
  Fourier bands, coordinate jitter, stronger augmentation (FAIL / token dropout). Wafer
  size was *not* the shortcut.
- **The recipe was.** A line-by-line comparison with the reference DINO found ~10
  deviations. Adopting it (multi-crop 2 global + 6 local, random crop area, pre-LN blocks,
  cosine WD 0.04 → 0.4, LR scaling rule, 3% die noise, head 2048/256) gave the first real
  gains; pre-LN and local crops each add ~4 points.

| Val, cross-device | macro recall | precision@10 |
|---|---|---|
| untrained encoder | 0.276 | 0.277 |
| old recipe, 2k steps | 0.288 | 0.278 |
| reference recipe, 2k steps (w128 d6) | 0.360 | 0.312 |
| **reference recipe, 10k steps (w128 d6)** | **0.405** | **0.352** |
| **polar baseline (target)** | **0.468** | **0.392** |

The 10k-step small run plateaus from ~4k steps. **Capacity does not lift it at this
budget:** the full-size run (w384 d12, same recipe and steps) ends at 0.407 / 0.348, the
same as the small model, at ~3x the cost (it starts slower and was still creeping up as
the LR decayed). Recipe work continues at the small size.

**Next steps (CPU-tested, GPU-run with go-ahead):**
1. ~~Capacity test~~ done: no gain at this budget (revisit with a tuned LR / longer
   schedule once the recipe beats polar).
2. ~~RankMe~~ done: label-free effective rank on a fixed train probe set, tracked for
   student and teacher (rises ~4 untrained → ~12 trained).
3. ~~Cheap levers~~ tried: batch 64 (0.397 / 0.352) and 1,024 tokens (0.422 / 0.330 vs
   polar 0.501 / 0.359 on its eval) — no gain over the baseline gap. RankMe rises ~4 → ~12.
4. **Descriptor-guided positives**: polar-histogram nearest neighbours as extra DINO /
   InfoNCE positives (NNCLR-style, label-free); plus a plain InfoNCE baseline.
5. If still short: Sonata-style masked self-distillation (anti geometric shortcut); or the
   label-free polar + learned concatenation as a safety net.

- **🚦 Gate G1 (go / no-go), revised:** a learned, label-free encoder that **beats the
  polar baseline on the val cross-device metrics** (macro recall and precision@10, beyond
  seed noise), with no dimensional collapse; then confirmed **once on test**. The old
  absolute targets (mAP@10 ≥ 0.80, NMI ≥ 0.85) and the external 0.87 NMI reference [9]
  are not comparable to our protocol: [9] mines positives with handcrafted similarity
  that includes lot information, uses label-stratified batches and random splits. They
  stay as long-term aspirations, not the go / no-go.
- *Bonus:* P1 training already runs on serverless GPU → ✅ R4 env retired early.

### P2 — R3 + R6: Scale & generalization *(after G1)*
- **Attention feasibility (§16 4):** benchmark full vs. **ISAB** at median (~1–2k) and
  capped (~4k) tokens on an A10 (wall-clock, peak memory, encode latency). ✅ partial: ISAB
  + cap + grad checkpointing already needed to fit WM-811K's giant maps.
- **ISAB quality (§16 5)** and **patch 1 vs 2 (§16 6)** ablations — keep quality, pick the
  smallest that passes (N4).
- **Variable-size generalization (§16 7, 8):** hold out whole size groups + token-cap
  sensitivity; same pattern across sizes → nearby embeddings (WM-811K's native size spread
  exercises this directly).
- **Invariance (§16 9, 10):** rotation/flip view-consistency; verify crops don't confuse
  Loc ↔ Edge-Loc; padding-invariance unit tests.
- **Exit:** ISAB retains quality and makes giant maps feasible within budget; embeddings stable
  across size; chosen invariances hold. (Still exact-search eval — no Lakebase.)

### End-to-end path — built 2026-10-09 on the provisional G1 model (R1-long-small)
The user gated R1-long-small provisionally (test, once: cross-device 0.389 / 0.356 vs
polar 0.442 / 0.359; beats polar on map@10 0.346 vs 0.315) to prove the full stack:
- ✅ **UC model** `mmf_mlops_demo_catalog.wafer_embeddings.wafer_encoder` v2 `@champion`
  (`jobs/register.py`: pyfunc over the CI-tested `serving.WaferEncoder`, 128-d).
- ✅ **Model Serving** `wafer-encoder` (bundle `resources/serving.yml`; CPU Small,
  scale-to-zero); output matches the local encoder to 1e-7.
- ✅ **Batch embeddings**: `ai_runtime/embed.yaml` (GPU, 696,599 maps in 7.5 min) → volume
  parquet → bundle job `embed` → Delta `wafer_map_embeddings` (CDF on). (In-Spark
  embedding OOMs on serverless Python workers importing CUDA torch.)
- ✅ **Lakebase**: project `wafer-embeddings` (PG 17), database `wafer_embeddings`; UC
  synced table → Postgres `wafer_embeddings.wafer_map_embeddings_pg` (`vector(128)`).
- ✅ **Lakebase Search index**: cosine `lakebase_ann` (2,919 IVF lists); app top-k ~40–120 ms.
- ✅ **Databricks App** `wafer-search` (bundle `resources/app.yml`): pick a wafer, see its
  map, top-k neighbours (optionally from other lots), optional live re-embedding via the
  endpoint.
- ✅ **N7 for Lakebase**: the bootstrap job's `lakebase` task does project → Search
  (REST `search-extensions`, found via DevTools) → database → `lakebase_vector` →
  synced table → index → grants → plan check, idempotently; proven on a fresh project.
  No direct pgvector. Remaining gaps: model registration runs locally (`jobs/register.py`),
  and there's no `CREATE CATALOG` on the metastore (synced table lives in the main catalog).

### P3 — R4 + R5: Training-at-scale + model serving on the platform
- **AI Runtime training path (§16 16):** ✅ **done** — `train` runs reproducibly via
  `databricks air run --file ai_runtime/train.yaml` on serverless GPU; torch/CUDA confirmed on
  the env, code shipped via `code_source.snapshot`.
- **UC registration:** ✅ `jobs/register.py` (pyfunc over `serving.WaferEncoder`, CPU torch
  wheels, signature + input example, `@champion` alias, gate metrics as version tags).
  Remaining: run it as a job instead of locally.
- **Model Serving (§16 18–20):** ✅ **CPU** endpoint (Small, scale-to-zero) healthy; the
  pyfunc returns the 128-d vector, matching the local encoder to 1e-7, ~1 s per call. Not
  done: the `GPU_SMALL` tier and its log-from-a-GPU-runtime packaging gotcha (only needed
  for large models / high throughput), and the 384-d encoder the spec targets (N1).
- **Batch embedding:** ✅ on AI Runtime (`ai_runtime/embed.yaml`, 696,599 maps in 7.5 min) +
  bundle job `embed` (parquet → Delta, CDF on). In-Spark embedding OOMs on serverless.
- **Exit:** ✅ train → register → real-time + batch embedding run from code (registration
  still a local command).

### P4 — R2: Lakebase Search online retrieval *(built early with the end-to-end path)*
- ✅ **Provisioning, fully automatic + idempotent** (`jobs/lakebase.py`, bootstrap job task
  `lakebase`): Lakebase **project** (Autoscaling API, PG 17) → **Lakebase Search** via
  `POST /api/2.0/postgres/projects/{p}/search-extensions` (the UI button's call) → database
  → `CREATE EXTENSION lakebase_vector CASCADE` (no direct pgvector) → UC **synced table**
  (Delta → `vector(128)`) → cosine **`lakebase_ann`** index → app grants → plan check.
  Computes suspend after 1 h idle. Proven on the live project and on a fresh one (§16 11, 12).
- ✅ Backfilled 696,599 embeddings; the embed job re-syncs with `--refresh`.
- ⏳ **Still to measure for G2:** **ANN recall@10 vs. exact ≥ 0.95** (§16 13) and
  server-side **p99 < 50 ms** at ~700K + synthetic 1M (§16 14); redo at 384-d once a 384-d
  encoder ships.
- **🚦 Gate G2:** recall + latency met. **If NO →** tune `lakebase_ann.probes` / `epsilon` /
  `build_mode`, then raise Lakebase-preview limits with stakeholders; the spec mandates
  Lakebase Search, so **no silent substitution**.

### P5 — Harden into the reproducible template (N7) + acceptance
- Wire P0–P4 into one bundle: jobs (ingest, train, batch-embed, lakebase-sync, eval), the
  serving endpoint, the Lakebase instance, **bootstrap + teardown** (§11.8); config-driven
  (§11.7). **Done:** bundle jobs `bootstrap` (UC + Lakebase), `ingest`, `embed` (+ Lakebase
  sync), the serving endpoint and the app; AI Runtime workloads for train / embed.
  **Remaining:** registration as a job, a one-shot orchestration of train → register →
  embed, a **teardown** target, and dim / version wiring derived from the model instead
  of the `embed_dim` / `encoder_version` bundle variables.
- Final §8 eval at the agreed gates; qualitative galleries / UMAP (§8.8); size + (deferred)
  binning hooks.
- **Deliverable** = the M1–M5 milestones (§12) as reproducible code.

---

## Critical path

```
P0 ─ P1 (G1) ─ P2 ─ P3 ─ [P4 Lakebase, when needed] ─ P5
```
- Single early critical path: **P0 → P1 (G1)**. Nothing else matters until the model learns.
- **Deviation (2026-10-09, user decision):** P3 + P4 were built early on a provisionally
  gated model to prove the platform path end to end. They don't depend on which encoder
  wins: a better model is a new UC version → `encoder_version` bump → re-embed → re-sync.
  The modelling critical path (beat polar → real G1) is unchanged.

---

## Open decisions that gate steps
- **G1 definition** — proposed: beat the polar baseline on val cross-device metrics (see
  P1); the plain mAP@10 still rewards same-lot neighbours inside val, so redefine it as
  cross-lot retrieval or drop it from the gate.
- **Pivot menu if the plateau holds** — descriptor-guided positives vs. masked
  self-distillation vs. polar + learned concatenation vs. (last resort) supervised encoder.
- ~~**Serve now or later?**~~ decided: served now, with the learned (provisional) encoder.
- **Embedding width** — the served encoder is 128-d (the spec's N1 says 384); keep 128 (the
  full-size model brought no gain) or ship 384 with the next real G1 model?
- **Registration as a job** — `jobs/register.py` runs locally today; make it a bundle /
  AI Runtime task so train → register → embed is one orchestrated flow.
- **Token cap value** — GPU job default ~4k; recipe experiments use 512 for speed; tune
  against the giant-map tail vs. accuracy (§16 4).
- ~~**When do we need Lakebase?**~~ decided: now (it backs the search app).
- **Dataset for binning** — none yet; R7 stays deferred until real multi-scheme bin maps
  exist (§16 22).
