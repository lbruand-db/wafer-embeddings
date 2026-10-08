# Build Plan — Wafer-Map Embedding Model (WM-811K)

Companion to [`SPEC/SPECS.md`](./SPECS.md). This is the **execution plan**: how we build
it, sequenced **risk-first**. Each early step is a cheap *spike* that tries to **kill or
confirm** a top assumption before we invest in polished infrastructure. Kill/confirm
signals come from the **§16 de-risking checklist** and the **§8.9 acceptance gates** in the
spec (section numbers below refer to `SPECS.md`).

**Progress (2026-10-08):** P0 + P1 scaffolding is code-complete (73 CPU unit tests green,
CI green) and the **GPU training path is proven end-to-end on AI Runtime** (torch
2.14.1+cu130, parquet read, DINO → embed → G1 → MLflow). Remaining on P1: a full Gate-G1
training run to clear the numeric gate. P2–P5 not started.

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
| **R1** | DINO-from-scratch doesn't learn a useful embedding on WM-811K | §16 1–3; §8 | **P1** |
| **R3** | Per-die attention infeasible on giant maps / ISAB loses quality | §16 4–6; §4 | **P2** |
| **R6** | Variable-size + invariance don't generalize | §16 7–10; §5 | **P2** |
| **R4** | Serverless-GPU AI Runtime training path / env not usable | §16 16–17; §11.3 | ✅ **P3** (proven) |
| **R5** | GPU model-serving packaging / custom pyfunc won't deploy | §16 18–20; §11.5 | **P3** |
| **R2** | Lakebase Search (Public Preview) can't do 384-d ANN at p99 < 50 ms | §16 11–14; §9 | **P4** *(deferred)* |
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
   content-hash **dedup** (696,599 unique of 811,457), leakage-safe splits (§16 21). ✅ done
2. **Tokenizer** (§5): on-wafer dies → (pass/fail) feature + center/radius coords; drop
   no-die; **token cap** for giant maps; padding + mask. (Pass/fail = the degenerate case
   of multi-scheme binning.) ✅ done
3. **Per-die ViT (patch=1), ISAB** + grad checkpointing; DINO self-distillation (weight-
   normed prototype head + LR/teacher-temp/momentum schedules); MLflow logging; live
   collapse diagnostics (§8.4) (§16 2). ✅ built; GPU run proven on AI Runtime
4. **Eval harness** (§8), all on **exact** search: kNN probe, clustering ARI / NMI /
   silhouette, retrieval mAP@k, effective-rank collapse check, vs **frozen DINOv2** +
   flattened-pixel/PCA baselines (§16 1, 24). ✅ `g1_metrics` built; baselines to wire
5. **False-negative check:** pure DINO vs. SWaCo same-class positives on retrieval (§16 3).
- **🚦 Gate G1 (go / no-go):** clears the §8.9 starting gate (mAP@10 ≥ 0.80, NMI ≥ 0.85),
  within reach of the 0.87 external reference on WM-811K [9], **no dimensional collapse**.
  **← current step: a full Gate-G1 training run (billable, needs go-ahead).**
  - **If NO →** pivot: stronger/different augmentation, same-class positives, more
    pretraining steps / larger model, or (last resort) a supervised encoder — then re-run G1.
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

### P3 — R4 + R5: Training-at-scale + model serving on the platform
- **AI Runtime training path (§16 16):** ✅ **done** — `train` runs reproducibly via
  `databricks air submit ai_runtime/train.yaml` on serverless GPU; torch/CUDA confirmed on
  the env, code shipped via `code_source.snapshot`. Remaining: register the encoder to UC.
- **Model Serving (§16 18–20):** log/register the pyfunc **from a GPU runtime**; deploy
  `GPU_SMALL`; endpoint healthy (no `DATABRICKS_ACCELERATOR` fail-fast); verify CPU
  viability for ~2k-token wafers; pyfunc (tokenizer + encoder) returns a 384-d vector.
- **Exit:** train → register → real-time embed all run from code (batch embedding → Delta
  too). Retrieval still served by exact search over the Delta embeddings at this stage.

### P4 — R2: Lakebase Search online retrieval *(deferred — when an online store is needed)*
Only once the model is proven (G1) and we actually need low-latency online ANN at scale.
**Requires provisioning (billable) → explicit go-ahead.**
- **Provision** Lakebase instance programmatically (`database create-database-instance`,
  Public Preview); enable Lakebase Search (§16 11).
- `lakebase_vector` + `lakebase_ann` on `vector(384)`, cosine; UC synced table → Postgres
  `vector(384)` (§16 11, 12).
- Backfill ~700K embeddings; **ANN recall@10 vs. exact ≥ 0.95** (§16 13); **p99 < 50 ms** at
  ~700K + synthetic 1M (§16 14).
- **🚦 Gate G2:** recall + latency met. **If NO →** raise Lakebase-preview limits with
  stakeholders; the spec mandates Lakebase Search, so **no silent substitution**.

### P5 — Harden into the reproducible template (N7) + acceptance
- Wire P0–P4 into one bundle: jobs (ingest, train, batch-embed, lakebase-sync, eval), the
  serving endpoint, the Lakebase instance, **bootstrap + teardown** (§11.8); config-driven
  (§11.7).
- Final §8 eval at the agreed gates; qualitative galleries / UMAP (§8.8); size + (deferred)
  binning hooks.
- **Deliverable** = the M1–M5 milestones (§12) as reproducible code.

---

## Critical path

```
P0 ─ P1 (G1) ─ P2 ─ P3 ─ [P4 Lakebase, when needed] ─ P5
```
- Single early critical path: **P0 → P1 (G1)**. Nothing else matters until the model learns.
- **Lakebase (P4) is off the early path** — pulled in only when an online vector store is
  genuinely required; retrieval quality is proven with exact search in P1–P3.

---

## Open decisions that gate steps
- **§8.9 numeric gates** — confirm or replace the 0.80 / 0.85 starting targets (gates G1).
- **Pivot menu if G1 fails** — stronger/different augmentation vs. same-class positives vs.
  more pretraining steps / larger model vs. (last resort) supervised encoder.
- **Token cap value** — default ~4k; tune against the giant-map tail vs. accuracy (§16 4).
- **When do we need Lakebase?** — triggers P4 + provisioning (deferred; not needed early).
- **Dataset for binning** — none yet; R7 stays deferred until real multi-scheme bin maps
  exist (§16 22).
