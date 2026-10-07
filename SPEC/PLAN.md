# Build Plan — Wafer-Map Embedding Model (MixedWM38)

Companion to [`SPEC/SPECS.md`](./SPECS.md). This is the **execution plan**: how we build
it, sequenced **risk-first**. Each early step is a cheap *spike* that tries to **kill or
confirm** a top assumption before we invest in polished infrastructure. Kill/confirm
signals come from the **§16 de-risking checklist** and the **§8.9 acceptance gates** in the
spec (section numbers below refer to `SPECS.md`).

---

## Principles

- **Fail fast, cheap.** Prove the risky core on MixedWM38 with minimal scaffolding; harden
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
| **R1** | DINO-from-scratch doesn't learn a useful embedding on ~38K maps | §16 1–3; §8 | **P1** |
| **R3** | Per-die attention infeasible at 100×100 / ISAB loses quality | §16 4–6; §4 | **P2** |
| **R6** | Variable-size + invariance don't generalize | §16 7–10; §5 | **P2** |
| **R4** | Serverless-GPU AI Runtime training path / env not usable | §16 16–17; §11.3 | **P3** |
| **R5** | GPU model-serving packaging / custom pyfunc won't deploy | §16 18–20; §11.5 | **P3** |
| **R2** | Lakebase Search (Public Preview) can't do 384-d ANN at p99 < 50 ms | §16 11–14; §9 | **P4** *(deferred)* |
| **R7** | Multi-scheme binning (needs non-MixedWM38 data) | §16 22–23; §5 | deferred |

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
Confirm a per-die ViT + DINO learns a useful embedding on MixedWM38 using **full attention**
(cheap at ~2k tokens — no ISAB yet) and **exact search for eval** (no Lakebase).
1. **Ingest** MixedWM38 → volume → Delta (`arr_0` 52×52 {0,1,2}, `arr_1` 8-d labels);
   **dedup** GAN near-duplicates; inductive train/val/test splits (§16 21).
2. **Tokenizer** (§5): on-wafer dies → (pass/fail) feature + center/radius coords; drop
   no-die; padding + mask. (Pass/fail = the degenerate case of multi-scheme binning.)
3. **Per-die ViT (patch=1), full attention**; DINO self-distillation; MLflow logging;
   live collapse diagnostics (§8.4) (§16 2).
4. **Eval harness** (§8), all on **exact / FAISS** search: kNN / linear probe (micro/macro
   F1), clustering ARI / NMI / silhouette, retrieval mAP@k, vs **frozen DINOv2** +
   flattened-pixel/PCA baselines (§16 1, 24).
5. **False-negative check:** pure DINO vs. SWaCo same-class positives on retrieval (§16 3).
- **🚦 Gate G1 (go / no-go):** clears the §8.9 starting gate (mAP@10 ≥ 0.80, NMI ≥ 0.85),
  within reach of the 0.87 external reference, **no dimensional collapse**.
  - **If NO →** pivot: unlabeled **WM-811K pretraining**, stronger/different augmentation,
    same-class positives, or (last resort) a supervised encoder — then re-run G1.
- *Bonus:* P1 training already runs on serverless GPU → first real read on R4/R5 env.

### P2 — R3 + R6: Scale & generalization *(after G1)*
- **Attention feasibility (§16 4):** benchmark full vs. **ISAB** at ~2k and synthetic ~8k
  tokens on an A10 (wall-clock, peak memory, encode latency).
- **ISAB quality (§16 5)** and **patch 1 vs 2 (§16 6)** ablations — keep quality, pick the
  smallest that passes (N4).
- **Variable-size generalization (§16 7, 8):** hold out whole grid-size groups + synthetic
  resized wafers; same pattern across sizes → nearby embeddings.
- **Invariance (§16 9, 10):** rotation/flip view-consistency; verify crops don't confuse
  Loc ↔ Edge-Loc; padding-invariance unit tests.
- **Exit:** ISAB retains quality and makes 100×100 feasible within budget; embeddings stable
  across size; chosen invariances hold. (Still exact-search eval — no Lakebase.)

### P3 — R4 + R5: Training-at-scale + model serving on the platform
- **AI Runtime training path (§16 16):** make `train` a reproducible DAB/bootstrap-driven
  run on serverless GPU (or the AI Runtime workload CLI if DAB can't launch it); confirm
  **torch / CUDA / timm on the GPU env** (§16 17-GPU).
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
- Backfill 40K embeddings; **ANN recall@10 vs. exact ≥ 0.95** (§16 13); **p99 < 50 ms** at
  40K + synthetic 1M (§16 14).
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
- **Pivot menu if G1 fails** — WM-811K pretrain vs. augmentation vs. same-class positives
  vs. supervised encoder.
- **When do we need Lakebase?** — triggers P4 + provisioning (deferred; not needed early).
- **Dataset for binning** — none yet; R7 stays deferred until real multi-scheme bin maps
  exist (§16 22).
