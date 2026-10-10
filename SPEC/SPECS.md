# Spec: Wafer-Map Embedding Model Template (WM-811K)

**Status:** Draft · **Date:** 2026-10-08 · **Owner:** lbruand-db

A reusable Databricks template that trains a **small, per-die vision-transformer
embedding model** for semiconductor wafer maps, backfills a vector corpus into
**Lakebase Search**, and serves embeddings both in **batch** and **real time**.
Primary dataset: **[WM-811K](http://mirlab.org/dataSet/public/MIR-WM811K.zip)** (~811K
real, variable-size maps). [MixedWM38](https://github.com/Junliangwangdhu/WaferMap)
(fixed 52×52, multi-label) is retained as a secondary reference corpus — the data layer
loads both.

---

## 1. Goal

Produce a compact (384-d) embedding for a wafer map such that:

- **Similarity search / retrieval** — "find wafers that look like this one" over a
  Lakebase Search index returns same-defect-pattern neighbors.
- **Clustering / pattern discovery** — embeddings cluster into recognizable defect
  signatures without labels.

Design priorities, in order: **per-die fidelity** (every die is a first-class unit) →
**small/fast embedding for Lakebase** → **smallest model that reaches the quality bar**.
Labels are used **for evaluation only**; the representation is learned/served label-free.

---

## 2. Background & dataset

**WM-811K (primary)**: **811,457** real wafer maps collected from ~47K lots, of which
**172,950 are labeled** with one of **9 classes** — 8 single-defect patterns (`Center`,
`Donut`, `Edge-Loc`, `Edge-Ring`, `Loc`, `Near-full`, `Random`, `Scratch`) plus `none`
(labeled defect-free). It is **single-label** (one class per map, unlike MixedWM38's
multi-label co-occurrence). The remaining **~638K maps are unlabeled** — ideal fuel for
label-free DINO pretraining; labels drive **evaluation only**. Ships as `LSWMD.pkl`, a
pickled pandas DataFrame (`waferMap`, `failureType`, `dieSize`, `lotName`,
`waferIndex`, `trianTestLabel`); each `waferMap` cell is **3-valued**: `0` no-die
(outside wafer), `1` passing die, `2` failing die.

> **Variable wafer size is a hard requirement, handled natively.** Unlike MixedWM38's
> fixed 52×52, WM-811K maps genuinely vary — from a handful of dies to **tens of
> thousands**. **No scaling/resizing is used**: resampling to a fixed grid distorts
> individual dies, and every die matters. The model consumes the **native grid** directly
> (§5), and giant maps are **token-capped** to bound GPU memory rather than downsampled
> (§4).

> **Binning is richer in production than in WM-811K.** Real wafer test assigns each die a
> **bin code**, often under several schemes at once — e.g. **electrical/parametric** and
> **functional** binning. WM-811K's pass/fail is just the degenerate single-scheme case.
> The per-die token design (§4) must therefore carry **multiple bin schemes per die
> simultaneously**, fused into one embedding (§5). This is forward-design: the
> architecture supports it, but WM-811K can only exercise the pass/fail special case.

> **MixedWM38 (secondary reference):** 38,015 synthetic maps, fixed **52×52**, **38
> classes** (1 normal + 8 single-defect + 29 mixed-defect), **multi-label**. The data
> layer still parses it (useful for the multi-label eval path, §8), but WM-811K is the
> primary train/eval corpus.

---

## 3. Requirements

### Functional
- F1. Ingest WM-811K into Unity Catalog; keep maps at **native resolution** (no resizing).
- F2. Produce a **384-d, L2-normalized** embedding per wafer map, **per-die** at native res.
- F3. Batch-embed the full corpus and load it into a **Lakebase Search** index.
- F4. Serve single-wafer embeddings in real time via a **Model Serving** endpoint.
- F5. Nearest-neighbor query API over Lakebase (cosine), returning top-k similar wafers.
- F6. Evaluation harness reporting retrieval + clustering quality against the 9 WM-811K
  labels (single-label); multi-label scoring retained for the MixedWM38 reference path.

### Non-functional
- N1. Embedding dim fixed at **384** (balanced size vs. quality for Lakebase).
  **Interim deviation (2026-10-09):** the model served today, `wafer_encoder` v2
  (provisional G1), is a **128-d** small encoder (width 128, depth 6, 2.4M parameters).
  The full-size 384-d model brought no gain at the same budget (PLAN.md), so 128-d stays
  until the next real G1 model; the Lakebase column is `vector(128)`. The serving,
  Lakebase and app code is width-agnostic: the dimension D follows the registered model
  (bundle variable `embed_dim`), and moving to 384 is a re-register → re-embed → re-sync.
- N2. Similarity query **p99 < 50 ms** at the serving corpus scale.
- N3. Corpus ~700K vectors (696,599 unique maps ingested after dedup); index design valid
  to **1–10M** vectors.
- N4. **Smallest model that clears the eval bar** — minimize ViT depth/width/token count,
  not by skipping training (training is mandatory here, see §4).
- N5. Runs on the existing FEVM **MMF MLOps-demo** workspace
  (`fevm-mmf-mlops-demo.cloud.databricks.com`, AWS); training + batch embedding use
  **AI Runtime serverless GPU** (GPU is required, not optional). ✅ Training path proven
  end-to-end on AI Runtime (single A10) — see §11.3.
- N6. Model tracked in **MLflow**, registered in **Unity Catalog Model Registry**.
- N7. **Fully reproducible from scratch, programmatically.** Every artifact — schema /
  volume, the Lakebase instance + `lakebase_ann` index, data ingest, training,
  UC-model registration, and the serving endpoint — is created by **code** (DAB + a
  bootstrap job using the Databricks SDK/CLI), **idempotently and with no manual UI steps**.
  A single `bundle deploy` + bootstrap entrypoint stands the project up on a clean
  workspace, and a teardown path removes it (§11.8).

---

## 4. Model approach & literature justification

**Encoder: a custom per-die ViT (ViT-S scale, 384-d output), patch size = 1, trained
with DINO self-distillation on wafer maps.** No pretrained DINOv2 reuse.

**Why this, not frozen DINOv2.** The widely variable-size + per-die-fidelity
requirements rule out DINOv2 ViT-S/14: patch 14 blends a 14×14 block of dies into one
token (coarse) and cannot represent a tiny wafer (smaller than one patch). Setting
**patch = 1 makes each die exactly one token**, preserving per-die information and
supporting any grid size natively — but it discards DINOv2's pretrained patch
embeddings, so the model must be **trained from scratch**. There is therefore **no
frozen Phase-A baseline**; training is part of the critical path.

**Why ViT + DINO specifically.** A patch-based transformer is the natural fit for
variable-length, per-die token sequences, and DINO is a strong **label-free** objective
whose features suit retrieval and clustering (we train unlabeled; labels are eval-only).

**Token construction (key design point).** Tokenize **only on-wafer dies** — drop
`no-die` cells (outside the wafer) rather than feed them. This (a) naturally represents
ragged/circular wafer shapes, and (b) cuts token count well below `H×W`
(a wafer disc fills ≈ π/4 of its bounding square). WM-811K maps span a wide range — most
are a few hundred to a few thousand dies, but the largest reach **tens of thousands**, so
the sequence is **token-capped** (random subsample of valid dies above a cap, default
~4,096) to bound GPU memory; small maps are used whole. Each die's feature is a
**multi-scheme bin descriptor** (electrical + functional + …, fused — §5), not just
pass/fail; position comes from its center/radius-normalized coordinates. Per-die tokens
are what make multi-scheme binning clean — a new scheme is just another embedding table
added to the token, with no change to sequence length or attention cost.

**Positional embeddings.** Encode each die's position as **coordinates normalized by
the wafer center and radius** — `(ũ, ṽ) = ((x−x_c)/R, (y−y_c)/R)` plus radial `r` —
rather than raw grid indices, then feed through a 2D sinusoidal / small-MLP encoding.
This is the field's standard answer to size variation (Deep Sets / Set-Transformer /
point-cloud wafer work, §13) and lets the model extend to grid sizes unseen in training
across WM-811K's full size range.

**Attention mechanism.** Use **induced set attention (ISAB)** rather than full
self-attention. A patch=1 per-die ViT is effectively a Set Transformer over dies, and
ISAB reduces attention cost from **O(N²) → O(N·m)** for `m` inducing points — the direct
mitigation for the ~10k-token cost at 100×100 (Lee et al., 2019; §13, §14). Output via
**pooling-by-multihead-attention (PMA)** or a CLS/masked-mean readout → L2-normalize → 384-d.

**Compute feasibility (per-die token budget).** Dropping off-wafer `no-die` cells leaves
~79% (π/4) of a square grid as tokens; after the token cap the per-forward cost is bounded
regardless of a wafer's raw size:

| Grid | Raw cells | On-wafer tokens | After cap (≤4,096) |
|---|---:|---:|---:|
| ~26×26 | 676 | ~530 | ~530 |
| 52×52 | 2,704 | ~2,100 | ~2,100 |
| 100×100 | 10,000 | ~7,850 | 4,096 |
| giant (WM-811K tail) | 50k+ | tens of thousands | 4,096 |

With the cap the effective ceiling is **~4k tokens**. At ViT-S scale (d=384, 12 layers)
full self-attention costs ≈ `4·N²·d` per layer; at 4k tokens that is ~**0.4 TFLOP** per
wafer forward, and most maps are far smaller. The naïve `N×N` score matrix is the memory
concern, mitigated by **ISAB** (O(N·m), below), **gradient checkpointing**, and
memory-efficient SDPA. DINO pretraining over the ~700K-map WM-811K corpus is the main cost
driver (hours on one A10). **Guidance: full attention is fine for the small/median maps;
use ISAB (O(N·m)) once many maps approach the cap**, which is the realistic WM-811K regime
— hence ISAB is the default (§6). The Lakebase query p99 (N2) is independent of token
count — it is ANN over fixed 384-d vectors; token count affects only the one-time encode.

**Training objective caveat.** Pure instance-contrastive / DINO self-distillation can
treat two *different* wafers of the *same* defect pattern as negatives and push them
apart, which hurts same-pattern retrieval — our primary goal. Mitigation: evaluate
against, and optionally add, **same-class positives** (SWaCo-style, label-assisted) in a
fine-tune pass; benchmark both against the pure label-free encoder (§8).

**Literature grounding — see §13 (Related Work) for the full discussion and §14 for
citations.** In short: ViT is proven on MixedWM38 (supervised, coarse patches); wafer SSL
is established (WaPIRL, SWaCo, MAE, graph-contrastive); a patch=1 per-die encoder is
equivalent to a Set Transformer over dies; and the specific combination here (per-die ViT
+ DINO + native variable-res, for retrieval into Lakebase) is a **synthesis of published
pieces, not a reproduction of any single paper**.

**Simplicity lever.** "Simplest model" here = the **smallest ViT** (fewest layers /
smallest width / lowest effective token count) that clears the §8 threshold — tuned via
ablation, keeping the 384-d output.

---

## 5. Input representation (no resizing)

- **Per-die features (multi-scheme bin fusion).** Each on-wafer die is a **multi-field
  categorical descriptor**, not a single state. One **learned embedding table per binning
  scheme** — e.g. `E_elec(electrical_bin)`, `E_func(functional_bin)`, `E_soft(soft_bin)` —
  is looked up for the schemes present on that die and **fused** (sum, or concat →
  linear-project) to the model width `d`. Pass/fail is the degenerate single-scheme case,
  so WM-811K runs unchanged. `no-die` cells are excluded from the token set entirely (§4).
  - **Missing schemes:** a die (or whole wafer) lacking a scheme gets a learned
    **null / "scheme-absent" embedding** for that field; apply **modality dropout** during
    pretraining so the encoder is robust to absent schemes at inference.
  - **Cross-dataset comparability** requires a shared **canonical bin taxonomy** — see the
    open question in §15; per-scheme vocabularies and their canonical mapping are declared
    in config (§11.7).
  - *(Deferred extension: continuous per-die parametrics — Idd/Vmin/Fmax — via a small MLP
    added to the token; out of scope until parametric data is available, §15.)*
- **Positions.** Per-die coordinates **normalized by wafer center and radius**
  (`(x−x_c)/R, (y−y_c)/R`, plus radial `r`) → 2D sinusoidal / small-MLP positional
  encoding. Normalize by wafer geometry, **not** by the defect-cloud bounding box (which
  would move an edge cluster to look central), and **not** by raw grid index (which would
  not generalize across grid sizes). This is the standard treatment in set/point-cloud
  wafer models (§13).
- **No scaling, no interpolation.** The native grid is consumed as-is; a 10×10 map and a
  100×100 map differ only in sequence length.
- **Batching.** Variable-length sequences are padded to the batch max with an attention
  mask (padding tokens contribute nothing).
- **Augmentation (DINO multi-crop, die-semantics-preserving).** The augmentation set
  *defines the invariances the embedding learns*, so it is grounded in the wafer taxonomy,
  not copied from natural-image SSL. MixedWM38 / WM-811K classes are defined by defect
  **morphology and position relative to the wafer boundary — not absolute angle** (Edge-Loc
  = edge-localized, not "top edge"; Scratch = elongated, not horizontal-only), so rotation
  + flip are **label-preserving** — the standard practice in wafer SSL [2][30].
  - **Rotation + flip (orientation invariances — default on):** applied by rotating /
    reflecting the normalized `(ũ,ṽ)` die coordinates. Per-die point tokens make this
    **lossless at any angle** (no pixel resampling), avoiding the interpolation damage that
    forces image pipelines to quarter-turns only [30]. Config flag `orientation_invariant`
    (default true) disables them for future orientation-aware / process-diagnostic targets
    (scratch direction, edge sector) — §15.
  - **Crops → DINO multi-crop + variable-size views:** the strongest single augmentation in
    WaPIRL [2], and the source of the smaller/variable-size views that train the native
    variable-res path. **Caveat (grounded):** rotation-invariance does *not* imply crop /
    translation-invariance — Loc vs. Edge-Loc depends on position relative to the boundary,
    so use **wafer-centered, boundary-preserving** crops, keep a global view retaining the
    edge reference, and avoid the **crop+shift** combination (degraded in WaPIRL) [2][31].
  - **Die-state noise:** Bernoulli pass↔fail toggle on a small fraction of valid dies
    (p ≈ 0.001–0.01; WaPIRL used 0.05), categorical / masked — per-class check since thin
    scratches are fragile [2][16].
  - **Cutout / token dropout:** zero or drop tokens in ≤4 random regions [2].
  - **Rotation-twist (optional ablation):** radius-dependent rotation angle [32] — natural
    on continuous coordinates.
  - **Not used:** color jitter, blur, Gaussian-on-codes, arbitrary pixel resampling — they
    corrupt discrete die states [16].
  - **GAN augmentation (optional, orthogonal):** DCGAN / conditional-GAN minority synthesis
    [33] targets *class imbalance* for the supervised probe / same-class-positive fine-tune,
    not label-free DINO (which needs no class balance); judge on real-only test and preserve
    categorical states + valid mask.

  The rotation/flip invariance choice is itself an **ablation** (§8.7).

---

## 6. Architecture

```
wafer map (native H×W, 3-valued)
  → tokenize on-wafer dies: per-die (pass,fail) feature
      + center/radius-normalized coords → 2D sinusoidal position
  → custom ViT-S (patch=1) with ISAB induced attention — Set-Transformer over dies,
      O(N·m) cost, variable seq len + padding mask
  → PMA / CLS / masked-mean readout → 384-d → L2 normalize
  → {Delta table, Lakebase Search index, Model Serving response}
```

---

## 7. Data & training pipeline

1. **Ingest** — read the WM-811K `LSWMD.pkl` archive from UC Volume
   `/Volumes/mmf_mlops_demo_catalog/wafer_embeddings/raw/`; **stream** records (the pickle
   is large — never materialize the whole corpus), clean labels, content-hash **dedup**,
   assign leakage-safe **lot-grouped** splits, and write chunked into Delta
   `mmf_mlops_demo_catalog.wafer_embeddings.wafer_maps` (`id`, flattened `wafer_map`,
   `height`, `width`, `label_id` single int [−1 = unlabeled], `split`, `lot`). Also export a
   **parquet** copy to the volume for the Spark-less AI Runtime trainer to read (§11.3).
   *(Downloaded archive is untrusted: extract into its own empty dir, treat as data, never
   execute from it; the pickle is loaded in a hardened `-I` interpreter.)*
2. **Tokenize** — §5 transform (on-wafer die tokens + positions, token cap for giant maps).
3. **Train (mandatory)** — DINO self-distillation of the per-die ViT on **AI Runtime
   serverless GPU**; ISAB attention + gradient checkpointing + token cap to handle long
   sequences. Log loss / collapse / G1 metrics to MLflow; ablate model size.
4. **Register** — best encoder → UC Model Registry
   `mmf_mlops_demo_catalog.wafer_embeddings.wafer_encoder` (alias `@champion`).
5. **Batch embed** — computes D-dim vectors (D = 384 target, **128** for the interim
   model, N1) for the corpus → Delta `mmf_mlops_demo_catalog.wafer_embeddings.wafer_map_embeddings`.
6. **Sync to Lakebase** — see §9.

---

## 8. Evaluation & metrics

Labels (WM-811K: **9 classes, single-label**; MixedWM38 reference path: 38 classes,
multi-label) are used **for evaluation only** — the encoder trains label-free (§4), and
label-based scoring is strictly **post-hoc**, kept separate from any supervised fine-tuning
(the protocol the wafer-SSL literature follows [9][26]). Only the **172,950 labeled**
WM-811K maps enter scoring; the unlabeled majority is train-only. Two cross-cutting
principles, grounded in the retrieval/IR literature [28][29]:

- **Separate three questions that are routinely conflated** [29]: (1) *representation
  quality* — are relevant wafers near each other under the embedding? (2) *index quality* —
  does the Lakebase ANN index reproduce exact search? (3) *serving performance* — latency /
  throughput. Score representation with **exact** search first (FAISS inner-product on
  L2-normalized vectors [19]); measure ANN degradation and latency separately (§8.6). An
  index can perfectly reproduce exact search while the embedding is semantically poor, and
  vice-versa.
- **Report per-pattern, not just aggregate.** WM-811K is **severely imbalanced** (`none`
  and `Edge-Ring` dominate; `Donut`, `Near-full`, `Random` are rare) and rare patterns
  matter [16]; always report **macro + per-class** beside micro so common patterns can't
  mask rare-pattern failures.

### 8.1 Retrieval (primary goal)
Ranked-retrieval metrics over held-out queries [28]: **mAP@k, Recall@k, nDCG@k, MRR,
precision@k** (k ∈ {1,5,10,50}). State conventions explicitly (AP@K denominator; whether
"Recall@K" is fraction-of-relevant or hit-rate — report both, labeled), since scores under
different conventions aren't comparable [28]. **Relevance definition:** for WM-811K
(single-label) a neighbor is relevant iff it shares the query's class — the simple,
unambiguous case. For the MixedWM38 reference path (multi-label), use **Jaccard overlap ≥
τ** on the defect set as primary, also report strict **exact label-set match**, and feed
graded relevance (exact / shares-some / none) to nDCG.

### 8.2 Clustering / pattern discovery (secondary goal)
Following the wafer-SSL clustering convention [9][26][27]:
- **External (label-based):** **ARI** and **NMI** as primary (one chance-adjusted, one not
  — state the NMI normalization), plus **purity** as supplementary, *always reported with
  the cluster count* since over-/singleton-clustering inflates purity [28].
- **Internal (label-free):** **silhouette** (and Davies–Bouldin) so clustering is judgeable
  on unseen patterns without labels [27].
- **Algorithms:** **HDBSCAN** (density-based, handles noise + variable cluster count) and
  k-means; report sensitivity to k / min-cluster-size.

### 8.3 Representation probes (SSL protocol)
The standard DINO frozen-feature protocol [1]: **weighted kNN** (cosine, exclude
self-matches; DINO defaults k ∈ {10,20,100,200}, τ=0.07) and a **frozen linear probe**.
For WM-811K (single-label): **top-1 / balanced accuracy, macro-F1, per-class recall**
(balanced metrics matter given the imbalance); the MixedWM38 path reports multi-label
micro/macro-F1 + mAP. Fit probe / kNN bank on train, tune on val, evaluate test once;
freeze encoder +
norm stats; **report ≥3 probe seeds (mean ± std)**; pin which feature vector is used (CLS
vs. last-4-CLS concat vs. backbone output — these differ in DINO) [1].

### 8.4 Embedding health & invariance
- **Collapse diagnostics** (DINO can collapse): centered-covariance **eigenspectrum**,
  **effective rank**, dominant-direction fraction, total variance — on both raw and
  L2-normalized features [25]. Flag dimensional collapse early in training.
- **Alignment & uniformity** [24] on unit-normalized embeddings: alignment = view
  invariance (lower is tighter), uniformity = hypersphere spread (lower is more spread);
  interpret *jointly* (a constant embedding has perfect alignment but zero uniformity) and
  treat as diagnostics, not a quality ranking.
- **Invariance check:** cosine similarity between a wafer and its augmented view
  (rotation/flip, §5) should be high vs. low to a different pattern — confirms we learned
  the intended invariances and no unwanted ones.

### 8.5 Size generalization & binning
- **Size:** hold out whole grid-size groups (and crops) unseen in training; verify the same
  pattern at different grid sizes maps to nearby embeddings (native variable-res path, §5).
- **Binning (validated later):** WM-811K exercises only the pass/fail case. With real
  multi-scheme bin maps, score retrieval/clustering *within* a single scheme and on the
  *fused* embedding, and verify robustness to a missing scheme (modality dropout, §5).

### 8.6 System & serving metrics (separate from representation)
- **ANN recall@k vs. exact** [29] over identical embeddings / metric / filters — gate so
  the index doesn't silently degrade retrieval.
- **Serving:** **QPS at a fixed ANN-recall target**, with p50/p95/**p99** latency (N2),
  stating what the timing includes (encode vs. index search); index build time; **encode
  latency per wafer by grid size** (§4 FLOP table); batch-embedding throughput. Recall–QPS
  curves, not QPS alone [29].
- **Footprint:** index size at 384-d across the ~700K → synthetic 1M corpus (N3).

### 8.7 Protocol
- **Splits (leakage control):** fixed train/val/test (80/10/10) by salted hash of the
  WM-811K **`lotName`**, so a whole lot lands in one split (maps with no lot fall back to a
  per-map content hash). Wafers of one lot share device and often defect: with the earlier
  per-map split, two maps of the same shape shared a label 75% of the time vs 22% overall,
  and a one-hot of map shape alone matched the untrained encoder on kNN / mAP@10. The
  lot split is enforced at ingest (fails if any lot straddles splits): 45,345 lots
  (train 36,120 / val 4,635 / test 4,590). Device-level correlation remains across lots
  (same shape → same label 0.33 vs 0.11 across query-bank pairs), so G1 also reports
  **cross-device** `xgroup_*` metrics (same-shape neighbours excluded; map shape is the
  device proxy). Exclude test wafers from DINO pretraining for an **inductive** read
  (also report transductive). WM-811K contains genuine duplicate maps, so content-hash
  **dedup** (696,599 unique of 811,457) runs before splitting.
- **Label protocol:** labels are **evaluation-only**; no label enters training (sampling,
  loss, weighting, or checkpointing; keep-best selection is off by default). All
  development decisions (ablations, hyperparameters, recipe choices) are judged on
  **val**-split queries against a train-split labeled kNN bank (`--eval-split val`, the
  default). The **test** split is scored **once**, for the final report
  (`--eval-split test`), and never used to tune. (Results before 2026-10-09 pooled
  val+test queries and predate the lot split, so they are superseded.)
- **Training-free bar:** every run also scores a handcrafted rotation-invariant polar
  FAIL-density histogram (`eval/baselines.py`, `polar_*` metrics). A learned encoder is
  only useful if it beats this, especially on the cross-device metrics.
- **Baselines:** random; flattened-pixels / PCA; **frozen off-the-shelf DINOv2** [21];
  ImageNet-pretrained CNN features (cf. [27]); a **supervised ViT** upper bound; and the
  external **graph-contrastive** reference — **ARI 0.89 / NMI 0.87 / silhouette 0.76** on
  WM-811K [9], now a **directly comparable** target since WM-811K is our primary corpus;
  WaPIRL [2] / SWaCo [10] where reproducible.
- **Ablations:** patch 1 vs. 2; **ISAB vs. full attention**; positional-encoding scheme;
  **DINO vs. +same-class-positives (SWaCo [10])** for the retrieval false-negative risk
  (§4); embedding dim; model-size sweep (the "smallest that passes" curve, N4);
  augmentation set.
- **Rigor:** multiple seeds, confidence intervals on headline claims.

### 8.8 Qualitative
Engineer-reviewed **query → top-k galleries** (especially mixed defects); **UMAP / t-SNE**
of the embedding space colored by label; nearest-neighbor montages for pattern discovery.

### 8.9 Acceptance gates
- **Quality gate (ship / stop shrinking, N4):** retrieval **mAP@10 ≥ 0.80** and clustering
  **NMI ≥ 0.85** — *recommended starting targets* set against the 0.87 external reference
  [9]; **confirm at project start.**
- **Index gate:** ANN **recall@10 ≥ 0.95** vs. exact, within the p99 budget (N2) [29].
- **Health gate:** no dimensional collapse (effective rank well above 1) at the shipped
  checkpoint [25].

---

## 9. Lakebase Search integration

Use **Lakebase Search** (the `lakebase_vector` extension + `lakebase_ann` index) —
**not** raw pgvector. `CREATE EXTENSION lakebase_vector CASCADE` brings the `vector` *type*
as its own dependency; pgvector is never installed or indexed directly, and all similarity
search goes through `lakebase_ann`.

**Automation (verified 2026-10-09, `jobs/lakebase.py`, bootstrap job task `lakebase`):**
fully automatic and idempotent — project (Lakebase Autoscaling, PG 17) → enable Lakebase
Search via `POST /api/2.0/postgres/projects/{project}/search-extensions` (body `{}`, a
long-running operation; the call behind the UI button, found with Chrome DevTools;
re-posting returns `done`) → database → extension → UC synced table (Delta →
`vector(dim)`, snapshot; `--refresh` re-snapshots after re-embedding) → cosine
`lakebase_ann` index → app service-principal grants → verify the top-k plan is an ANN
index scan. Proven on the live project and on a fresh throwaway project from nothing.
Top-k query latency through the app's SQL: ~40–120 ms at 696,599 rows (vs ~450 ms exact).

- Lakebase project `wafer-embeddings` (Postgres 17, branch `production`); database
  `wafer_embeddings`.
- Table `wafer_embeddings.wafer_map_embeddings_pg(id bigint pk, embedding vector(D),
  map_code, label, label_id, split, lot, height, width, n_dies, fail_frac, model_version)`,
  populated by the UC synced table from Delta `wafer_map_embeddings`. **D = 384** per N1;
  **D = 128 today** (interim model) — `vector(128)`, 696,599 rows.
- Index (cosine `lakebase_ann`; on the partitioned synced table it appears per partition):
  ```sql
  CREATE EXTENSION IF NOT EXISTS lakebase_vector CASCADE;
  CREATE INDEX IF NOT EXISTS wafer_map_embeddings_ann
    ON wafer_embeddings.wafer_map_embeddings_pg
    USING lakebase_ann (embedding vector_cosine_ops) WITH (build_mode = 'standard');
  ```
- Query (bind `$1` = the D-dim query embedding):
  ```sql
  SELECT id, label, 1 - (embedding <=> $1::vector) AS cosine_similarity
  FROM wafer_embeddings.wafer_map_embeddings_pg
  ORDER BY embedding <=> $1::vector
  LIMIT :k;
  ```

---

## 10. Serving

- **Batch** — scheduled job (§7.5) backfills/refreshes the corpus and re-syncs Lakebase.
- **Real-time** — Model Serving endpoint `wafer-encoder` wraps tokenize + encoder; input
  a native wafer grid, output a D-dim L2-normalized vector (384 per N1; **128** from the
  interim `wafer_encoder` v2, CPU Small, scale-to-zero). New/online wafers are embedded
  then upserted into the Lakebase table (and picked up by the ANN index); today the search
  app uses the endpoint to re-embed a query live.

---

## 11. Databricks implementation stack

Workspace: existing FEVM **MMF MLOps-demo** workspace —
`fevm-mmf-mlops-demo.cloud.databricks.com` (AWS). UC **catalog `mmf_mlops_demo_catalog`**,
**schema `wafer_embeddings`**. Tooling: **Python** (≥3.11,<3.13), **uv** (env + wheel
build), **Databricks Asset Bundles (DAB)**, Databricks CLI. *(Version pins below reflect 2025 Databricks release notes — verify current values at
build time; the exact accelerator menu and environment versions drift.)*

### 11.1 Green-start scaffold
No single template covers this stack (ViT/DINO + uv + DAB + Lakebase vector search) —
confirmed against Databricks docs and the FE knowledge catalog. Start from a known-good
base and layer the ML pieces:

- **Base:** `databricks bundle init default-python` — Databricks-maintained DAB template
  that natively uses **uv + `pyproject.toml`** and deploys on serverless. Minimal, green,
  matches the python/uv/dab stack.
- **MLOps Stacks** (`mlops-stacks`) is the fuller lifecycle template (training / validation
  / deployment / batch_inference / monitoring / resources + CI/CD) but is heavier and its
  uv + serverless-GPU support is unconfirmed — **borrow its directory taxonomy, don't start
  from it.**

### 11.2 Project layout (DAB + uv)
```
wafer-embeddings/
├── databricks.yml            # bundle entry: targets (dev/prod), resources, artifacts
├── pyproject.toml            # uv project; deps: torch, scikit-learn, numpy (custom ViT/DINO, no timm)
│                             # extras: jobs (mlflow, databricks-sdk, pandas, pyarrow), search (faiss-cpu)
├── src/wafer_embeddings/
│   ├── data/                 # WM-811K (+ MixedWM38) parse, dedup, splits (§2,§5)
│   ├── tokenize/             # native per-die tokenizer + token cap + augmentations (§5)
│   ├── model/                # per-die ViT (patch=1, ISAB) + DINO head/loss/EMA (§4,§6)
│   ├── train/                # DINO training loop + schedules + Gate-G1 eval
│   ├── eval/                 # exact-search retrieval / clustering / health metrics (§8)
│   └── jobs/                 # bootstrap, ingest, train entry points
├── ai_runtime/train.yaml     # AI Runtime serverless-GPU training workload (§11.3)
├── resources/                # *.yml job (+ later endpoint + Lakebase) resource defs
└── tests/
```
`uv.lock` is **gitignored** (it pins through a corporate PyPI proxy and isn't portable);
CI and clean installs run `uv sync` against public PyPI, with runtime deps pinned in
`[project].dependencies`. The DAB bundle jobs reference a wheel built via artifact
`build: uv build --wheel`; the AI Runtime trainer instead ships `./src` via
`code_source.snapshot` (§11.3). Use `dynamic_version: true` for iterative dev deploys.

### 11.3 Training compute — Databricks AI Runtime (serverless GPU) ✅ proven
- **AI Runtime = serverless GPU.** ViT-S/DINO is small ⇒ default **single A10**
  (`GPU_1xA10`); single-node **H100** only if scaling up / FSDP. The training workload is
  declared in `ai_runtime/train.yaml` (`experiment_name`, `compute`, `environment`,
  `code_source`, `command`) and submitted with the **`databricks air`** CLI.
- **✅ Verified end-to-end** (2026-10-08): a run on `GPU_1xA10` loaded **torch 2.14.1+cu130**
  on CUDA, read the parquet export from the volume, pretrained the per-die ViT with DINO,
  embedded the labeled subset, computed Gate-G1 metrics, and logged to MLflow.
- **No Spark session on AI Runtime.** The trainer therefore reads the **parquet export**
  from the UC volume via pyarrow (bounded with `.head(...)`), not `spark.read.table` — the
  key architectural consequence of the serverless-GPU environment.
- **Code shipping:** local `./src` is uploaded the documented way via
  `code_source.snapshot` and exposed to the command through `$CODE_SOURCE_PATH` (no
  wheel-on-volume install). Dependencies (torch, scikit-learn, mlflow, pandas, pyarrow)
  are declared in the workload `environment`; the base env does **not** preinstall torch.
- **Custom model, no `timm`.** The per-die ViT (patch=1) and DINO are implemented in-repo
  (`model/`), so no `timm`/`lightly` dependency — see §4 and the HF-equivalent note.

### 11.4 MLflow + Unity Catalog registry
- `mlflow.set_registry_uri("databricks-uc")`; 3-part name
  `mmf_mlops_demo_catalog.wafer_embeddings.wafer_encoder` (v2 `@champion` = interim 128-d).
- Package as **custom `mlflow.pyfunc.PythonModel`**: `load_context` loads the torch encoder;
  `predict` runs the §5 tokenizer + encoder → D-dim L2-normalized vector (N1); provide
  `infer_signature`. MLflow 3 uses `log_model(name=...)`. UC perms: `USE CATALOG`,
  `USE SCHEMA`, `CREATE MODEL`.

### 11.5 Serving
- **Real-time:** Model Serving endpoint on the pyfunc model. Tiers: `GPU_SMALL` (T4),
  `GPU_MEDIUM` (A10G), or **CPU** — per §4, **CPU serving is viable for ~2k-token wafers**;
  use GPU for dense 100×100 / high throughput. **GPU-serving gotcha:** log/register the
  model **from a GPU runtime**, else it is packaged with CPU deps and the GPU endpoint fails
  to start (fail-fast on `DATABRICKS_ACCELERATOR`). Express deploy needs `mlflow≥3.12`,
  `databricks-sdk≥0.102.0`, and GPU-aware `predict` code (device handling is not automatic
  for custom pyfunc).
- **Batch:** serverless job computes embeddings → Delta → sync to Lakebase (§9).

### 11.6 Lakebase wiring
Register the instance in UC (`w.database.register_database_instance(...)`) and declare
`DatabricksLakebase(database_instance_name="wafer-embeddings")` in the model's MLflow
`resources` for credential provisioning; build the `lakebase_ann` index per §9.

### 11.7 Config-driven
MLflow experiments + UC registry; bundle variables drive dataset path, embedding dim,
patch/token settings, model size, eval thresholds, and the **binning schemes** (per-scheme
names, vocabulary sizes, and canonical-taxonomy mapping — §5/§15) so the template retargets
to other wafer datasets and binning setups.

### 11.8 Reproducibility — from scratch, programmatically (N7)
Nothing is created by hand. The bundle plus a **bootstrap job** (Databricks SDK/CLI)
create and wire everything, **idempotently** (safe to re-run):
1. **Create the catalog if missing** (`SHOW CATALOGS LIKE` first, then
   `CREATE CATALOG IF NOT EXISTS` only when absent — Unity Catalog checks the metastore
   `CREATE CATALOG` privilege before existence, so the bare statement fails for users
   without it even when the catalog exists); `USE CATALOG mmf_mlops_demo_catalog`;
   **create schema + volume** if absent. A single catalog holds everything, including the
   Lakebase synced table.
2. **Ingest WM-811K** from `LSWMD.pkl` in the volume: stream → clean → dedup → split →
   Delta, plus a parquet export for the AI Runtime trainer (§7, §11.3).
3. **Provision Lakebase** (bootstrap job task `lakebase`, `jobs/lakebase.py`): project,
   Lakebase Search (REST `search-extensions`), database, `lakebase_vector`, synced table +
   `lakebase_ann` index + app grants (the embed job re-runs it with `--refresh`).
4. **Train → register → serve** (train on AI Runtime, register to UC, create the endpoint).
Expose one-shot **`bundle deploy` + bootstrap** and a **teardown**: `jobs/teardown.py`
(code-managed resources, `--scope serving|all`, dry run by default) + `bundle destroy`. **Pin
everything** — a modern serverless `environment_version` and all deps — because the
default serverless env is minimal and old (§16 item 17): no mlflow/torch, Python 3.10.12,
`databricks-sdk` 0.20.0. No notebook-only / click-ops steps may be load-bearing.

---

## 12. Deliverables & milestones

1. **M1 — Data & tokenize:** ingest WM-811K, native-res per-die tokenizer, Delta + parquet.
2. **M2 — Train encoder:** DINO self-distillation ViT (patch=1) on AI Runtime GPU + eval
   report vs. threshold; size ablation for the smallest model that passes.
3. **M3 — Lakebase Search:** synced vector table, `lakebase_ann` index, query API, p99 check.
4. **M4 — Serving:** real-time endpoint + online upsert path.
5. **M5 — DAB packaging** of the full template.

---

## 13. Related work

Grouped by how it bears on the design. Reported metrics are as claimed by the authors
(not independently verified here); full citations in §14.

**ViT on MixedWM38 (supervised, coarse patches).** The dataset is well-established for
transformers, but every entry here is *supervised classification* with multi-die patches —
none use per-die tokens or label-free training. MLR-WM-ViT [5] reports 99.15% on
MixedWM38; a compact deformable-convolution transformer [6] targets mixed defects; an
open ViT notebook [7] uses 4×4 patches (98.98%, unverified protocol); a wafer ViT-Tiny
study [8] found patch 16 best in its setup. ⇒ ViT is proven here; the per-die, label-free
angle is open.

**Self-supervised representation learning on wafer maps (closest cluster).**
Graph-contrastive learning [9] is the most on-point: dies as graph nodes (8-connectivity),
GIN encoder, contrastive SSL, evaluated by *clustering* — ARI 0.89 / NMI 0.87 on WM-811K
(our clustering baseline to beat, §8). WaPIRL [2] is CNN noise-contrastive SSL; SWaCo [10]
adds same-class positives (label-assisted) — the basis for our false-negative mitigation
(§4); MAE-based SSL for complex/mixed wafer bin maps [11] and few-shot MAE [12] show
masked-reconstruction pretraining works on wafer maps; Podder et al. [3] find
domain-specific representations beat frozen DINOv2 for wafer clustering (why we train, not
freeze); BEiT+DBSCAN [4] corroborates transformer embeddings + clustering.

**Per-die / set / graph representations (our patch=1 approach).** No wafer paper uses a
literal patch=1 ViT, but the equivalent is well-grounded: a per-die transformer *is* a Set
Transformer [13] over dies — whose ISAB induced attention gives the O(N·m) cost we adopt
(§4, §14). Deep Sets [14] is the permutation-invariant baseline over (coord, state) die
tuples; DGCNN/EdgeConv [15] models defective dies as a point cloud. These consistently
recommend normalizing die coordinates by wafer center/radius (§5).

**Variable-size handling.** Common practice resizes to a fixed grid (e.g. 64×64) with
nearest-neighbor even for categorical maps, but scale-invariance is not guaranteed and
thin/small defects can vanish on downsampling [16]; spatial-pyramid pooling [17] and
set/coordinate models are the no-resize alternatives — supporting our native, no-resize
choice.

**Retrieval & indexing.** Wafer SSL work almost exclusively evaluates
classification/clustering, so our retrieval objective needs its own precision@k / mAP eval
(§8). The DINO [1] / SimCLR [18] instance-contrastive objective risks same-pattern false
negatives (mitigation §4). FAISS inner-product on normalized vectors [19] is the cosine
baseline before Lakebase Search.

---

## 14. References

1. Caron, Touvron, Misra, Jégou, Mairal, Bojanowski, Joulin. *Emerging Properties in
   Self-Supervised Vision Transformers* (DINO). ICCV 2021. arXiv:2104.14294.
2. Kahng, Kim. *Self-Supervised Representation Learning for Wafer Bin Map Defect Pattern
   Classification* (WaPIRL). IEEE Trans. Semiconductor Mfg. 34(1):74–86, 2021.
   DOI 10.1109/TSM.2020.3038165. Code: `hgkahng/WaPIRL`.
3. Podder, Miller, Fischl, Bub. *Unsupervised Representation Learning and Explainable
   Clustering for Wafer Map Pattern Analysis.* IEEE Trans. Semiconductor Mfg.
   38(3):693–708, 2025. DOI 10.1109/TSM.2025.3579031.
4. *Using BERT Pre-Trained Image Transformers to Identify Potential Parametric Wafer Map
   Defects.* SEMI Advanced Semiconductor Mfg. Conf. (ASMC) 2024.
   DOI 10.1109/ASMC61125.2024.10545454.
5. *MLR-WM-ViT: Global high-performance classification of mixed-type wafer map defect
   using a multi-level relay Vision Transformer.* Expert Systems with Applications, 2025.
   ScienceDirect S0957417425007432.
6. *Efficient Mixed-Type Wafer Defect Pattern Recognition Using Compact Deformable
   Convolutional Transformers.* 2023. arXiv:2303.13827.
7. PanithanS. *Wafers-Defect-Recognition-using-Visual-Transformer* (MixedWM38, 4×4
   patches). GitHub: `PanithanS/Wafers-Defect-Recognition-using-Visual-Transformer`.
8. Wafer ViT-Tiny patch-size study. 2025. arXiv:2504.02494.
9. Awais, Postolache, Oliveira. *Graph-based contrastive learning for self-supervised
   semiconductor wafer defect detection.* J. Intelligent Manufacturing, 2026.
   DOI 10.1007/s10845-026-02906-3. Code: `irjawais/wafer-gnn-contrastive`.
10. Kwak, Lee, Kim. *SWaCo: Safe Wafer Bin Map Classification With Self-Supervised
    Contrastive Learning.* IEEE Trans. Semiconductor Mfg. 36(3):416–424, 2023.
    DOI 10.1109/TSM.2023.3280891.
11. Wang et al. *A self-supervised learning framework based on masked autoencoder for
    complex wafer bin map classification* (patchMC). Expert Systems with Applications,
    2024. DOI 10.1016/j.eswa.2024.123601.
12. Liang, Zhou, Wang. *Masked autoencoder with dynamic multi-loss adaptation mechanism
    for few-shot wafer map pattern recognition.* Engineering Applications of Artificial
    Intelligence 137(A):109070, 2024. DOI 10.1016/j.engappai.2024.109070.
13. Lee, Lee, Kim, Kosiorek, Choi, Teh. *Set Transformer: A Framework for Attention-based
    Permutation-Invariant Neural Networks* (ISAB, PMA). ICML 2019.
14. Zaheer, Kottur, Ravanbakhsh, Póczos, Salakhutdinov, Smola. *Deep Sets.* NeurIPS 2017.
15. Wang, Sun, Liu, Sarma, Bronstein, Solomon. *Dynamic Graph CNN for Learning on Point
    Clouds* (EdgeConv/DGCNN). 2018. arXiv:1801.07829.
16. *Interpolation difficulties at low/inconsistent wafer-map resolution.* Expert Systems
    with Applications, 2024. ScienceDirect S095741742403001X.
17. He, Zhang, Ren, Sun. *Spatial Pyramid Pooling in Deep Convolutional Networks* (SPP).
    2014. arXiv:1406.4729.
18. Chen, Kornblith, Norouzi, Hinton. *A Simple Framework for Contrastive Learning of
    Visual Representations* (SimCLR). ICML 2020. arXiv:2002.05709.
19. FAISS — *MetricType and distances* (`IndexFlatIP` = cosine on normalized vectors).
    GitHub: `facebookresearch/faiss`.
20. Dosovitskiy et al. *An Image is Worth 16×16 Words: Transformers for Image Recognition
    at Scale* (ViT). ICLR 2021. arXiv:2010.11929.
21. Oquab et al. *DINOv2: Learning Robust Visual Features without Supervision.* 2023.
    arXiv:2304.07193. (Methodology reference; weights not reused.)
22. WM-811K dataset (MIR-WM811K) — Wu, Jang, Chen. *Wafer Map Failure Pattern Recognition
    and Similarity Ranking for Large-Scale Data Sets.* IEEE Trans. Semiconductor Mfg.
    28(1):1–12, 2015. Download: http://mirlab.org/dataSet/public/MIR-WM811K.zip ·
    MixedWM38 (secondary) — https://github.com/Junliangwangdhu/WaferMap
23. pgvector / Lakebase Search dimension & index notes —
    https://github.com/pgvector/pgvector
24. Wang, Isola. *Understanding Contrastive Representation Learning through Alignment and
    Uniformity on the Hypersphere.* ICML 2020. arXiv:2005.10242.
25. Jing, Vincent, LeCun, Tian. *Understanding Dimensional Collapse in Contrastive
    Self-Supervised Learning.* ICLR 2022. arXiv:2110.09348.
26. Kim, Kang. *Dynamic Clustering for Wafer Map Patterns Using Self-Supervised Learning on
    Convolutional Autoencoders.* IEEE Trans. Semiconductor Mfg. 34(4):444–454, 2021.
    DOI 10.1109/TSM.2021.3107720.
27. Pleli et al. *Iterative Cluster Harvesting for Wafer Map Defect Patterns.* 2024.
    arXiv:2404.15436.
28. Manning, Raghavan, Schütze. *Introduction to Information Retrieval.* Cambridge Univ.
    Press, 2008 (mAP, nDCG, purity, cluster evaluation).
29. Aumüller, Bernhardsson, Faithfull. *ANN-Benchmarks: A Benchmarking Tool for Approximate
    Nearest Neighbor Algorithms* (recall vs. QPS methodology). Information Systems, 2020.
30. Jeong, Lee, Park, Kim, Huh, Lee. *Wafer map failure pattern classification using
    geometric transformation-invariant convolutional neural network.* Scientific Reports
    13:8127, 2023. DOI 10.1038/s41598-023-34147-2. (Rotation/flip label-preservation;
    identifies Edge-Loc/Loc/Scratch.)
31. Yu, Chen, Xu, Hasan, Sie. *Wafer map defect patterns classification based on a
    lightweight network and data augmentation.* CAAI Trans. Intelligence Technology
    8(3):1029–1042, 2023. DOI 10.1049/cit2.12126. (Uses rotation but avoids cropping —
    Loc vs. Edge-Loc depends on boundary position.)
32. Hu, He, Li. *Semi-supervised Wafer Map Pattern Recognition using Domain-Specific Data
    Augmentation and Contrastive Learning* (rotation-twist = radius-dependent rotation;
    contrastive SSL). IEEE Int. Test Conf. (ITC) 2021, pp. 113–122.
33. Park, You. *Deep Convolutional Generative Adversarial Networks-Based Data Augmentation
    Method for Classifying Class-Imbalanced Defect Patterns in Wafer Bin Map.* Applied
    Sciences 13(9):5507, 2023. DOI 10.3390/app13095507.

---

## 15. Open questions / risks

- **Long-sequence attention cost:** WM-811K's largest maps have tens of thousands of dies
  (§4). The **token cap** (default ~4k) bounds the sequence, and **ISAB** induced attention
  (O(N·m)) is the default; combined with dropping `no-die` tokens, gradient checkpointing,
  and memory-efficient SDPA. Open: tune the cap vs. accuracy on the giant-map tail.
- **DINO from scratch** is no longer data-limited — WM-811K provides ~638K **unlabeled**
  maps, exactly the regime DINO wants. Open: confirm the recipe (weight-normed prototype
  head + schedules) escapes uniform collapse at scale and clears G1 (in progress).
- **Positional-encoding generalization:** confirm the center/radius sinusoidal scheme
  transfers to grid sizes not seen in training, per the §8 size-generalization eval.
- **Single-label eval semantics (WM-811K):** relevance = same class (§8.1); the Jaccard /
  exact-set multi-label semantics apply only to the MixedWM38 reference path.
- **Lakebase `lakebase_ann` dimension limits** at 384-d: confirm against current docs.
  (Works at 128-d today: 696,599 vectors, index built, top-k ~40–120 ms.)
- **Canonical bin taxonomy (owned by domain experts):** cross-product/cross-fab retrieval
  needs a shared bin ontology (pass/fail + failure-category) so per-scheme embeddings are
  comparable across datasets. Until it exists, cross-dataset retrieval is only meaningful
  within a shared scheme. Defining this mapping is an open, non-modeling task (§5, §11.7).
- **Continuous parametrics (deferred):** feeding per-die measurements (Idd/Vmin/Fmax)
  alongside categorical bins is a natural extension once parametric test data is available.
- **Orientation-aware targets (future):** rotation/flip invariance is correct for the
  standard morphology taxonomy (§5), grounded in [2][30]. If a dataset encodes scratch
  direction, edge sector, or orientation-dependent root cause, set
  `orientation_invariant=false` and re-ablate (§8.7) — those are different targets from the
  WM-811K / MixedWM38 morphology classes.

---

## 16. Technical de-risking checklist (validate empirically)

Concrete questions to settle with experiments, ordered roughly by how early each could
**kill or reshape** the approach. Tags: **[W8]** testable now on WM-811K · **[PLAT]**
Databricks platform · **[DATA]** needs data beyond WM-811K (multi-scheme bins) · **[NM]**
non-modeling/domain.

### A. Does the model learn a useful representation?
1. **[W8] DINO-from-scratch on WM-811K isn't data-starved.** Check: train the per-die ViT
   with DINO on the ~638K unlabeled maps; eval kNN/linear probe + clustering vs. baselines
   (§8.7). Pass: beats frozen DINOv2 and approaches the GNN-contrastive reference on WM-811K
   (ARI 0.89 / NMI 0.87 [9]). If it fails: stronger augmentation, same-class positives.
2. **[W8] No dimensional collapse.** Check: effective rank / eigenspectrum + alignment &
   uniformity during training (§8.4). Pass: effective rank ≫ 1, stable uniformity.
3. **[W8] DINO false negatives don't sink same-pattern retrieval.** Check: pure DINO vs.
   SWaCo same-class-positive variant on retrieval mAP (§8.7). Pass: pure-DINO mAP@10 ≥ gate;
   else adopt same-class positives.

### B. Is the per-die architecture actually feasible?
4. **[W8] Per-die attention cost at scale.** Check: wall-clock + peak memory for full
   attention vs. ISAB at median (~1–2k) and capped (~4k) tokens on an A10; encode latency by
   size (§4). Pass: within train budget and serving p99 (N2). ✅ Partial: ISAB + token cap
   + grad checkpointing fit depth-8/12 on an A10 (full attention OOM'd on the giant tail).
5. **[W8] ISAB keeps quality.** Check: ablate ISAB vs. full attention at ~2k tokens (§8.7).
   Pass: negligible retrieval/clustering drop.
6. **[W8] patch=1 vs. 2.** Check: ablation on eval + latency. Pass: patch=1 earns its cost,
   else fall back to 2×2.

### C. Does variable-resolution / invariance hold?
7. **[W8] Size generalization.** Check: hold out whole grid-size groups + synthetic resized
   wafers; same pattern at different sizes → nearby embeddings (§8.5). Pass: retrieval /
   clustering holds on unseen sizes; cosine stable across size.
8. **[W8] Positional encoding generalizes** to grid sizes unseen in training — confirm the
   center/radius sinusoidal scheme specifically (folds into 7).
9. **[W8] Learned the intended invariances.** Check: cosine(wafer, rotated/flipped view) high
   vs. different pattern low; verify crops don't confuse Loc ↔ Edge-Loc (boundary caveat,
   §5). Pass: invariance holds for the chosen mode; no boundary-class confusion.
10. **[W8] Masking / ragged-wafer correctness.** Check: embedding invariant to padding;
    dropping no-die tokens doesn't change results. Pass: padding-invariance unit tests green.

### D. Does the Lakebase retrieval system work?
11. **[PLAT] Lakebase Search supports 384-d cosine.** Check: create instance,
    `lakebase_vector` extension, `lakebase_ann` index on `vector(384)` (§9) — confirm no
    dimension-limit issue. Pass: index builds, cosine query returns.
    ✅ **Verified at 128-d (2026-10-09)** with the interim model: extension + cosine
    `lakebase_ann` index on `vector(128)`, top-k served via the index. Re-check at 384-d.
12. **[PLAT] UC synced-table → Postgres `vector(384)` mapping** works end-to-end. Pass:
    Delta embeddings sync into the Lakebase table + index.
    ✅ **Verified at 128-d**: `array<float>` → `vector(128)` via a synced-table type override
    (`PG_SPECIFIC_TYPE_VECTOR`, size = D); 696,599 rows. Re-check at 384-d.
13. **[PLAT] ANN recall vs. exact.** Check: recall@10 vs. FAISS exact (§8.6). Pass: ≥ 0.95.
14. **[PLAT] Query latency.** Check: p50/p99 at ~700K and synthetic 1M (N2/N3). Pass: p99 < 50 ms.

### E. Does training + serving run on this workspace?
15. **[PLAT] Catalog/schema/volume exist + perms.** Check: authenticate to
    `fevm-mmf-mlops-demo`; `USE CATALOG/SCHEMA`, `CREATE MODEL`, volume read/write on
    `mmf_mlops_demo_catalog.wafer_embeddings`. Pass: all granted.
    ✅ **Verified 2026-10-07:** workspace reachable; catalog + schema `wafer_embeddings`
    exist; user is a **workspace admin**. Volume/tables not yet created. For N7 the bundle
    must declare `USE CATALOG` + create the schema/volume idempotently (schema exists
    out-of-band today).
16. **[PLAT] AI Runtime serverless-GPU train path.** Check: a reproducible train →
    register-to-UC path on serverless GPU (§11.3). Pass: training runs on AI Runtime.
    ✅ **Verified 2026-10-08:** `databricks air submit ai_runtime/train.yaml` runs on
    `GPU_1xA10` — torch 2.14.1+cu130 on CUDA, parquet read from the volume (no Spark), DINO
    train → embed → G1 → MLflow. Code shipped via `code_source.snapshot` + `$CODE_SOURCE_PATH`
    (not a wheel-on-volume). UC model registration + serving still to wire (P3).
17. **[PLAT] uv + DAB green loop.** Check: `bundle init default-python`, `uv build --wheel`,
    deploy a trivial serverless job; torch/CUDA resolve in the AI Runtime env (§11.3).
    Pass: job runs, deps resolve.
    ✅ **Resolved 2026-10-08:** the **default** serverless env is minimal/old (Python
    3.10.12, `databricks-sdk` 0.20.0, no mlflow/torch), so the AI Runtime workload declares
    a modern `environment` with torch/scikit-learn/mlflow/pandas/pyarrow pinned; CI runs
    `uv sync` + black + ty + pytest green. No `timm` needed (custom ViT/DINO, §11.3).
18. **[PLAT] GPU-serving packaging gotcha.** Check: log/register the pyfunc **from a GPU
    runtime**, deploy `GPU_SMALL`, confirm the endpoint starts (no `DATABRICKS_ACCELERATOR`
    fail-fast); `mlflow≥3.12`, `databricks-sdk≥0.102.0` present (§11.5). Pass: GPU endpoint
    healthy.
19. **[PLAT] CPU serving viability** for ~2k-token wafers. Check: CPU encode latency vs.
    budget. Pass: CPU acceptable (else GPU tier).
20. **[PLAT] Custom pyfunc loads + predicts** (tokenizer + torch encoder, `code_paths` / deps)
    on the endpoint. Pass: endpoint returns a 384-d normalized vector.
    ✅ **Verified with the interim 128-d model**: the CPU endpoint returns a 128-d
    L2-normalized vector matching the local encoder to 1e-7.

### F. Data & forward-design items
21. **[W8] WM-811K ingest / parse** — stream `LSWMD.pkl` (`waferMap` variable-size {0,1,2},
    `failureType` → 9 classes, `trianTestLabel`); normalize messy labels, content-hash
    **dedup** real duplicates before splitting (§8.7). Pass: shapes/labels verified, no split
    leakage. ✅ Done: 696,599 unique of 811,457 streamed to Delta + parquet.
22. **[DATA] Multi-scheme binning fusion** — cannot be validated on WM-811K (pass/fail
    only). When real multi-bin maps exist: within-scheme vs. fused retrieval, missing-scheme
    robustness (modality dropout, §5/§8.5).
23. **[NM] Canonical bin taxonomy** for cross-product comparability — a domain-expert
    ontology task, not modeling (§5/§15).
24. **[W8] Acceptance thresholds are realistic.** Check: calibrate the mAP@10 / NMI gates
    against baselines early (§8.9); adjust the 0.80 / 0.85 starting targets once baselines
    land.
