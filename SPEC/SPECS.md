# Spec: Wafer-Map Embedding Model Template (MixedWM38)

**Status:** Draft · **Date:** 2026-10-07 · **Owner:** lbruand-db

A reusable Databricks template that trains a **small, per-die vision-transformer
embedding model** for semiconductor wafer maps, backfills a vector corpus into
**Lakebase Search**, and serves embeddings both in **batch** and **real time**.
Reference dataset: [MixedWM38](https://github.com/Junliangwangdhu/WaferMap).

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

**MixedWM38**: 38,015 wafer maps, each a fixed **52×52** grid, **38 classes**
(1 normal + 8 single-defect + 29 mixed-defect). It is inherently **multi-label** — a
mixed map is a co-occurrence of single-defect patterns (e.g. `Center+Scratch`). Each
cell is **3-valued**: no-die (outside wafer), passing die, failing die.

> **Binning is richer in production than in MixedWM38.** Real wafer test assigns each die
> a **bin code**, often under several schemes at once — e.g. **electrical/parametric**
> binning and **functional** binning. MixedWM38's pass/fail is just the degenerate
> single-scheme case. The per-die token design (§4) must therefore carry **multiple bin
> schemes per die simultaneously**, fused into one embedding (§5). This is forward-design:
> the architecture supports it, but MixedWM38 can only exercise the pass/fail special case.

> **Variable wafer size is a hard requirement, handled natively from day one.**
> Production wafer maps range from **10×10 to 100×100**. MixedWM38 at a fixed 52×52 is
> only the *reference* corpus. **No scaling/resizing is used** — resampling to a fixed
> grid distorts individual dies, and every die matters. The model consumes the **native
> grid** directly.

---

## 3. Requirements

### Functional
- F1. Ingest MixedWM38 into Unity Catalog; keep maps at **native resolution** (10–100 px/side).
- F2. Produce a **384-d, L2-normalized** embedding per wafer map, **per-die** at native res.
- F3. Batch-embed the full corpus and load it into a **Lakebase Search** index.
- F4. Serve single-wafer embeddings in real time via a **Model Serving** endpoint.
- F5. Nearest-neighbor query API over Lakebase (cosine), returning top-k similar wafers.
- F6. Evaluation harness reporting retrieval + clustering quality against the 38 labels.

### Non-functional
- N1. Embedding dim fixed at **384** (balanced size vs. quality for Lakebase).
- N2. Similarity query **p99 < 50 ms** at the serving corpus scale.
- N3. Corpus ~40K vectors today; index design valid to **1–10M** vectors.
- N4. **Smallest model that clears the eval bar** — minimize ViT depth/width/token count,
  not by skipping training (training is mandatory here, see §4).
- N5. Runs on a FEVM **AWS Stable Serverless** workspace; training + batch embedding use
  **serverless GPU or an attached classic GPU cluster** (GPU is required, not optional).
- N6. Model tracked in **MLflow**, registered in **Unity Catalog Model Registry**.

---

## 4. Model approach & literature justification

**Encoder: a custom per-die ViT (ViT-S scale, 384-d output), patch size = 1, trained
with DINO self-distillation on wafer maps.** No pretrained DINOv2 reuse.

**Why this, not frozen DINOv2.** The variable-size (10–100 px) + per-die-fidelity
requirements rule out DINOv2 ViT-S/14: patch 14 blends a 14×14 block of dies into one
token (coarse) and cannot represent a 10×10 wafer (smaller than one patch). Setting
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
(a wafer disc fills ≈ π/4 of its bounding square). Max tokens ≈ 10,000 at 100×100
(dense square) → ~7,800 typical; 10×10 → ≤100. Each die's feature is a **multi-scheme bin
descriptor** (electrical + functional + …, fused — §5), not just pass/fail; position comes
from its center/radius-normalized coordinates. Per-die tokens are what make multi-scheme
binning clean — a new scheme is just another embedding table added to the token, with no
change to sequence length or attention cost.

**Positional embeddings.** Encode each die's position as **coordinates normalized by
the wafer center and radius** — `(ũ, ṽ) = ((x−x_c)/R, (y−y_c)/R)` plus radial `r` —
rather than raw grid indices, then feed through a 2D sinusoidal / small-MLP encoding.
This is the field's standard answer to size variation (Deep Sets / Set-Transformer /
point-cloud wafer work, §13) and lets the model extend to grid sizes unseen in training
across the whole 10–100 range.

**Attention mechanism.** Use **induced set attention (ISAB)** rather than full
self-attention. A patch=1 per-die ViT is effectively a Set Transformer over dies, and
ISAB reduces attention cost from **O(N²) → O(N·m)** for `m` inducing points — the direct
mitigation for the ~10k-token cost at 100×100 (Lee et al., 2019; §13, §14). Output via
**pooling-by-multihead-attention (PMA)** or a CLS/masked-mean readout → L2-normalize → 384-d.

**Compute feasibility (per-die token budget).** Dropping off-wafer `no-die` cells leaves
~79% (π/4) of the square grid as tokens:

| Grid | Raw cells | On-wafer tokens |
|---|---:|---:|
| 10×10 | 100 | ~80 |
| 52×52 (MixedWM38) | 2,704 | ~2,100 |
| 100×100 (max) | 10,000 | ~7,850 |

So the real ceiling is **~8k tokens**, hit only by the largest dense wafers. At ViT-S scale
(d=384, 12 layers) full self-attention costs ≈ `4·N²·d` per layer ⇒ ~**1.4 TFLOP** per
100×100 wafer forward (~15 ms on one A10 GPU; ~1 s on CPU), while MixedWM38 at ~2.1k tokens
is ~**0.08 TFLOP** (trivial — DINO pretraining over all 38K maps is a few GPU-hours). The
naïve `N×N` score matrix (~0.7 GB/layer/sample at 8k) is the only memory concern and is
removed by **FlashAttention / memory-efficient SDPA** (O(N) memory). 8k tokens is routine
for transformers (cf. SAM ViT ~4k, long-context models 8k–32k). **Guidance: use full
attention for ≤~2k tokens — train/validate on MixedWM38 as-is — and switch the attention
block to ISAB (O(N·m), §4) before serving native 100×100 at scale**; a 2×2 patch for only
the largest wafers is the fallback. The Lakebase query p99 (N2) is independent of token
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
  so MixedWM38 runs unchanged. `no-die` cells are excluded from the token set entirely (§4).
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

1. **Ingest** — download MixedWM38 archive from the GitHub repo into UC Volume
   `/Volumes/main/wafer_embeddings/raw/`; parse into Delta
   `main.wafer_embeddings.wafer_maps` (`id`, `grid` array, `height`, `width`,
   `labels` multi-hot[38], `split`). *(Downloaded archive is untrusted: extract into its
   own empty dir, treat as data, never execute from it.)*
2. **Tokenize** — §5 transform (on-wafer die tokens + positions); cache.
3. **Train (mandatory)** — DINO self-distillation of the per-die ViT on GPU
   (serverless GPU or attached classic GPU cluster); memory-efficient / FlashAttention
   and gradient checkpointing to handle long sequences. Log to MLflow; ablate model size.
4. **Register** — best encoder → UC Model Registry `main.wafer_embeddings.encoder`.
5. **Batch embed** — job computes 384-d vectors for the corpus → Delta
   `main.wafer_embeddings.embeddings`.
6. **Sync to Lakebase** — see §9.

---

## 8. Evaluation & metrics

Labels (38 classes, multi-label) are used **for evaluation only** — the encoder trains
label-free (§4), and label-based scoring is strictly **post-hoc**, kept separate from any
supervised fine-tuning (the protocol the wafer-SSL literature follows [9][26]). Two
cross-cutting principles, grounded in the retrieval/IR literature [28][29]:

- **Separate three questions that are routinely conflated** [29]: (1) *representation
  quality* — are relevant wafers near each other under the embedding? (2) *index quality* —
  does the Lakebase ANN index reproduce exact search? (3) *serving performance* — latency /
  throughput. Score representation with **exact** search first (FAISS inner-product on
  L2-normalized vectors [19]); measure ANN degradation and latency separately (§8.6). An
  index can perfectly reproduce exact search while the embedding is semantically poor, and
  vice-versa.
- **Report per-pattern, not just aggregate.** MixedWM38 is imbalanced and mixed/rare
  patterns matter [16]; always report **macro + per-class** beside micro so common patterns
  can't mask rare-pattern failures.

### 8.1 Retrieval (primary goal)
Ranked-retrieval metrics over held-out queries [28]: **mAP@k, Recall@k, nDCG@k, MRR,
precision@k** (k ∈ {1,5,10,50}). State conventions explicitly (AP@K denominator; whether
"Recall@K" is fraction-of-relevant or hit-rate — report both, labeled), since scores under
different conventions aren't comparable [28]. **Relevance definition** (fix before M2):
primary = **Jaccard overlap ≥ τ** on the multi-label defect set; also report strict
**exact label-set match**; graded relevance (exact / shares-some / none) feeds nDCG.

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
Multi-label aware: **micro/macro F1, mAP, per-class recall**; balanced accuracy for
imbalance. Fit probe / kNN bank on train, tune on val, evaluate test once; freeze encoder +
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
- **Binning (validated later):** MixedWM38 exercises only the pass/fail case. With real
  multi-scheme bin maps, score retrieval/clustering *within* a single scheme and on the
  *fused* embedding, and verify robustness to a missing scheme (modality dropout, §5).

### 8.6 System & serving metrics (separate from representation)
- **ANN recall@k vs. exact** [29] over identical embeddings / metric / filters — gate so
  the index doesn't silently degrade retrieval.
- **Serving:** **QPS at a fixed ANN-recall target**, with p50/p95/**p99** latency (N2),
  stating what the timing includes (encode vs. index search); index build time; **encode
  latency per wafer by grid size** (§4 FLOP table); batch-embedding throughput. Recall–QPS
  curves, not QPS alone [29].
- **Footprint:** index size at 384-d across the 40K → synthetic 1M corpus (N3).

### 8.7 Protocol
- **Splits (leakage control):** fixed train/val/test; exclude test wafers from DINO
  pretraining for an **inductive** read (also report transductive); **de-duplicate**
  MixedWM38's GAN-generated maps so near-identical synthetic copies don't straddle splits.
- **Baselines:** random; flattened-pixels / PCA; **frozen off-the-shelf DINOv2** [21];
  ImageNet-pretrained CNN features (cf. [27]); a **supervised ViT** upper bound (MixedWM38
  reaches ~99% supervised, §13); and the external **graph-contrastive** reference —
  **ARI 0.89 / NMI 0.87 / silhouette 0.76** on WM-811K [9]; WaPIRL [2] / SWaCo [10] where
  reproducible.
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
**not** raw pgvector.

- Lakebase instance: `wafer-embeddings`; database `wafer_embeddings`.
- Table `wafer_embeddings(id bigint pk, labels …, embedding vector(384))`, populated by
  syncing the Delta embeddings table (UC synced table → Postgres `vector` column).
- Index:
  ```sql
  CREATE EXTENSION IF NOT EXISTS lakebase_vector CASCADE;
  CREATE INDEX wafer_embeddings_ann
    ON wafer_embeddings USING lakebase_ann (embedding vector_cosine_ops);
  ```
- Query (bind `$1` = 384-d query embedding):
  ```sql
  SELECT id, labels, 1 - (embedding <=> $1::vector(384)) AS cosine_similarity
  FROM wafer_embeddings
  ORDER BY embedding <=> $1::vector(384)
  LIMIT :k;
  ```

---

## 10. Serving

- **Batch** — scheduled job (§7.5) backfills/refreshes the corpus and re-syncs Lakebase.
- **Real-time** — Model Serving endpoint `wafer-encoder` wraps tokenize + encoder; input
  a native wafer grid, output a 384-d vector. New/online wafers are embedded then upserted
  into the Lakebase table (and picked up by the ANN index).

---

## 11. Databricks implementation stack

Workspace: FEVM **AWS Stable Serverless** (`aws_stable_serverless`). Tooling: **Python**
(≥3.11,<3.13), **uv** (env + wheel build), **Databricks Asset Bundles (DAB)**, Databricks
CLI. *(Version pins below reflect 2025 Databricks release notes — verify current values at
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
├── pyproject.toml            # uv project; deps: timm, torch, mlflow, numpy/polars, …
├── uv.lock
├── src/wafer_embeddings/
│   ├── data/                 # ingest + native per-die tokenizer (§5)
│   ├── model/                # per-die ViT (patch=1, ISAB) encoder (§4,§6)
│   ├── train/                # DINO self-distillation entrypoint
│   ├── embed/                # batch embedding job
│   ├── serve/                # pyfunc wrapper for Model Serving
│   └── eval/                 # retrieval/clustering harness (§8)
├── resources/                # *.yml job + endpoint + Lakebase resource defs
└── tests/
```
Wheel build via DAB artifact `build: uv build --wheel`; serverless jobs reference it through
`environments[].spec.dependencies: [./dist/*.whl]` + `environment_key`. Note `uv.lock` is not
auto-applied by installing the wheel — pin runtime deps in `[project].dependencies`. Use
`dynamic_version: true` for iterative dev deploys.

### 11.3 Training compute — Databricks AI Runtime (serverless GPU)
- **AI Runtime = serverless GPU.** ViT-S/DINO is small ⇒ default **single A10**
  (`GPU_1xA10`); single-node **H100** only if scaling up / FSDP. Workload YAML fields:
  `experiment_name`, `environment`, `compute`, `command`, optional UC Docker image.
- **Serverless GPU env v4** (Aug 2025): **PyTorch 2.7.1, torchvision 0.22.1, CUDA 12.6**;
  MLflow + PyTorch preinstalled. (Classic alt: **DBR 16.4 LTS ML GPU** — PyTorch 2.6.0+cu124,
  torchvision 0.21.0, CUDA 12.4 — for a pinned LTS cluster.)
- **Libraries (via uv):** `timm` (ViT backbone), a DINO implementation (or `lightly`),
  `numpy`/`polars` for wafer tensors.
- **Known rough edge:** launching serverless-GPU AI Runtime training *as a DAB job* is a
  documented gap. Plan: **DAB owns ingest / batch-embed / lakebase-sync / serving; the
  train step runs on AI Runtime** (notebook or workload CLI) and registers to UC.

### 11.4 MLflow + Unity Catalog registry
- `mlflow.set_registry_uri("databricks-uc")`; 3-part name `main.wafer_embeddings.encoder`.
- Package as **custom `mlflow.pyfunc.PythonModel`**: `load_context` loads the torch encoder;
  `predict` runs the §5 tokenizer + encoder → 384-d L2-normalized vector; provide
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

---

## 12. Deliverables & milestones

1. **M1 — Data & tokenize:** ingest MixedWM38, native-res per-die tokenizer, Delta tables.
2. **M2 — Train encoder:** DINO self-distillation ViT (patch=1) on GPU + eval report vs.
   threshold; size ablation for the smallest model that passes.
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
22. MixedWM38 dataset — https://github.com/Junliangwangdhu/WaferMap
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

- **Long-sequence attention cost:** dense 100×100 ≈ 8k tokens (quantified in §4). **ISAB
  induced attention (§4) reduces this to O(N·m)** and is the primary mitigation; combined
  with dropping `no-die` tokens (§4), FlashAttention / memory-efficient SDPA, gradient
  checkpointing, and (if still needed) a 2×2 patch for the largest wafers when latency (N2)
  or training cost is exceeded. MixedWM38 (~2k tokens) runs fine with full attention.
- **DINO from scratch on ~38K maps** may be data-limited — augmentation design (§5) and
  possibly pretraining on unlabeled WM-811K are levers; validate at M2.
- **Positional-encoding generalization:** confirm 2D sinusoidal transfers to grid sizes
  not seen in training (the whole 10–100 range), per the §8 size-generalization eval.
- **Multi-label eval semantics:** define "relevant neighbor" precisely (exact set vs.
  Jaccard) before M2.
- **Lakebase `lakebase_ann` dimension limits** at 384-d: confirm against current docs.
- **Canonical bin taxonomy (owned by domain experts):** cross-product/cross-fab retrieval
  needs a shared bin ontology (pass/fail + failure-category) so per-scheme embeddings are
  comparable across datasets. Until it exists, cross-dataset retrieval is only meaningful
  within a shared scheme. Defining this mapping is an open, non-modeling task (§5, §11.7).
- **Continuous parametrics (deferred):** feeding per-die measurements (Idd/Vmin/Fmax)
  alongside categorical bins is a natural extension once parametric test data is available.
- **Orientation-aware targets (future):** rotation/flip invariance is correct for the
  standard morphology taxonomy (§5), grounded in [2][30]. If a dataset encodes scratch
  direction, edge sector, or orientation-dependent root cause, set
  `orientation_invariant=false` and re-ablate (§8.7) — those are different targets from
  MixedWM38/WM-811K.
