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
(dense square) → ~7,800 typical; 10×10 → ≤100. Each die's state (pass/fail) is embedded;
position comes from its `(row, col)`.

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

- **Per-die features.** For each on-wafer die, a feature encoding its state — one-hot
  `(pass, fail)` (2-d), with `no-die` cells excluded from the token set entirely (§4).
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
- **Augmentation (DINO multi-crop, die-semantics-preserving).** 90° rotations, flips,
  and **random grid crops** (which also produce the smaller/variable-size views DINO's
  multi-crop needs — directly training the variable-res capability). **No** color jitter
  / blur / resampling — those would corrupt discrete die states.

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

## 8. Evaluation

Labels (38 classes, multi-label) are used **only here**:
- **Retrieval:** precision@k / mAP where a neighbor is "relevant" if it shares the
  defect-pattern label set (exact-match and Jaccard-overlap variants).
- **Classification probe:** kNN accuracy and linear-probe F1 on frozen embeddings
  (micro/macro, multi-label aware).
- **Clustering:** NMI and ARI of k-means / HDBSCAN assignments vs. the 38 labels.
- **Generalization across size:** eval on held-out maps at grid sizes/crops not seen in
  training, to validate the native variable-res path.
- **Size/latency:** embedding dim (fixed 384), encoder token-count vs. accuracy curve,
  Lakebase query p50/p99 at 40K and a synthetic 1M-vector corpus.

A quantitative quality threshold (retrieval mAP ≥ X, NMI ≥ Y) is set at project start and
gates "model is good enough / stop shrinking it."

**Baseline to beat.** The strongest comparable label-free wafer result is the
graph-contrastive encoder of Awais et al. (2026): **ARI 0.89 / NMI 0.87** clustering on
WM-811K (§13/§14). Use it as the clustering reference point; also benchmark the pure DINO
encoder against a SWaCo-style same-class-positive variant for *retrieval* (the
false-negative caveat, §4).

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

## 11. Environment & template

- **Workspace:** FEVM **AWS Stable Serverless** (`aws_stable_serverless`).
- **Compute:** serverless for ingest/sync/serving; **GPU required** for training + batch
  embedding (serverless GPU or an attached classic GPU cluster).
- **Packaging:** Databricks Asset Bundle (DAB) — jobs (ingest, train, batch-embed,
  lakebase-sync, eval), the Model Serving endpoint, and the Lakebase instance — so the
  whole template deploys reproducibly.
- **MLOps:** MLflow experiments + UC Model Registry; config-driven (dataset path,
  embedding dim, patch/token settings, model size, eval thresholds) so the template
  retargets to other wafer datasets.

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
