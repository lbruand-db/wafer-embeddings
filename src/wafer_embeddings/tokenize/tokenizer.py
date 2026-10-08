"""Turn a native-resolution wafer map into per-die tokens (SPECS.md §4, §5).

One token per **on-wafer die** (no-die cells are dropped). Each token carries a
discrete state (pass/fail — the degenerate single-scheme case of multi-scheme
binning) and a position given by coordinates **normalized by the wafer center and
radius** (not grid index, not defect bbox), so the representation is identical in
shape for a 10x10 and a 100x100 wafer and generalizes across sizes.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from wafer_embeddings.data.mixedwm38 import FAIL, NO_DIE

# Token state ids (no-die is dropped, never a token). Multi-scheme binning (SPECS.md
# §5) generalizes this to several categorical fields; pass/fail is the base case.
STATE_PASS, STATE_FAIL = 0, 1
N_STATES = 2


@dataclass(frozen=True)
class TokenizedWafer:
    coords: np.ndarray  # (M, 3) float32: (u, v, r) center/radius-normalized
    state_ids: np.ndarray  # (M,) int64 in {STATE_PASS, STATE_FAIL}

    @property
    def n_tokens(self) -> int:
        return int(self.coords.shape[0])


def tokenize(wafer_map: np.ndarray, eps: float = 1e-6) -> TokenizedWafer:
    """Tokenize one wafer map (H, W) with cells in {no-die, pass, fail}."""
    m = np.asarray(wafer_map)
    ys, xs = np.nonzero(m != NO_DIE)  # on-wafer dies only
    if ys.size == 0:
        return TokenizedWafer(np.zeros((0, 3), np.float32), np.zeros((0,), np.int64))
    cy, cx = ys.mean(), xs.mean()  # wafer center = on-wafer centroid
    du = xs.astype(np.float64) - cx
    dv = ys.astype(np.float64) - cy
    radius = max(float(np.sqrt(du * du + dv * dv).max()), eps)
    u = du / radius
    v = dv / radius
    r = np.sqrt(u * u + v * v)
    coords = np.stack([u, v, r], axis=1).astype(np.float32)
    state_ids = np.where(m[ys, xs] == FAIL, STATE_FAIL, STATE_PASS).astype(np.int64)
    return TokenizedWafer(coords, state_ids)


def cap_tokens(w: TokenizedWafer, max_tokens: int, rng: np.random.Generator) -> TokenizedWafer:
    """Subsample dies to ``max_tokens`` while **keeping every FAIL die** (SPECS.md §5).

    WM-811K maps vary from tiny to tens of thousands of dies; capping keeps the
    attention/activation tensors bounded. Defects are **sparse** (often <5% of dies), so a
    *uniform* random cap decimates the exact signal that defines a defect class — e.g. on a
    3k-die map capped to 512 it keeps only ~1/6 of the failing dies, blurring the pattern.
    We therefore keep **all** FAIL dies and subsample only PASS dies to fill the remaining
    budget; positions/states of kept dies are unchanged. (Measured on WM-811K: this lifts
    untrained 1-NN class accuracy from ~0.50 to ~0.74 — see git history / scratch/diag.py.)
    If FAIL dies alone exceed the budget (pathological, e.g. near-full maps) we subsample
    them too so the cap still holds.
    """
    if w.n_tokens <= max_tokens:
        return w
    fail = np.nonzero(w.state_ids == STATE_FAIL)[0]
    npass = np.nonzero(w.state_ids != STATE_FAIL)[0]
    if fail.size >= max_tokens:
        keep = rng.choice(fail, size=max_tokens, replace=False)
    else:
        budget = max_tokens - fail.size
        kept_pass = rng.choice(npass, size=budget, replace=False)
        keep = np.concatenate([fail, kept_pass])
    keep = np.sort(keep)
    return TokenizedWafer(w.coords[keep], w.state_ids[keep])


def collate(batch: list[TokenizedWafer]):
    """Pad a batch to the max token count; return torch tensors + validity mask.

    Returns ``(coords [B,L,3] float32, state_ids [B,L] int64, mask [B,L] bool)``
    where ``mask`` is True for real tokens. Padding tokens must be ignored by the
    encoder's attention (SPECS.md §5).
    """
    import torch

    b = len(batch)
    lengths = [t.n_tokens for t in batch]
    lmax = max(lengths) if lengths else 0
    coords = torch.zeros((b, lmax, 3), dtype=torch.float32)
    state_ids = torch.zeros((b, lmax), dtype=torch.int64)
    mask = torch.zeros((b, lmax), dtype=torch.bool)
    for i, t in enumerate(batch):
        n = t.n_tokens
        if n:
            coords[i, :n] = torch.from_numpy(t.coords)
            state_ids[i, :n] = torch.from_numpy(t.state_ids)
            mask[i, :n] = True
    return coords, state_ids, mask
