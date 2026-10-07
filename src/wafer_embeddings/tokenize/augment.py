"""Die-semantics-preserving augmentations on tokens (SPECS.md §5).

The augmentation set defines the invariances the embedding learns. Operating on
per-die *point* tokens makes geometric transforms lossless at any angle (we rotate
coordinates, never resample pixels). Defaults are orientation-invariant (rotation +
flip), appropriate for the MixedWM38 morphology taxonomy; disable for
orientation-aware targets (`orientation_invariant=False`, SPECS.md §5/§15).
"""

from __future__ import annotations

import numpy as np

from wafer_embeddings.tokenize.tokenizer import TokenizedWafer


def rotate_coords(coords: np.ndarray, angle: float) -> np.ndarray:
    """Rotate (u, v) by ``angle`` radians about the wafer center; r is preserved."""
    u, v, r = coords[:, 0], coords[:, 1], coords[:, 2]
    c, s = np.cos(angle), np.sin(angle)
    nu = c * u - s * v
    nv = s * u + c * v
    return np.stack([nu, nv, r], axis=1).astype(np.float32)


def flip_coords(coords: np.ndarray, axis: str = "u") -> np.ndarray:
    """Mirror across the u- or v-axis; r is preserved."""
    out = coords.copy()
    out[:, 0 if axis == "u" else 1] *= -1.0
    return out.astype(np.float32)


def toggle_die_noise(state_ids: np.ndarray, p: float, rng: np.random.Generator) -> np.ndarray:
    """Flip pass<->fail on a Bernoulli(p) fraction of dies (SPECS.md §5)."""
    if p <= 0:
        return state_ids.copy()
    flip = rng.random(state_ids.shape) < p
    out = state_ids.copy()
    out[flip] = 1 - out[flip]  # two states: pass<->fail
    return out


def crop_window(
    coords: np.ndarray,
    state_ids: np.ndarray,
    scale: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Keep tokens inside a random sub-window — a DINO local view.

    Coordinates are **not** re-normalized, so each kept die retains its position
    relative to the wafer boundary (avoids Loc<->Edge-Loc confusion, SPECS.md §5).
    Falls back to all tokens if the window would be empty.
    """
    if coords.shape[0] == 0 or scale >= 1.0:
        return coords.copy(), state_ids.copy()
    lo = coords[:, :2].min(axis=0)
    hi = coords[:, :2].max(axis=0)
    span = (hi - lo) * scale
    start = lo + rng.random(2) * (hi - lo - span)
    end = start + span
    inside = np.all((coords[:, :2] >= start) & (coords[:, :2] <= end), axis=1)
    if not inside.any():
        return coords.copy(), state_ids.copy()
    return coords[inside].copy(), state_ids[inside].copy()


def cutout(
    coords: np.ndarray,
    state_ids: np.ndarray,
    n_regions: int,
    size: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Drop tokens falling inside up to ``n_regions`` random square holes."""
    if coords.shape[0] == 0 or n_regions <= 0:
        return coords.copy(), state_ids.copy()
    drop = np.zeros(coords.shape[0], dtype=bool)
    lo = coords[:, :2].min(axis=0)
    hi = coords[:, :2].max(axis=0)
    for _ in range(n_regions):
        center = lo + rng.random(2) * (hi - lo)
        half = size / 2.0
        in_box = np.all(np.abs(coords[:, :2] - center) <= half, axis=1)
        drop |= in_box
    keep = ~drop
    if not keep.any():  # never drop everything
        return coords.copy(), state_ids.copy()
    return coords[keep].copy(), state_ids[keep].copy()


def random_view(
    wafer: TokenizedWafer,
    rng: np.random.Generator,
    *,
    orientation_invariant: bool = True,
    crop_scale: float = 0.7,
    die_noise_p: float = 0.005,
    cutout_regions: int = 2,
    cutout_size: float = 0.3,
) -> TokenizedWafer:
    """Compose one augmented DINO view (SPECS.md §5).

    Rotation/flip are applied only in orientation-invariant mode. Crop + die-noise +
    cutout always apply. We deliberately do not chain crop with a coordinate shift
    (crop+shift degraded in WaPIRL, SPECS.md §5).
    """
    coords, state_ids = wafer.coords, wafer.state_ids
    coords, state_ids = crop_window(coords, state_ids, crop_scale, rng)
    if orientation_invariant:
        coords = rotate_coords(coords, float(rng.uniform(0, 2 * np.pi)))
        if rng.random() < 0.5:
            coords = flip_coords(coords, "u")
        if rng.random() < 0.5:
            coords = flip_coords(coords, "v")
    coords, state_ids = cutout(coords, state_ids, cutout_regions, cutout_size, rng)
    state_ids = toggle_die_noise(state_ids, die_noise_p, rng)
    return TokenizedWafer(coords.astype(np.float32), state_ids)
