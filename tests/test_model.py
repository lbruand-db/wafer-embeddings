import numpy as np
import pytest
import torch

from wafer_embeddings.model.encoder import PerDieViT
from wafer_embeddings.tokenize import tokenizer as tk

CONFIGS = [
    dict(attention=att, readout=ro, pre_norm=pn)
    for att in ("full", "isab")
    for ro in ("mean", "pma")
    for pn in (False, True)
]


def _small(**kw):
    torch.manual_seed(0)
    m = PerDieViT(embed_dim=16, width=32, depth=2, n_heads=4, n_inducing=8, **kw)
    return m.eval()


def _batch(lengths, seed=0):
    rng = np.random.default_rng(seed)
    tokens = []
    for n in lengths:
        coords = rng.standard_normal((n, 3)).astype(np.float32)
        states = rng.integers(0, tk.N_STATES, size=n).astype(np.int64)
        tokens.append(tk.TokenizedWafer(coords, states))
    return tk.collate(tokens)


@pytest.mark.parametrize("cfg", CONFIGS)
def test_output_shape_and_unit_norm(cfg):
    model = _small(**cfg)
    coords, states, mask = _batch([5, 9, 3])
    with torch.no_grad():
        emb = model(coords, states, mask)
    assert emb.shape == (3, 16)
    assert torch.allclose(emb.norm(dim=-1), torch.ones(3), atol=1e-5)
    assert torch.isfinite(emb).all()


@pytest.mark.parametrize("cfg", CONFIGS)
def test_permutation_invariance(cfg):
    model = _small(**cfg)
    coords, states, mask = _batch([7])
    g = torch.Generator().manual_seed(1)
    perm = torch.randperm(coords.shape[1], generator=g)
    with torch.no_grad():
        a = model(coords, states, mask)
        b = model(coords[:, perm], states[:, perm], mask[:, perm])
    assert torch.allclose(a, b, atol=1e-4)


@pytest.mark.parametrize("cfg", CONFIGS)
def test_padding_invariance(cfg):
    model = _small(**cfg)
    coords, states, mask = _batch([6])
    pad = 4
    coords2 = torch.cat([coords, torch.zeros(1, pad, 3)], dim=1)
    states2 = torch.cat([states, torch.zeros(1, pad, dtype=torch.int64)], dim=1)
    mask2 = torch.cat([mask, torch.zeros(1, pad, dtype=torch.bool)], dim=1)
    with torch.no_grad():
        a = model(coords, states, mask)
        b = model(coords2, states2, mask2)
    assert torch.allclose(a, b, atol=1e-4)


def test_default_dims_emit_384():
    model = PerDieViT(depth=2).eval()  # keep depth small for speed, default dims else
    coords, states, mask = _batch([4, 11])
    with torch.no_grad():
        emb = model(coords, states, mask)
    assert emb.shape == (2, 384)
    assert torch.allclose(emb.norm(dim=-1), torch.ones(2), atol=1e-5)


def test_gradients_flow():
    model = _small(attention="isab", readout="pma")
    coords, states, mask = _batch([8])
    emb = model(coords, states, mask)
    emb.sum().backward()
    grads = [p.grad for p in model.parameters() if p.requires_grad]
    assert any(g is not None and torch.isfinite(g).all() and g.abs().sum() > 0 for g in grads)


def test_grad_checkpoint_forward_backward():
    torch.manual_seed(0)
    m = PerDieViT(embed_dim=16, width=32, depth=3, n_heads=4, grad_checkpoint=True).train()
    coords, states, mask = _batch([6, 4])
    emb = m(coords, states, mask)
    assert emb.shape == (2, 16)
    emb.sum().backward()
    grads = [p.grad for p in m.parameters() if p.requires_grad]
    assert any(g is not None and torch.isfinite(g).all() and g.abs().sum() > 0 for g in grads)


def test_bad_config_rejected():
    with pytest.raises(ValueError):
        PerDieViT(attention="sparse")
    with pytest.raises(ValueError):
        PerDieViT(readout="max")


def test_pre_norm_adds_final_norm_and_trains():
    post, pre = _small(attention="isab"), _small(attention="isab", pre_norm=True)
    assert post.final_norm is None and pre.final_norm is not None
    pre.train()
    coords, states, mask = _batch([8, 5])
    pre(coords, states, mask).sum().backward()
    g = pre.blocks[0].mab0.ln_k.weight.grad  # the key norm is on the gradient path
    assert g is not None and torch.isfinite(g).all() and g.abs().sum() > 0
