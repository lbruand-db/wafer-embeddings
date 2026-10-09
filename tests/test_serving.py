import numpy as np
import pytest
import torch

from wafer_embeddings.data.mixedwm38 import FAIL, NO_DIE, PASS
from wafer_embeddings.serving import WaferEncoder, build_encoder, decode_map, encode_map

CONFIG = {
    "embed_dim": 16,
    "depth": 2,
    "heads": 4,
    "attention": "isab",
    "pre_norm": True,
    "max_tokens": 64,
    "steps": 10,
}  # extra training keys are ignored


def _map(seed, h=12, w=15):
    rng = np.random.default_rng(seed)
    m = np.full((h, w), PASS, dtype=np.uint8)
    m[0, :3] = NO_DIE
    m[rng.random((h, w)) < 0.2] = FAIL
    return m


def _encoder(tmp_path=None):
    torch.manual_seed(0)
    model = build_encoder(CONFIG)
    sd = {k: v.clone() for k, v in model.state_dict().items()}
    if tmp_path is None:
        return WaferEncoder(sd, CONFIG), model
    path = tmp_path / "ck.pt"
    torch.save({"state_dict": sd, "config": CONFIG}, path)
    return WaferEncoder.from_checkpoint(str(path)), model


def test_checkpoint_round_trip_and_shapes(tmp_path):
    enc, _ = _encoder(tmp_path)
    maps = [_map(i) for i in range(5)]
    emb = enc.embed_maps(maps, batch_size=2)
    assert emb.shape == (5, 16) and emb.dtype == np.float32
    np.testing.assert_allclose(np.linalg.norm(emb, axis=1), 1.0, atol=1e-5)
    assert enc.max_tokens == 64 and enc.dim == 16


def test_embedding_independent_of_batch_composition():
    enc, _ = _encoder()
    big = _map(9, 30, 30)  # > max_tokens -> capped with the per-map seeded RNG
    alone = enc.embed_maps([big])
    mixed = enc.embed_maps([_map(1), big, _map(2)], batch_size=3)
    np.testing.assert_allclose(alone[0], mixed[1], atol=1e-5)


def test_embed_rows_matches_embed_maps():
    enc, _ = _encoder()
    maps = [_map(i, 10 + i, 12) for i in range(3)]
    rows = enc.embed_rows(
        [m.shape[0] for m in maps],
        [m.shape[1] for m in maps],
        [m.reshape(-1).tolist() for m in maps],
    )
    np.testing.assert_allclose(rows, enc.embed_maps(maps), atol=1e-6)


def test_matches_the_raw_model():
    from wafer_embeddings.tokenize.tokenizer import collate, tokenize

    enc, model = _encoder()
    m = _map(3, 6, 8)  # 45 dies < max_tokens: no cap involved
    with torch.no_grad():
        ref = model.eval()(*collate([tokenize(m)])).numpy()
    np.testing.assert_allclose(enc.embed_maps([m]), ref, atol=1e-5)


@pytest.mark.parametrize("shape", [(1, 1), (3, 5), (7, 4), (52, 52)])
def test_encode_decode_map_round_trip(shape):
    rng = np.random.default_rng(0)
    m = rng.integers(0, 3, size=shape).astype(np.uint8)
    s = encode_map(m)
    assert s.startswith(f"{shape[0]}x{shape[1]}:")
    np.testing.assert_array_equal(decode_map(s), m)
    if m.size >= 100:
        assert len(s) < m.size  # compact: < 1 char per die


def test_encode_map_rejects_bad_input():
    with pytest.raises(ValueError):
        encode_map(np.zeros((0, 3), np.uint8))
    with pytest.raises(ValueError):
        encode_map(np.full((2, 2), 7, np.uint8))


def test_encoder_does_not_change_global_grad_mode():
    assert torch.is_grad_enabled()
    enc, _ = _encoder()
    enc.embed_maps([_map(0)])
    assert torch.is_grad_enabled()  # other code in the process can still train
