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


def test_register_helpers():
    from wafer_embeddings.jobs.register import _args, pip_requirements, sample_input

    req = pip_requirements("2.14.1+cu130", "2.5.3")
    assert "torch==2.14.1" in req and "numpy==2.5.3" in req  # CUDA local tag dropped
    assert any("download.pytorch.org/whl/cpu" in r for r in req)  # CPU wheels for serving
    x = sample_input()
    assert list(x.columns) == ["height", "width", "wafer_map"]
    assert all(len(m) == h * w for h, w, m in zip(x.height, x.width, x.wafer_map))
    a = _args(["--catalog", "c", "--schema", "s", "--run-id", "r1"])
    assert a.name == "wafer_encoder" and a.alias == "champion"


def test_pyfunc_round_trip(tmp_path):
    mlflow = pytest.importorskip("mlflow")  # only with the jobs extra
    from wafer_embeddings.jobs.register import sample_input
    from wafer_embeddings.serving.pyfunc import CHECKPOINT, WaferEncoderModel

    torch.manual_seed(0)
    sd = build_encoder(CONFIG).state_dict()
    ck = tmp_path / "ck.pt"
    torch.save({"state_dict": sd, "config": CONFIG}, ck)
    path = str(tmp_path / "model")
    mlflow.pyfunc.save_model(
        path, python_model=WaferEncoderModel(), artifacts={CHECKPOINT: str(ck)}
    )
    out = mlflow.pyfunc.load_model(path).predict(sample_input())
    emb = np.array(out["embedding"].tolist())
    assert emb.shape == (2, 16)
    np.testing.assert_allclose(np.linalg.norm(emb, axis=1), 1.0, atol=1e-5)


def test_embed_frames_rows_match_encoder_and_metadata():
    import pandas as pd

    from wafer_embeddings.jobs.embed import OUTPUT_COLUMNS, embed_frames

    enc, _ = _encoder()
    maps = [_map(i, 9 + i, 11) for i in range(4)]
    pdf = pd.DataFrame(
        {
            "id": [10, 11, 12, 13],
            "height": [m.shape[0] for m in maps],
            "width": [m.shape[1] for m in maps],
            "wafer_map": [m.reshape(-1).tolist() for m in maps],
            "label": ["Center", None, "Loc", None],
            "label_id": [0, -1, 4, -1],
            "split": ["train", "val", "test", "train"],
            "lot": ["lotA", "lotA", "lotB", None],
        }
    )
    (out,) = list(embed_frames(iter([pdf]), enc, "2"))
    assert tuple(out.columns) == OUTPUT_COLUMNS
    np.testing.assert_allclose(np.array(out["embedding"].tolist()), enc.embed_maps(maps), atol=1e-6)
    for code, m in zip(out["map_code"], maps):
        np.testing.assert_array_equal(decode_map(code), m)
    assert out["n_dies"].tolist() == [int((m != NO_DIE).sum()) for m in maps]
    np.testing.assert_allclose(
        out["fail_frac"], [(m == FAIL).sum() / (m != NO_DIE).sum() for m in maps], rtol=1e-6
    )
    assert out["model_version"].unique().tolist() == ["2"] and out["id"].tolist() == [
        10,
        11,
        12,
        13,
    ]


def test_embed_parquet_streams_files(tmp_path):
    import pyarrow as pa
    import pyarrow.dataset as ds
    import pyarrow.parquet as pq

    from wafer_embeddings.jobs.embed import OUTPUT_COLUMNS, embed_parquet

    enc, _ = _encoder()
    maps = [_map(i, 8, 9) for i in range(7)]
    tbl = pa.table(
        {
            "id": list(range(7)),
            "height": [8] * 7,
            "width": [9] * 7,
            "wafer_map": [m.reshape(-1).tolist() for m in maps],
            "label": ["Loc"] * 7,
            "label_id": [4] * 7,
            "split": ["test"] * 7,
            "lot": ["l"] * 7,
        }
    )
    src = tmp_path / "src"
    src.mkdir()
    pq.write_table(tbl, src / "a.parquet")
    n = embed_parquet(ds.dataset(str(src)), enc, str(tmp_path / "out"), "2", rows_per_file=3)
    out = ds.dataset(str(tmp_path / "out")).to_table().to_pandas().sort_values("id")
    assert n == 7 and len(list((tmp_path / "out").glob("part-*.parquet"))) == 3  # 3+3+1
    assert tuple(out.columns) == OUTPUT_COLUMNS
    np.testing.assert_allclose(np.array(out["embedding"].tolist()), enc.embed_maps(maps), atol=1e-6)


def test_n_params_counts_every_encoder_weight():
    from wafer_embeddings.model.encoder import count_parameters

    enc, model = _encoder()
    expected = sum(p.numel() for p in model.parameters())
    assert enc.n_params == expected == count_parameters(model) > 0
    # inference freezes the weights; the total still counts them, trainable-only does not
    assert count_parameters(enc.model, trainable_only=True) == 0
    assert count_parameters(enc.model) == expected


def test_register_latest_run_selection():
    from wafer_embeddings.jobs.register import _args, pick_run_id, run_filter

    assert run_filter("pipeline-train") == (
        "attributes.run_name = 'pipeline-train' AND attributes.status = 'FINISHED'"
    )
    assert run_filter("pipeline-train", 1700000000000).endswith(
        "AND attributes.start_time >= 1700000000000"
    )
    with pytest.raises(ValueError):
        run_filter("x' OR '1'='1")
    assert pick_run_id(["newest", "older"], "pipeline-train") == "newest"
    with pytest.raises(SystemExit):
        pick_run_id([], "pipeline-train", 5)  # training didn't finish: don't register
    a = _args(
        "--catalog c --schema s --latest-run-name pipeline-train --since-ms 7 "
        "--experiment /x --alias candidate".split()
    )
    assert (a.latest_run_name, a.since_ms, a.alias) == ("pipeline-train", 7, "candidate")
