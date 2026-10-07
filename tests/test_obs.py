import logging

from wafer_embeddings import obs


def test_periodic_logs_milestones_and_final():
    msgs: list[str] = []
    cb = obs.periodic(msgs.append, "parse", every_frac=0.25)
    for i in range(1, 101):
        cb(i, 100)
    # Throttled: a handful of lines, not 100.
    assert len(msgs) <= 6
    # Always emits the final 100% line.
    assert any("100/100 (100%)" in m for m in msgs)
    assert all(m.startswith("parse: ") for m in msgs)


def test_periodic_rejects_bad_fraction():
    import pytest

    for bad in (0.0, -0.1, 1.5):
        with pytest.raises(ValueError):
            obs.periodic(lambda s: None, "x", every_frac=bad)


def test_periodic_handles_zero_total():
    msgs: list[str] = []
    obs.periodic(msgs.append, "x")(0, 0)  # no ZeroDivision
    assert msgs and "0/0" in msgs[0]


def test_get_logger_is_configured_once():
    a = obs.get_logger("wafer_embeddings.test")
    b = obs.get_logger("wafer_embeddings.test")
    assert a is b
    assert a.level == logging.INFO
    assert len(a.handlers) == 1  # not duplicated on repeated calls
