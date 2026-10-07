"""Lightweight observability: timestamped logging + throttled progress.

Job-friendly by design: in Databricks job stdout, tqdm-style carriage-return bars
render as noise, so we emit periodic percentage lines instead. The progress helper
takes a plain ``log`` callable (e.g. ``logger.info``) so it is trivially testable.
"""

from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from typing import Callable

_CONFIGURED: set[str] = set()


def get_logger(name: str = "wafer_embeddings") -> logging.Logger:
    """Return a logger with a single timestamped handler (configured once)."""
    logger = logging.getLogger(name)
    if name not in _CONFIGURED:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s [%(name)s] %(message)s", "%H:%M:%S")
        )
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
        _CONFIGURED.add(name)
    return logger


def periodic(log: Callable[[str], None], desc: str, every_frac: float = 0.1):
    """Return a ``cb(done, total)`` that logs progress at ~``every_frac`` steps.

    Always logs the final (done == total) tick. ``log`` is any ``str -> None``.
    """
    if not 0 < every_frac <= 1:
        raise ValueError("every_frac must be in (0, 1]")
    state = {"next_pct": 0.0}

    def cb(done: int, total: int) -> None:
        pct = 100.0 * done / max(total, 1)
        if done >= total or pct >= state["next_pct"]:
            log(f"{desc}: {done}/{total} ({pct:.0f}%)")
            # advance threshold past the current pct so we don't double-log
            state["next_pct"] = (int(pct / (every_frac * 100)) + 1) * (every_frac * 100)

    return cb


@contextmanager
def stage(logger: logging.Logger, desc: str):
    """Log ``desc`` start/finish with elapsed seconds."""
    logger.info(f"{desc} ...")
    t0 = time.perf_counter()
    try:
        yield
    finally:
        logger.info(f"{desc} done in {time.perf_counter() - t0:.1f}s")
