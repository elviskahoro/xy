"""Server-side completion of bounded Reflex selection events."""

from __future__ import annotations

import contextlib
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

import reflex_xy
from reflex_xy.registry import registry
from xy._figure import Figure


def _event(token: str, selection: dict) -> dict:
    return {"version": 1, "type": "select_end", "token": token, "selection": selection}


def _box(data_bounds: dict | None = None) -> dict:
    bounds = {"x0": 0, "x1": 1, "y0": 0, "y1": 1} if data_bounds is None else data_bounds
    return {"kind": "box", "data_bounds": bounds, "total_count": 0}


def test_resolve_selection_box_and_lasso(_fresh_registry):
    fig = Figure().scatter(np.arange(6.0), np.arange(6.0))
    token = registry.register(fig)

    box = reflex_xy.resolve_selection(
        _event(
            token,
            {
                "kind": "box",
                "data_bounds": {"x0": 1, "x1": 3, "y0": 0, "y1": 4},
                "total_count": 3,
            },
        )
    )
    assert box is not None
    np.testing.assert_array_equal(box.index, [1, 2, 3])

    lasso = reflex_xy.resolve_selection(
        _event(
            token,
            {
                "kind": "lasso",
                "polygon": [[-1, -1], [4, -1], [2, 4]],
                "total_count": 3,
            },
        )
    )
    assert lasso is not None
    np.testing.assert_array_equal(lasso.index, [0, 1, 2])


def test_resolve_selection_returns_none_for_clear_unknown_and_garbage(_fresh_registry):
    assert reflex_xy.resolve_selection(_event("missing", {"kind": "box"})) is None
    assert (
        reflex_xy.resolve_selection(_event("missing", {"kind": "clear", "cleared": True})) is None
    )
    assert reflex_xy.resolve_selection({}) is None
    assert reflex_xy.resolve_selection(None) is None


# ---------------------------------------------------------------------------
# Concurrency / generation-lease discipline (G1–G6).
# ---------------------------------------------------------------------------


def _make_scatter(n: int = 200_000, seed: int = 0) -> Figure:
    rng = np.random.default_rng(seed)
    xs = rng.random(n)
    ys = rng.random(n)
    return Figure().scatter(xs, ys)


class _BlockingSelectFigure:
    """A figure stub whose ``select_range`` blocks until released, so a test
    can drive a deterministic race between the kernel read and a concurrent
    registry mutation (publish / sweep)."""

    def __init__(self) -> None:
        self.started = threading.Event()
        self.resume = threading.Event()

    def select_range(self, *args, **kwargs):
        self.started.set()
        assert self.resume.wait(2.0)
        return {0: np.array([0], dtype=np.uint32)}

    def select_polygon(self, *args, **kwargs):  # pragma: no cover
        raise AssertionError("select_polygon must not be reached by a box event")


def test_resolve_selection_no_spurious_none_under_concurrent_append(_fresh_registry):
    """G1: a valid selection resolved while the figure is being streamed to
    must not return ``None`` because of a torn x/y read."""
    n = 200_000
    fig = _make_scatter(n)
    token = registry.register(fig)
    box = _event(token, _box({"x0": 0.0, "x1": 1.0, "y0": 0.0, "y1": 1.0}))

    counts = {"ok": 0, "none": 0, "leak": 0}
    stop = threading.Event()

    def resolver() -> None:
        while not stop.is_set():
            try:
                res = reflex_xy.resolve_selection(box)
            except Exception:  # noqa: BLE001 - must never happen
                counts["leak"] += 1
                continue
            if res is None:
                counts["none"] += 1
            else:
                counts["ok"] += 1

    def appender(seed: int) -> None:
        rng = np.random.default_rng(seed)
        while not stop.is_set():
            with contextlib.suppress(Exception):
                registry.append(token, [rng.random()], [rng.random()])

    prev = sys.getswitchinterval()
    sys.setswitchinterval(1e-6)
    try:
        with ThreadPoolExecutor(max_workers=4) as pool:
            futs = [pool.submit(resolver), pool.submit(resolver)]
            futs += [pool.submit(appender, 1), pool.submit(appender, 2)]
            time.sleep(2.0)
            stop.set()
            for f in futs:
                f.result(timeout=5.0)
    finally:
        sys.setswitchinterval(prev)

    assert counts["leak"] == 0, "resolve_selection must never raise out"
    assert counts["none"] == 0, f"spurious None under concurrent append: {counts}"


def test_resolve_selection_discards_stale_generation_after_publish(_fresh_registry):
    """G2: an in-flight resolution whose kernel read races a republish returns
    ``None``, never rows from the superseded generation.

    Deterministic via a blocking ``select_range``: the resolver holds the
    kernel read open while the main thread publishes a new generation, then
    releases it — ``is_current`` must discard the stale result.
    """
    blocking = _BlockingSelectFigure()
    token = registry.register(blocking)
    first = registry.get(token)
    assert first is not None
    box = _event(token, _box({"x0": 0.0, "x1": 1.0, "y0": 0.0, "y1": 1.0}))

    with ThreadPoolExecutor(max_workers=1) as pool:
        fut = pool.submit(reflex_xy.resolve_selection, box)
        assert blocking.started.wait(1.0)
        new_fig = _make_scatter(16, seed=1)
        registry.publish(token, new_fig, broadcast=False)
        blocking.resume.set()
        res = fut.result(timeout=5.0)

    new = registry.get(token)
    assert new is not None
    assert not registry.is_current(token, first)
    assert registry.is_current(token, new)
    assert res is None, "stale-generation resolution must be discarded as None"


def test_resolve_selection_lease_blocks_ttl_sweep(_fresh_registry):
    """G3: while ``resolve_selection`` holds the generation lease, a TTL sweep
    must skip the entry even if its ``last_access`` is past the TTL."""
    blocking = _BlockingSelectFigure()
    original_ttl = registry._ttl
    registry._ttl = 0.0
    try:
        entry = registry.publish("slow", blocking, broadcast=False)
        box = _event("slow", _box({"x0": 0.0, "x1": 1.0, "y0": 0.0, "y1": 1.0}))

        with ThreadPoolExecutor(max_workers=1) as pool:
            fut = pool.submit(reflex_xy.resolve_selection, box)
            assert blocking.started.wait(1.0)
            forced_now = entry.last_access + 1_000_000.0
            assert registry.sweep(now=forced_now) == []
            assert registry.is_current("slow", entry)
            blocking.resume.set()
            res = fut.result(timeout=5.0)

        assert res is not None
        assert entry.active_operations == 0
        assert registry.sweep(now=entry.last_access + 1_000_000.0) == ["slow"]
    finally:
        registry._ttl = original_ttl


def test_resolve_selection_releases_lease_on_every_path(_fresh_registry):
    """G4: every return path releases the acquired lease exactly once."""
    fig = Figure().scatter(np.arange(6.0), np.arange(6.0))
    token = registry.register(fig)
    entry = registry.get(token)
    assert entry is not None

    # 1. unknown kind (else-branch, returns before the kernel call)
    assert reflex_xy.resolve_selection(_event(token, {"kind": "sphere"})) is None
    assert entry.active_operations == 0
    # 2. malformed data_bounds (missing x0) -> KeyError swallowed
    assert (
        reflex_xy.resolve_selection(
            _event(token, {"kind": "box", "data_bounds": {"x1": 1, "y0": 0, "y1": 1}})
        )
        is None
    )
    assert entry.active_operations == 0
    # 3. malformed polygon (non-numeric point) -> ValueError swallowed
    assert (
        reflex_xy.resolve_selection(
            _event(token, {"kind": "lasso", "polygon": [["nope", 0], [1, 0], [0, 1]]})
        )
        is None
    )
    assert entry.active_operations == 0
    # 4. empty selection (box outside all points)
    assert (
        reflex_xy.resolve_selection(
            _event(
                token,
                {"kind": "box", "data_bounds": {"x0": 100, "x1": 200, "y0": 100, "y1": 200}},
            )
        )
        is None
    )
    assert entry.active_operations == 0
    # 5. valid box
    res = reflex_xy.resolve_selection(
        _event(token, {"kind": "box", "data_bounds": {"x0": 1, "x1": 3, "y0": 0, "y1": 4}})
    )
    assert res is not None
    assert entry.active_operations == 0

    # No double-release possible: a bare release now must error.
    with pytest.raises(RuntimeError, match="released without acquire"):
        registry._release_operation(entry)


def test_resolve_selection_never_raises_on_client_input(_fresh_registry):
    """G6: no client input raises out of ``resolve_selection``."""
    fig = Figure().scatter(np.arange(6.0), np.arange(6.0))
    token = registry.register(fig)
    cases = [
        None,
        {},
        {"type": "other", "token": token, "selection": {"kind": "box"}},
        {"type": "select_end", "token": 123, "selection": {"kind": "box"}},
        {"type": "select_end", "token": token, "selection": "not-a-dict"},
        {"type": "select_end", "token": token, "selection": {"kind": "clear", "cleared": True}},
        _event("missing", {"kind": "box"}),
        _event(token, {"kind": "sphere"}),
        _event(token, {"kind": "box", "data_bounds": {"x1": 1, "y0": 0, "y1": 1}}),
        _event(token, {"kind": "lasso", "polygon": "a-string"}),
        _event(token, {"kind": "box", "data_bounds": {"x0": None, "x1": 1, "y0": 0, "y1": 1}}),
        {"type": "select_end", "token": token, "selection": {}},
    ]
    for case in cases:
        assert reflex_xy.resolve_selection(case) is None
