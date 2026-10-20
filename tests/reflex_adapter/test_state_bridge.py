"""Rebuild-from-state: the multi-worker / reconnect recovery path."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import numpy as np
import pytest
import reflex as rx
from reflex.istate.manager.memory import StateManagerMemory

import reflex_xy
import xy
from reflex_xy.plan import build_plan
from reflex_xy.state_bridge import make_rebuild_hook, rebuild_plan_figure
from reflex_xy.tokens import (
    build_data_token,
    build_plan_token,
    build_state_token,
    parse_plan_token,
    parse_token,
)


class BridgeDemo(rx.State):
    points: int = 12

    @reflex_xy.figure
    def chart(self) -> xy.Chart:
        xs = np.linspace(0.0, 1.0, self.points)
        return xy.scatter_chart(xy.scatter(xs, xs), width=400, height=300)


def make_app_stub():
    # 0.9.6 memory manager needs no root class up front: the BaseStateToken
    # passed to get_state/modify_state carries it.
    return SimpleNamespace(state_manager=StateManagerMemory())


def test_rebuild_default_state(_fresh_registry, client_token):
    """A fresh node with no prior events still serves the figure (defaults)."""
    app = make_app_stub()
    token = build_state_token(client_token, BridgeDemo.get_full_name(), "chart")
    hook = make_rebuild_hook(app)
    figure = asyncio.run(hook(token))
    assert figure is not None
    assert figure.traces[0].n_points == 12


def test_rebuild_reads_session_state(_fresh_registry, client_token):
    """State mutated by earlier events drives the rebuilt figure."""
    app = make_app_stub()
    token_obj = rx.BaseStateToken(ident=client_token, cls=rx.State)

    async def main():
        async with app.state_manager.modify_state(token_obj) as root:
            sub = await root.get_state(BridgeDemo)
            sub.points = 77
        hook = make_rebuild_hook(app)
        return await hook(build_state_token(client_token, BridgeDemo.get_full_name(), "chart"))

    figure = asyncio.run(main())
    assert figure is not None
    assert figure.traces[0].n_points == 77


@pytest.mark.parametrize(
    "token",
    [
        "not-a-state-token",
        # valid grammar, unknown state
        "xyv1|11111111-2222-4333-8444-555566667777|no.such_state|chart",
    ],
)
def test_rebuild_unknown_fails_closed(_fresh_registry, token):
    hook = make_rebuild_hook(make_app_stub())
    assert asyncio.run(hook(token)) is None


def test_rebuild_var_without_builder_fails_closed(_fresh_registry, client_token):
    """A plain @rx.var of the same name is not a figure recipe."""

    class NotAFigure(rx.State):
        @rx.var
        def chart(self) -> str:
            return "hello"

    token = build_state_token(client_token, NotAFigure.get_full_name(), "chart")
    hook = make_rebuild_hook(make_app_stub())
    assert asyncio.run(hook(token)) is None


def test_rebuild_builder_returning_non_figure_fails_closed(_fresh_registry, client_token):
    class NotActuallyAFigure(rx.State):
        @reflex_xy.figure
        def chart(self) -> object:
            return object()

    token = build_state_token(client_token, NotActuallyAFigure.get_full_name(), "chart")
    hook = make_rebuild_hook(make_app_stub())
    assert asyncio.run(hook(token)) is None


def test_token_full_name_resolves_class(client_token):
    token = build_state_token(client_token, BridgeDemo.get_full_name(), "chart")
    parsed = parse_token(token)
    cls = rx.State.get_class_substate(tuple(parsed.state_full_name.split(".")))
    assert cls is BridgeDemo


# -- cache-miss rebuild vs. concurrent republish ordering ----------------

# Per-test asyncio gates shared with the rebuilding data method. The method
# reads state (its pre-mutation snapshot of n), then parks at the gate
# exactly once so the test can interleave a concurrent republish, then
# releases it to finish the stale rebuild. A second invocation (the
# republish path) skips the gate.
_gate_holder: list = [None]
_parked_holder: list = [None]
_park_used: list = [False]


class _RebuildRaceDemo(rx.State):
    n: int = 12

    @reflex_xy.data
    async def cloud(self) -> dict:
        n = self.n  # snapshot the pre-await state value
        if not _park_used[0]:
            _park_used[0] = True
            _parked_holder[0].set()
            await _gate_holder[0].wait()
        xs = np.linspace(0.0, 1.0, n)
        return {"x": xs, "y": xs}


def test_rebuild_plan_figure_discards_stale_columns_when_a_fresher_republish_lands(
    _fresh_registry, client_token
):
    """The registry's "Republish ordering" invariant must hold for the
    composite rebuild's column cache-fill, not just the republish path.

    A cache-miss rebuild reads state, awaits a slow data source, and a
    concurrent republish lands a fresher column generation during the await.
    The rebuild's stale columns must be discarded (not stamped over the
    fresher generation at a newer version), and the rebuild must bind the
    fresher columns so subscribers see current state, not the stale read."""
    _gate_holder[0] = asyncio.Event()
    _parked_holder[0] = asyncio.Event()
    _park_used[0] = False

    app = make_app_stub()
    token_obj = rx.BaseStateToken(ident=client_token, cls=rx.State)

    async def seed():
        async with app.state_manager.modify_state(token_obj) as root:
            sub = await root.get_state(_RebuildRaceDemo)
            sub.n = 12

    asyncio.run(seed())

    data_token = build_data_token(client_token, _RebuildRaceDemo.get_full_name(), "cloud")
    plan = build_plan("scatter_chart", (xy.scatter("x", "y"),), {})
    composite = parse_plan_token(build_plan_token(plan.digest, data_token))

    async def main():
        # Column cache is empty (cache miss) and the composite figure entry
        # is absent, so rebuild_plan_figure awaits rebuild_data, which parks
        # at the gate after reading n=12.
        rebuild_task = asyncio.create_task(rebuild_plan_figure(app, composite))
        await asyncio.wait_for(_parked_holder[0].wait(), timeout=2.0)

        # While the rebuild is parked, a normal dependency-driven republish
        # (state mutation -> re-evaluation -> data_vars._publish ->
        # registry.publish_columns) lands fresher 48-point columns.
        fresh = {"x": np.linspace(0.0, 1.0, 48), "y": np.linspace(0.0, 1.0, 48)}
        _fresh_registry.publish_columns(data_token, fresh)

        # Release the rebuild. Its stale 12-point read attempts the
        # cache-fill publish; the guard must reject it.
        _gate_holder[0].set()
        figure = await rebuild_task

        # The fresher generation is still current — the stale rebuild did
        # NOT overwrite it at a newer version.
        entry = _fresh_registry.get_columns(data_token)
        assert entry is not None
        assert len(entry.columns["x"]) == 48
        assert entry.version == 1

        # The rebuild returns the fresher-bound figure, not its stale read.
        assert figure is not None
        assert figure.traces[0].n_points == 48
        return figure

    asyncio.run(main())


def test_rebuild_plan_figure_serves_rebuilt_columns_when_no_republish_intervenes(
    _fresh_registry, client_token
):
    """Happy path: no concurrent republish lands, so the cache-miss rebuild's
    columns are inserted and bound — the guard must not regress the single-
    threaded cache-fill."""
    _gate_holder[0] = asyncio.Event()
    _gate_holder[0].set()  # no parking
    _parked_holder[0] = asyncio.Event()
    _park_used[0] = False

    app = make_app_stub()
    token_obj = rx.BaseStateToken(ident=client_token, cls=rx.State)

    async def seed():
        async with app.state_manager.modify_state(token_obj) as root:
            sub = await root.get_state(_RebuildRaceDemo)
            sub.n = 24

    asyncio.run(seed())

    data_token = build_data_token(client_token, _RebuildRaceDemo.get_full_name(), "cloud")
    plan = build_plan("scatter_chart", (xy.scatter("x", "y"),), {})
    composite = parse_plan_token(build_plan_token(plan.digest, data_token))

    async def main():
        figure = await rebuild_plan_figure(app, composite)
        assert figure is not None
        assert figure.traces[0].n_points == 24
        entry = _fresh_registry.get_columns(data_token)
        assert entry is not None
        assert entry.version == 1
        assert len(entry.columns["x"]) == 24
        # The cache-fill did not fan out to dependents (no composite entry
        # existed to re-bind), but the column cache is populated for future
        # binds.
        return figure

    asyncio.run(main())


def test_rebuild_plan_figure_uses_cached_columns_on_a_cache_hit(_fresh_registry, client_token):
    """The cache-hit path still serves the existing generation without
    rebuilding, and the guard is not involved."""
    app = make_app_stub()
    token_obj = rx.BaseStateToken(ident=client_token, cls=rx.State)

    async def seed():
        async with app.state_manager.modify_state(token_obj) as root:
            sub = await root.get_state(_RebuildRaceDemo)
            sub.n = 16

    asyncio.run(seed())

    data_token = build_data_token(client_token, _RebuildRaceDemo.get_full_name(), "cloud")
    plan = build_plan("scatter_chart", (xy.scatter("x", "y"),), {})
    composite = parse_plan_token(build_plan_token(plan.digest, data_token))
    cached = _fresh_registry.publish_columns(
        data_token, {"x": np.linspace(0.0, 1.0, 16), "y": np.linspace(0.0, 1.0, 16)}
    )

    async def main():
        figure = await rebuild_plan_figure(app, composite)
        assert figure is not None
        assert figure.traces[0].n_points == 16
        # Cache hit: the existing generation is unchanged.
        assert _fresh_registry.get_columns(data_token) is cached

    asyncio.run(main())
