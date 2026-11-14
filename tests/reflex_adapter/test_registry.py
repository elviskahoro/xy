from __future__ import annotations

import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

import xy
from reflex_xy.registry import FigureRegistry
from reflex_xy.tokens import build_data_token, build_plan_token


def make_figure(n: int = 16):
    xs = np.linspace(0.0, 1.0, n)
    return xy.scatter_chart(xy.scatter(xs, xs * 2.0), width=400, height=300).figure()


def test_register_release_roundtrip(_fresh_registry):
    registry = _fresh_registry
    token = registry.register(make_figure())
    assert token.startswith("xyfig-")
    assert registry.get(token) is not None
    registry.release(token)
    assert registry.get(token) is None
    registry.release(token)  # idempotent


def test_figure_only_helpers_reject_data_handles_and_junk(_fresh_registry):
    """A DataHandle names columns, never a figure: figure-only operations
    fail immediately instead of resolving to a room that can never exist."""
    import reflex_xy
    from reflex_xy.handles import DataHandle

    data = DataHandle("xyd1|client-token-1234|app.app.State|cloud")
    with pytest.raises(TypeError, match="DataHandle"):
        reflex_xy.release(data)
    with pytest.raises(TypeError, match="DataHandle"):
        reflex_xy.append(data, [0.0], [0.0])
    with pytest.raises(TypeError, match="int"):
        reflex_xy.release(123)  # never silently registry.release("")


def test_release_preserves_version_while_rebuildable_subscriber_remains(
    _fresh_registry,
):
    registry = _fresh_registry
    registry.subscribe("tok", "sid-1", rebuildable=True)
    first = registry.publish("tok", make_figure(), broadcast=False)
    assert registry.bump("tok", expected=first).version == 2

    registry.release("tok")
    assert registry.get("tok") is None
    assert registry._evicted_versions == {"tok": 2}

    republished = registry.publish("tok", make_figure(32), broadcast=False)
    assert republished.version == 3


def test_publish_versioning(_fresh_registry):
    registry = _fresh_registry
    fig1 = make_figure()
    entry = registry.publish("tok", fig1, broadcast=False)
    assert entry.version == 1
    # same object republished: no version bump
    assert registry.publish("tok", fig1, broadcast=False).version == 1
    # new figure object: bump
    assert registry.publish("tok", make_figure(32), broadcast=False).version == 2


def test_publish_if_missing_preserves_a_concurrent_current_generation(
    _fresh_registry,
):
    registry = _fresh_registry
    missing, guard = registry.begin_rebuild("tok")
    assert missing is None
    assert guard is not None
    current_figure = make_figure(8)
    current = registry.publish("tok", current_figure, broadcast=False)

    entry, inserted = registry.publish_if_missing("tok", make_figure(32), guard=guard)
    registry.finish_rebuild("tok", guard)

    assert not inserted
    assert entry is current
    assert entry.figure is current_figure
    assert entry.version == 1
    assert registry.is_current("tok", entry)
    assert registry._active_rebuild_guards == {}


def test_release_invalidates_an_active_rebuild_without_unbounded_revision_state(
    _fresh_registry,
):
    registry = _fresh_registry
    registry.subscribe("tok", "sid-1", rebuildable=True)
    first = registry.publish("tok", make_figure(), broadcast=False)
    assert registry.bump("tok", expected=first).version == 2
    registry.release("tok")

    missing, guard = registry.begin_rebuild("tok")
    assert missing is None
    assert guard is not None
    registry.release("tok")  # a newer canonical absence while the builder awaits

    entry, inserted = registry.publish_if_missing("tok", make_figure(32), guard=guard)
    registry.finish_rebuild("tok", guard)

    assert entry is None
    assert not inserted
    assert registry.get("tok") is None
    assert registry._evicted_versions == {"tok": 2}
    assert registry._active_rebuild_guards == {}
    assert registry.publish("tok", make_figure(32), broadcast=False).version == 3


def test_failed_rebuild_cleanup_is_identity_and_guard_checked(
    _fresh_registry,
):
    registry = _fresh_registry
    registry.subscribe("tok", "sid-1", rebuildable=True)
    missing, guard = registry.begin_rebuild("tok")
    assert missing is None
    assert guard is not None
    inserted, did_insert = registry.publish_if_missing("tok", make_figure(), guard=guard)
    assert inserted is not None
    assert did_insert
    replacement = registry.publish("tok", make_figure(32), broadcast=False)

    assert not registry.remove_if_current("tok", inserted, guard=guard)
    registry.finish_rebuild("tok", guard)
    assert registry.is_current("tok", replacement)
    assert registry._evicted_versions == {}


def test_same_object_publish_invalidates_failed_rebuild_cleanup(_fresh_registry):
    registry = _fresh_registry
    figure = make_figure()
    missing, guard = registry.begin_rebuild("tok")
    assert missing is None
    assert guard is not None
    inserted, did_insert = registry.publish_if_missing("tok", figure, guard=guard)
    assert inserted is not None
    assert did_insert
    assert registry.get_with_rebuild_guard("tok") == (inserted, True)

    authoritative = registry.publish("tok", figure, broadcast=False)

    assert authoritative is inserted
    assert registry.get_with_rebuild_guard("tok") == (inserted, False)
    assert not registry.remove_if_current("tok", inserted, guard=guard)
    registry.finish_rebuild("tok", guard)
    assert registry.is_current("tok", inserted)


def test_same_object_publish_reauthorization_schedules_without_version_bump(_fresh_registry):
    registry = _fresh_registry
    figure = make_figure()
    missing, guard = registry.begin_rebuild("tok")
    assert missing is None
    assert guard is not None
    inserted, did_insert = registry.publish_if_missing("tok", figure, guard=guard)
    assert inserted is not None
    assert did_insert
    seen = []

    async def on_publish(token, entry):
        seen.append((token, entry))

    async def main():
        registry.attach_loop(asyncio.get_running_loop())
        registry.on_publish(on_publish)

        authoritative = registry.publish("tok", figure)
        await asyncio.sleep(0)

        assert authoritative is inserted
        assert authoritative.version == 1
        assert seen == [("tok", inserted)]

    asyncio.run(main())
    registry.finish_rebuild("tok", guard)
    assert registry._active_rebuild_guards == {}


def test_failed_rebuild_cleanup_without_subscriber_drops_version(_fresh_registry):
    registry = _fresh_registry
    missing, guard = registry.begin_rebuild("tok")
    assert missing is None
    assert guard is not None
    inserted, did_insert = registry.publish_if_missing("tok", make_figure(), guard=guard)
    assert inserted is not None
    assert did_insert

    assert registry.remove_if_current("tok", inserted, guard=guard)
    registry.finish_rebuild("tok", guard)
    assert registry._evicted_versions == {}
    assert registry.publish("tok", make_figure(32), broadcast=False).version == 1


def test_publish_replacement_creates_a_consistent_generation(_fresh_registry):
    registry = _fresh_registry
    first_figure = make_figure(8)
    first = registry.publish("tok", first_figure, broadcast=False)
    second = registry.publish("tok", make_figure(32), broadcast=False)

    assert second is not first
    assert first.figure is first_figure
    assert first.version == 1
    assert second.version == 2
    assert registry.is_current("tok", second)
    assert not registry.is_current("tok", first)
    assert registry.bump("tok", expected=first) is None
    assert second.version == 2


def test_bump_records_in_place_mutation(_fresh_registry):
    registry = _fresh_registry
    token = registry.register(make_figure())
    assert registry.bump(token).version == 2
    assert registry.bump("missing") is None


def test_ttl_sweep(_fresh_registry):
    registry = FigureRegistry(ttl_seconds=0.0)
    token = registry.register(make_figure())
    dropped = registry.sweep(now=registry.get(token).last_access + 1.0)
    assert dropped == [token]
    assert registry.get(token) is None


def test_subscribed_rebuild_after_ttl_sweep_keeps_wire_version_monotonic(
    _fresh_registry,
):
    registry = FigureRegistry(ttl_seconds=0.0)
    registry.subscribe("tok", "sid-1", rebuildable=True)
    first = registry.publish("tok", make_figure(), broadcast=False)
    assert registry.bump("tok", expected=first).version == 2

    assert registry.sweep(now=first.last_access + 1.0) == ["tok"]
    rebuilt = registry.publish("tok", make_figure(), broadcast=False)

    assert rebuilt.version == 3


def test_opaque_sweeps_do_not_retain_version_tombstones(_fresh_registry):
    registry = FigureRegistry(ttl_seconds=0.0)
    registry.subscribe("opaque", "sid-1", rebuildable=False)
    first = registry.publish("opaque", make_figure(), broadcast=False)
    assert registry.bump("opaque", expected=first).version == 2

    assert registry.sweep(now=first.last_access + 1.0) == ["opaque"]
    assert "opaque" not in registry._evicted_versions

    # An opaque token has no automatic rebuild contract. If application code
    # explicitly reuses its string, it starts a fresh wire generation.
    assert registry.publish("opaque", make_figure(), broadcast=False).version == 1


def test_sweep_keeps_recently_touched(_fresh_registry):
    registry = FigureRegistry(ttl_seconds=1000.0)
    token = registry.register(make_figure())
    assert registry.sweep() == []
    assert registry.get(token) is not None


def test_broadcast_scheduling_from_loop_and_thread(_fresh_registry):
    registry = _fresh_registry
    token = registry.register(make_figure())
    seen: list[tuple[str, int]] = []

    async def on_publish(tok, entry):
        seen.append((tok, entry.version))

    async def main():
        registry.attach_loop(asyncio.get_running_loop())
        registry.on_publish(on_publish)
        # same-loop publish
        registry.publish(token, make_figure(8))
        await asyncio.sleep(0.05)
        # cross-thread publish (sync reflex handlers run in a thread pool)
        await asyncio.to_thread(registry.publish, token, make_figure(4))
        await asyncio.sleep(0.05)

    asyncio.run(main())
    assert seen == [(token, 2), (token, 3)]


def test_rapid_publishes_coalesce(_fresh_registry):
    registry = _fresh_registry
    token = registry.register(make_figure())
    seen: list[int] = []

    async def on_publish(tok, entry):
        seen.append(entry.version)

    async def main():
        registry.attach_loop(asyncio.get_running_loop())
        registry.on_publish(on_publish)
        # Two publishes before the loop can run the first broadcast: one
        # fan-out, carrying the latest state — never a stale intermediate.
        registry.publish(token, make_figure(8))
        registry.publish(token, make_figure(4))
        await asyncio.sleep(0.05)

    asyncio.run(main())
    assert seen == [3]


def test_broadcast_noop_before_setup(_fresh_registry):
    registry = _fresh_registry
    token = registry.register(make_figure())
    # No loop attached: must not raise, must not queue anything.
    registry.publish(token, make_figure(8))
    assert registry.get(token).version == 2


def test_figure_accepts_chart_or_figure(_fresh_registry):
    registry = _fresh_registry
    xs = np.linspace(0.0, 1.0, 8)
    chart = xy.scatter_chart(xy.scatter(xs, xs), width=300, height=200)
    token_from_chart = registry.register(chart.figure())
    assert registry.get(token_from_chart) is not None

    import reflex_xy

    handle = reflex_xy.register(chart)  # public API accepts the composed Chart
    assert reflex_xy.registry.get(handle.token) is not None


def test_entry_lock_serializes(_fresh_registry):
    registry = _fresh_registry
    token = registry.register(make_figure())
    entry = registry.get(token)
    order: list[int] = []

    async def user(i: int):
        async with entry.lock:
            order.append(i)
            await asyncio.sleep(0.01)
            order.append(i)

    async def main():
        await asyncio.gather(user(1), user(2))

    asyncio.run(main())
    assert order in ([1, 1, 2, 2], [2, 2, 1, 1])


def test_unwired_slow_append_does_not_hold_the_registry_mutex(_fresh_registry):
    registry = _fresh_registry
    started = threading.Event()
    resume = threading.Event()

    class BlockingFigure:
        def append(self, *args, **kwargs):
            started.set()
            assert resume.wait(2.0)
            return {"type": "append"}, []

    registry.publish("slow", BlockingFigure(), broadcast=False)
    replacement = make_figure()

    with ThreadPoolExecutor(max_workers=2) as pool:
        append_future = pool.submit(registry.append, "slow", [1.0], [2.0])
        assert started.wait(1.0)
        try:
            replaced = pool.submit(registry.publish, "slow", replacement, broadcast=False).result(
                timeout=0.5
            )
        finally:
            resume.set()
        append_future.result(timeout=1.0)

    current = registry.get("slow")
    assert current is replaced
    assert current.figure is replacement
    assert current.version == 2


def test_forced_sweep_skips_a_generation_with_a_blocking_headless_append(
    _fresh_registry,
):
    registry = FigureRegistry(ttl_seconds=0.0)
    started = threading.Event()
    resume = threading.Event()

    class BlockingFigure:
        def append(self, *args, **kwargs):
            started.set()
            assert resume.wait(2.0)
            return {"type": "append"}, []

    entry = registry.publish("slow", BlockingFigure(), broadcast=False)

    with ThreadPoolExecutor(max_workers=1) as pool:
        append_future = pool.submit(registry.append, "slow", [1.0], [2.0])
        assert started.wait(1.0)
        try:
            forced_now = entry.last_access + 1_000_000.0
            assert registry.sweep(now=forced_now) == []
            assert registry.is_current("slow", entry)
        finally:
            resume.set()
        append_future.result(timeout=1.0)

    assert entry.version == 2
    assert entry.active_operations == 0
    assert registry.sweep(now=entry.last_access + 1_000_000.0) == ["slow"]


def test_forced_sweep_skips_a_generation_with_a_blocking_wired_append(
    _fresh_registry,
):
    registry = FigureRegistry(ttl_seconds=0.0)
    started = threading.Event()
    resume = threading.Event()

    class BlockingFigure:
        def append(self, *args, **kwargs):
            started.set()
            assert resume.wait(2.0)
            return {"type": "append"}, []

    entry = registry.publish("slow", BlockingFigure(), broadcast=False)

    async def main():
        registry.attach_loop(asyncio.get_running_loop())
        tasks_before = asyncio.all_tasks()
        registry.append("slow", [1.0], [2.0])
        append_tasks = asyncio.all_tasks() - tasks_before
        assert len(append_tasks) == 1
        append_task = append_tasks.pop()
        assert await asyncio.to_thread(started.wait, 1.0)
        try:
            forced_now = entry.last_access + 1_000_000.0
            assert registry.sweep(now=forced_now) == []
            assert registry.is_current("slow", entry)
        finally:
            resume.set()
        await append_task

    asyncio.run(main())
    assert entry.version == 2
    assert entry.active_operations == 0
    assert registry.sweep(now=entry.last_access + 1_000_000.0) == ["slow"]


def test_unsubscribe_cleans_tombstone_only_after_last_subscriber(_fresh_registry):
    registry = FigureRegistry(ttl_seconds=0.0)
    registry.subscribe("tok", "sid-1", rebuildable=True)
    registry.subscribe("tok", "sid-1", rebuildable=True)  # idempotent
    registry.subscribe("tok", "sid-2", rebuildable=True)
    entry = registry.publish("tok", make_figure(), broadcast=False)
    assert registry.bump("tok", expected=entry).version == 2
    assert registry.sweep(now=entry.last_access + 1.0) == ["tok"]

    registry.unsubscribe("tok", "sid-1")
    registry.unsubscribe("tok", "sid-1")  # idempotent
    assert registry._evicted_versions == {"tok": 2}
    assert registry._rebuildable_subscribers == {"tok": {"sid-2"}}
    assert "sid-1" not in registry._rebuildable_tokens_by_sid

    registry.unsubscribe("tok", "sid-2")
    assert registry._evicted_versions == {}
    assert registry._rebuildable_subscribers == {}
    assert registry._rebuildable_tokens_by_sid == {}


def test_disconnect_cleans_all_tokens_and_is_idempotent(_fresh_registry):
    registry = FigureRegistry(ttl_seconds=0.0)
    registry.subscribe("shared", "sid-1", rebuildable=True)
    registry.subscribe("shared", "sid-2", rebuildable=True)
    registry.subscribe("only-sid-1", "sid-1", rebuildable=True)
    registry.subscribe("only-sid-1", "sid-1", rebuildable=True)  # idempotent

    for token in ("shared", "only-sid-1"):
        entry = registry.publish(token, make_figure(), broadcast=False)
        assert registry.sweep(now=entry.last_access + 1.0) == [token]

    registry.disconnect("sid-1")
    registry.disconnect("sid-1")  # idempotent
    assert registry._evicted_versions == {"shared": 1}
    assert registry._rebuildable_subscribers == {"shared": {"sid-2"}}
    assert registry._rebuildable_tokens_by_sid == {"sid-2": {"shared"}}

    registry.disconnect("sid-2")
    assert registry._evicted_versions == {}
    assert registry._rebuildable_subscribers == {}
    assert registry._rebuildable_tokens_by_sid == {}


@pytest.mark.parametrize("n", [1, 3])
def test_len_and_tokens(_fresh_registry, n):
    registry = _fresh_registry
    tokens = {registry.register(make_figure()) for _ in range(n)}
    assert len(registry) == n
    assert set(registry.tokens()) == tokens


# -- data-bound tier: atomic plan binding on rebuild-attempt -----------------


_CLIENT = "11111111-2222-4333-8444-555566667777"


def _plan_composite():
    """A composite plan token + its (data_token, digest) identity."""
    data_token = build_data_token(_CLIENT, "app.Demo", "cloud")
    digest = "deadbeef" * 4
    return build_plan_token(digest, data_token), data_token, digest


def test_publish_if_missing_without_plan_bound_does_not_index_digest(_fresh_registry):
    """Backward compatibility: the new ``plan_bound`` kwarg is opt-in. Plain
    ``publish_if_missing`` (state-token rebuilds, tests) must NOT touch the
    data-token → {digest} index."""
    registry = _fresh_registry
    composite, _data_token, _digest = _plan_composite()
    missing, guard = registry.begin_rebuild(composite)
    assert missing is None and guard is not None

    entry, inserted = registry.publish_if_missing(composite, make_figure(8), guard=guard)
    registry.finish_rebuild(composite, guard)
    assert inserted
    assert entry is not None
    # No plan_bound -> the composite's digest is NOT in the fan-out index.
    assert registry._digests_by_data_token == {}


def test_publish_if_missing_binds_plan_atomically_with_insert(_fresh_registry):
    """The cold-subscribe race fix: a composite plan's digest is inserted into
    ``_digests_by_data_token`` UNDER THE SAME mutex hold as the entry's
    insertion, so any ``publish_columns`` that observes the new entry also
    observes its digest and rebuilds this plan. Without atomicity a republish
    landing between the insert and the (now-removed) ``bind_plan`` after the
    rebuild would snapshot an empty set and silently never fan out."""
    registry = _fresh_registry
    composite, data_token, digest = _plan_composite()
    missing, guard = registry.begin_rebuild(composite)
    assert missing is None and guard is not None

    # Observe the index in the same critical window a concurrent
    # publish_columns would: it must be empty BEFORE the insert, present
    # atomically WITH the entry. We cannot interleave pure-python under the
    # mutex, so the contract is: after publish_if_missing returns, BOTH
    # the entry and the index are visible together (never one without the
    # other).
    assert registry._digests_by_data_token.get(data_token) is None
    assert registry.get(composite) is None

    entry, inserted = registry.publish_if_missing(
        composite, make_figure(8), guard=guard, plan_bound=(data_token, digest)
    )
    registry.finish_rebuild(composite, guard)

    assert inserted
    assert entry is not None
    assert registry.get(composite) is entry
    # The digest landed in the index together with the entry — no window
    # where the entry is observable but the digest is not.
    assert registry._digests_by_data_token.get(data_token) == {digest}


def test_publish_if_missing_plan_bound_pre_existing_entry_does_not_re_bind(
    _fresh_registry,
):
    """When ``publish_if_missing`` returns a pre-existing entry (a concurrent
    publisher won while the rebuild awaited user code), it MUST NOT touch the
    digest index: that publisher's own path owns the index invariant. Only a
    generation this call inserts is bound here."""
    registry = _fresh_registry
    composite, data_token, digest = _plan_composite()
    other_digest = "cafe" * 8

    # Step 1: cold miss -> guard the rebuild attempt.
    missing, guard = registry.begin_rebuild(composite)
    assert missing is None and guard is not None
    # An unrelated plan was already bound for this data token.
    registry.bind_plan(data_token, other_digest)
    assert registry._digests_by_data_token.get(data_token) == {other_digest}

    # Step 2: a canonical publisher wins the race while the rebuild awaited
    # user code (this is the _run_rebuild_attempt path where rebuild() returns
    # a figure but a sibling publish_columns triggered _rebuild_dependent and
    # already published the composite).
    pre_existing = registry.publish(composite, make_figure(8), broadcast=False)

    # Step 3: the rebuild's publish_if_missing now finds the entry present.
    entry, inserted = registry.publish_if_missing(
        composite, make_figure(32), guard=guard, plan_bound=(data_token, digest)
    )
    registry.finish_rebuild(composite, guard)

    assert not inserted
    assert entry is pre_existing
    # The pre-existing-entry path leaves the index exactly as the canonical
    # publisher left it: ``digest`` is NOT silently added by this rebuild.
    assert registry._digests_by_data_token.get(data_token) == {other_digest}


def test_publish_if_missing_plan_bound_invalid_guard_does_not_index(_fresh_registry):
    """An invalidated guard (a canonical release/publish won while the rebuild
    awaited user code) returns ``(None, False)`` and MUST NOT index a digest
    for an entry that was never inserted — otherwise the digest would dangle
    forever on a token with no figure."""
    registry = _fresh_registry
    composite, data_token, digest = _plan_composite()
    missing, guard = registry.begin_rebuild(composite)
    assert missing is None and guard is not None

    # A canonical release invalidates the guard and leaves the token absent.
    registry.release(composite)

    entry, inserted = registry.publish_if_missing(
        composite, make_figure(8), guard=guard, plan_bound=(data_token, digest)
    )
    registry.finish_rebuild(composite, guard)

    assert entry is None
    assert not inserted
    assert registry.get(composite) is None
    assert registry._digests_by_data_token.get(data_token) is None


def test_publish_if_missing_plan_bound_reaped_by_failed_broadcast_cleanup(
    _fresh_registry,
):
    """A binding added by an attempt that ultimately fails (broadcast_payload
    raises -> remove_if_current) is reaped by ``_unbind_plan_if_unmounted_locked``
    exactly like any other unmount transition. With no subscriber yet (subscribe
    runs after the attempt), the binding must vanish with the entry."""
    registry = _fresh_registry
    composite, data_token, digest = _plan_composite()
    missing, guard = registry.begin_rebuild(composite)
    assert missing is None and guard is not None

    entry, inserted = registry.publish_if_missing(
        composite, make_figure(8), guard=guard, plan_bound=(data_token, digest)
    )
    assert inserted

    # Simulate the namespace's failed-broadcast cleanup path: no subscribers
    # were registered yet (subscribe runs after the attempt in on_sub), so
    # remove_if_current unbinds the plan atomically with removing the entry.
    assert registry.remove_if_current(composite, entry, guard=guard)
    registry.finish_rebuild(composite, guard)

    assert registry.get(composite) is None
    assert registry._digests_by_data_token.get(data_token) is None
    assert registry._evicted_versions == {}


def test_publish_columns_observes_plan_bound_from_concurrent_rebuild_attempt(
    _fresh_registry,
):
    """End-to-end at the registry layer: a publish_columns that runs AFTER a
    rebuild attempt's publish_if_missing (so the entry is observable) MUST
    also observe the digest in the index, and so must fan out via
    ``_rebuild_dependent``. This is the exact ordering the race used to break:
    without the atomic bind, publish_columns would snapshot an empty digest
    set and silently never rebuild this plan."""
    registry = _fresh_registry
    composite, data_token, digest = _plan_composite()
    # Prime v1 columns so the rebuild's snapshot is deterministic.
    v1_columns = {"x": np.linspace(0.0, 1.0, 24), "y": np.linspace(0.0, 1.0, 24) * 2.0}
    v2_columns = {"x": np.linspace(0.0, 1.0, 5), "y": np.linspace(0.0, 1.0, 5) * 2.0}
    registry.publish_columns(data_token, v1_columns)

    # A rebuild attempt pumps a figure from v1 and binds the digest
    # atomically with the insert.
    missing, guard = registry.begin_rebuild(composite)
    assert missing is None and guard is not None
    figure_v1 = xy.scatter_chart(xy.scatter(v1_columns["x"], v1_columns["y"])).figure()
    entry, inserted = registry.publish_if_missing(
        composite, figure_v1, guard=guard, plan_bound=(data_token, digest)
    )
    assert inserted
    registry.finish_rebuild(composite, guard)

    # A column republish lands here (the racing window). With the fix it
    # sees the now-bound digest and rebuilds the composite from v2; without
    # the fix the digest set was empty and the republish silently dropped.
    # We stub ``_rebuild_dependent`` to record the fan-out call without
    # running the plan bind (the unregistered digest would otherwise raise
    # PlanMissError); the dispatch observability is what we assert here.
    rebuilt_called = []
    real_rebuild_dependent = registry._rebuild_dependent

    def tracking_rebuild_dependent(data_token_, digest_, column_entry_):
        rebuilt_called.append((data_token_, digest_, column_entry_.version))
        # Don't run the real path: there's no registered plan for this digest.

    registry._rebuild_dependent = tracking_rebuild_dependent
    try:
        registry.publish_columns(data_token, v2_columns)
    finally:
        registry._rebuild_dependent = real_rebuild_dependent

    # The republish fanned out to this plan precisely once, from v2 columns.
    assert rebuilt_called == [(data_token, digest, 2)]
    # The v2 columns are now canonical. The composite entry remains the one
    # the rebuild attempt inserted (the stub didn't rebuild the figure);
    # the next publish_columns against a *registered* plan would replace it.
    assert registry.get_columns(data_token).version == 2
    assert registry.get(composite) is entry
    assert registry._digests_by_data_token.get(data_token) == {digest}
