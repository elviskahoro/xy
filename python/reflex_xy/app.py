"""Wiring the data plane into a Reflex app.

Two equivalent entry points, both one line for the user:

- ``rxconfig.py``: ``plugins=[reflex_xy.XYPlugin()]`` — the plugin's
  `post_compile` hook runs once at backend worker startup with the live
  App and calls `setup(app)`. Zero app-code changes.
- ``app.py``: ``reflex_xy.setup(app)`` right after ``app = rx.App()`` —
  the socket server already exists at that point.

`setup` is idempotent; using both costs nothing.

What setup does: registers the `/_xy` socket.io namespace on the app's
existing AsyncServer (same physical websocket as the app plane — see
namespace.py), wires publish fan-out, and adds a lifespan task that
registers this worker's chart plans (`_ensure_page_plans`, fail-closed),
captures the event loop (for thread-safe broadcasts from sync handlers),
and runs the registry TTL sweep.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import contextlib
import inspect
import warnings
from collections.abc import Coroutine
from typing import Any, Optional, cast

from reflex.plugins import Plugin

from .handles import FigureHandle, token_of
from .namespace import XYNamespace
from .registry import _figure_of, registry
from .state_bridge import make_rebuild_hook
from .tokens import BUILDER_ATTR, PROBE_ATTR
from .vars import AsyncFigureVar, FigureVar

__all__ = [
    "FigureProbeError",
    "XYPlugin",
    "append",
    "clear_selection",
    "probe_figure_builders",
    "reset_view",
    "select",
    "set_view",
    "setup",
]

_namespace: Optional[XYNamespace] = None


def setup(app: Any) -> XYNamespace:
    """Attach the xy data plane to a Reflex app (idempotent)."""
    global _namespace
    if _namespace is not None:
        return _namespace
    sio = getattr(app, "sio", None)
    if sio is None:
        msg = (
            "reflex_xy.setup(app) needs the app's socket server; it exists "
            "only when state is enabled (rx.App(enable_state=True), the default)."
        )
        raise RuntimeError(msg)
    namespace = XYNamespace(registry, rebuild=make_rebuild_hook(app))
    sio.register_namespace(namespace)
    wire(namespace)

    def _lifespan() -> Coroutine[Any, Any, None]:
        # Deliberately a *sync* function returning the sweep coroutine, not an
        # `async def`. Reflex starts a coroutine lifespan task with
        # `asyncio.create_task(task())` and then yields, so anything raised
        # inside an `async def` body surfaces in the background *after* the
        # worker is already serving — exactly the fail-open shape
        # `_ensure_page_plans` exists to prevent. Reflex calls `task()` inline
        # to *get* that coroutine, before create_task and before the lifespan
        # yields, so raising from this body aborts startup instead.
        _ensure_page_plans(app)
        return _xy_lifespan()

    app.register_lifespan_task(_lifespan)
    _namespace = namespace
    return namespace


def _ensure_page_plans(app: Any) -> None:
    """Evaluate the app's page component functions so chart plans register.

    The data-bound tier's plan map is process-local and populated by the
    chart factories *as page bodies run* (reflex-integration.md §3.6). A
    backend-only worker — dev backend subprocesses and prod workers alike —
    imports the app module but skips the frontend compile, so its pages sit
    unevaluated and every plan subscription would answer `err {resync}`
    forever. Running the page functions here makes "the plan map is
    populated in every worker" true by construction; the built component
    trees are discarded (plans and payload assets are content-addressed and
    idempotent).

    Failure is fail-closed: a page that cannot evaluate here leaves this
    worker with an incomplete plan map, and behind a load balancer that is
    the worst failure shape there is — charts blank or not depending on
    which worker answers, with only a startup warning to explain it. Every
    failing page is collected and the worker refuses to start, naming the
    pages; the same page code already fails `reflex run`'s real compile, so
    a healthy deployment never hits this. "Refuses to start" is load-bearing
    and depends on *where* this runs: `setup`'s lifespan calls it in the
    synchronous part of the task, before Reflex schedules the sweep
    coroutine, so the exception aborts lifespan startup rather than landing
    in a background task on an already-serving worker.
    """
    pages = getattr(app, "_unevaluated_pages", None) or {}
    failures: list[str] = []
    for route, page in dict(pages).items():
        component = getattr(page, "component", None)
        if not callable(component):
            continue  # already-built component instances registered at add_page
        try:
            component()
        except Exception as exc:  # noqa: BLE001 - user page code is an input boundary
            failures.append(f"{route!r}: {type(exc).__name__}: {exc}")
    if failures:
        msg = (
            "reflex_xy: evaluating page component functions for chart-plan "
            "registration failed on this worker; serving would leave its "
            "plan map incomplete (load-balancer-dependent blank charts), "
            "so startup is refused. Failing pages: " + "; ".join(sorted(failures))
        )
        raise RuntimeError(msg)


def wire(namespace: XYNamespace) -> None:
    """Point the registry's fan-out seams at a namespace (setup and tests)."""
    registry.on_publish(namespace.broadcast_payload)
    registry.on_push(namespace.broadcast_message)
    registry.on_error(namespace.broadcast_error)


async def _xy_lifespan() -> None:
    """Capture the serving loop, then sweep idle figures forever."""
    registry.attach_loop(asyncio.get_running_loop())
    with contextlib.suppress(asyncio.CancelledError):  # normal shutdown
        await registry.sweep_forever()


class FigureProbeError(Exception):
    """A `@reflex_xy.figure` builder failed its compile-time probe."""


def _builder_location(builder: Any) -> str:
    try:
        filename = inspect.getsourcefile(builder)
        _, line = inspect.getsourcelines(builder)
    except (OSError, TypeError):
        return "<source unavailable>"
    return f"{filename}:{line}"


def _touches_session(builder: Any) -> bool:
    """Heuristic escape valve (constraint 2): builders that read the session
    (`self.router`) cannot be expected to run against default state — their
    probe failures degrade to a warning instead of failing the compile."""
    try:
        source = inspect.getsource(builder)
    except (OSError, TypeError):
        return False
    return "self.router" in source


def _iter_probed_figure_vars(state_cls: Any, seen: set[Any]):
    if state_cls in seen:
        return
    seen.add(state_cls)
    computed = getattr(state_cls, "computed_vars", {})
    for name, var in computed.items():
        if isinstance(var, (FigureVar, AsyncFigureVar)):
            yield state_cls, name, var
    for subclass in state_cls.get_substates():
        yield from _iter_probed_figure_vars(subclass, seen)


class _ProbeLoopError(RuntimeError):
    """Internal sentinel: the probe could not even *set up* the worker loop
    for an opted-in async builder (e.g. ``asyncio.new_event_loop`` fails during
    interpreter shutdown, or in a runtime that restricts loop creation).
    Distinct from a builder failure so the surfaced
    :class:`FigureProbeError` does not advise ``probe=False`` for a builder
    the user explicitly opted into probing. Builder-raised exceptions
    (including ``RuntimeError``) are *not* wrapped in this — they propagate
    as ordinary builder failures."""


def _run_in_fresh_loop(builder: Any, substate: Any) -> Any:
    """Worker-thread body: create a private loop and drive the builder on it.

    Loop *creation* failure is wrapped in :class:`_ProbeLoopError` so the
    caller can surface it as a probe-infrastructure limitation rather than a
    builder defect; anything raised from inside the coroutine (the builder
    body) propagates unchanged as a builder failure.
    """
    try:
        loop = asyncio.new_event_loop()
    except RuntimeError as loop_exc:
        raise _ProbeLoopError(str(loop_exc)) from loop_exc
    try:
        return loop.run_until_complete(builder(substate))
    finally:
        loop.close()


def _run_async_builder(builder: Any, substate: Any) -> Any:
    """Run an opted-in async figure builder to completion for the probe.

    ``asyncio.run`` cannot be called from inside a running event loop, but
    ``post_compile`` (and therefore this probe) can fire from inside one:
    uvicorn invokes the Reflex ASGI factory (``App.__call__``) from within
    its server coroutine, so ``reflex run``,
    ``python -m uvicorn --factory`` (the Windows prod path), and any manual
    ``uvicorn ... --factory`` deployment reach this code with a loop already
    running.

    Driving the builder on a fresh, separate loop in the *same* thread does
    not work either: ``loop.run_until_complete(...)`` on a non-running loop
    still raises ``RuntimeError: Cannot run the event loop while another
    loop is running`` whenever another loop is running on this thread
    (Python 3.10+, enforced in 3.13). So when a foreign loop is already
    running we run the builder in a dedicated worker thread with its own
    fresh loop — the main thread's loop is untouched and never asked to
    schedule work the synchronous probe would then deadlock waiting on.
    Builder-raised exceptions propagate through ``future.result()``
    unchanged so the caller surfaces them as ordinary builder failures;
    only loop *creation* failure is reclassified as a probe-infrastructure
    limitation via :class:`_ProbeLoopError`.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(builder(substate))
    # A foreign loop is running on this thread (uvicorn's server loop,
    # typically). Run the builder in a worker thread so it gets its own
    # event loop and we never block on a future scheduled on the foreign
    # loop from the very thread that loop needs to keep running.
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(_run_in_fresh_loop, builder, substate).result()


def probe_figure_builders(root_cls: Any = None) -> list[str]:
    """Run every probe-enabled `@reflex_xy.figure` builder against default
    state (reflex-integration.md §3.1): the compile gate of the escape-hatch
    tier. Returns the probed var full names; raises :class:`FigureProbeError`
    (wrapping the original) on the first failing builder.

    Level ``"build"`` runs the builder body and type-checks its return —
    hallucinated `xy.*` names, wrong kwargs, eager chrome errors, and a
    return value that is not a chart (or ``None``) fail here. ``"figure"``
    also compiles the returned chart. ``False`` (and, by default, async
    builders) are skipped. Builders whose source touches ``self.router`` degrade
    failures to a `RuntimeWarning`: they are session-dependent by
    declaration and only a live session can validate them.
    """
    import reflex as rx

    root_cls = root_cls if root_cls is not None else cast("Any", rx.State)
    root = root_cls(_reflex_internal_init=True)
    probed: list[str] = []
    for state_cls, name, var in _iter_probed_figure_vars(root_cls, set()):
        fget = var._fget
        level = getattr(fget, PROBE_ATTR, False)
        builder = getattr(fget, BUILDER_ATTR, None)
        if not level or builder is None:
            continue
        full_name = f"{state_cls.get_full_name()}.{name}"
        try:
            substate = root.get_substate(tuple(state_cls.get_full_name().split("."))[1:])
        except (KeyError, ValueError):
            continue  # not reachable from this root (e.g. mixin scaffolding)
        try:
            if inspect.iscoroutinefunction(builder):
                chart = _run_async_builder(builder, substate)
            else:
                chart = builder(substate)
            if chart is not None and not (
                callable(getattr(chart, "figure", None))
                or callable(getattr(chart, "build_payload", None))
            ):
                # Every probe level checks the return *type*: a builder that
                # returns something no registry publish can accept would
                # otherwise reach hydrate before failing.
                msg = (
                    f"builder returned {type(chart).__name__}; expected an "
                    "xy Chart (or internal Figure), or None for 'no chart'"
                )
                raise TypeError(msg)
            if level == "figure" and chart is not None:
                _figure_of(chart)
        except Exception as exc:
            location = _builder_location(builder)
            if _touches_session(builder):
                warnings.warn(
                    f"@reflex_xy.figure probe: {full_name} ({location}) reads the "
                    f"session and failed against default state: {exc!r}. Probes "
                    "validate what they can; pass probe=False to silence.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                continue
            if isinstance(exc, _ProbeLoopError):
                # Not a builder defect: the probe could not even *set up* the
                # worker loop for this async builder (uvicorn invokes the
                # ASGI factory — and therefore this plugin's `post_compile` —
                # from inside its server loop, so the probe drives opted-in
                # async builders in a worker thread; loop *creation* can fail
                # during interpreter shutdown or in a runtime that restricts
                # loop creation). Advising `probe=False` would tell the user
                # to disable the feature they explicitly opted into; the fix
                # is in the probe, not the builder.
                msg = (
                    f"@reflex_xy.figure probe could not run async builder "
                    f"{full_name} ({location}) here: {exc!r}. The probe "
                    "executes opted-in async builders on a fresh, private "
                    "loop (uvicorn runs the ASGI factory — and therefore "
                    "this probe — inside its server loop), but the worker "
                    "loop could not be created. This is a probe "
                    "infrastructure limitation, not a builder failure; the "
                    "opt-in is correct, so do not opt out of the probe for "
                    "this builder."
                )
                raise FigureProbeError(msg) from exc
            msg = (
                f"@reflex_xy.figure probe failed for {full_name} ({location}): "
                f"{type(exc).__name__}: {exc}. The builder ran against default "
                "state at compile so this error would not wait for a browser "
                "session; pass @reflex_xy.figure(probe=False) if this builder "
                "cannot run outside a live session."
            )
            raise FigureProbeError(msg) from exc
        probed.append(full_name)
    return probed


class XYPlugin(Plugin):
    """Reflex plugin: `plugins=[reflex_xy.XYPlugin()]` in rxconfig.py.

    `post_compile` is the one plugin hook that receives the live App, and it
    fires at backend worker startup — after the socket server exists, before
    any client connects, and never during frontend-only compiles. It wires
    the data plane and then runs the figure-builder compile probes (§3.1) so
    escape-hatch builders fail `reflex run`, not the browser.
    """

    def post_compile(self, **context: Any) -> None:
        app = context.get("app")
        if app is not None:
            setup(app)
            probe_figure_builders()


def _token(source: "str | FigureHandle") -> str:
    """Normalize a public figure argument (handle or bare token string)."""
    token = token_of(source)
    if token is None:
        msg = f"expected a FigureHandle or figure token string, got {type(source).__name__}"
        raise TypeError(msg)
    return token


def append(
    token: "str | FigureHandle",
    x: Any,
    y: Any,
    *,
    color: Any = None,
    size: Any = None,
    trace: int = 0,
) -> None:
    """Stream-append points to a registered figure and push to subscribers.

    Thin alias for `registry.append` — see its docstring for the threading
    contract.
    """
    registry.append(_token(token), x, y, color=color, size=size, trace=trace)


def set_view(
    token: "str | FigureHandle", ranges: Any, *, animate: bool = True, history: bool = True
) -> None:
    """Out-of-band programmatic view patch (view-state.md §5.2).

    Mirrors `append`: callable from any event handler, background task, or
    thread; one wire message pushed room-wide, applied by every client
    through the same clamped mutation path as a gesture, `source: "api"`.
    """
    registry.set_view(_token(token), ranges, animate=animate, history=history)


def reset_view(token: "str | FigureHandle", axes: Any = None) -> None:
    """Out-of-band navigation to the home ranges (room-wide)."""
    registry.reset_view(_token(token), axes)


def select(
    token: "str | FigureHandle",
    *,
    range: Any = None,
    polygon: Any = None,
    rows: Any = None,
    history: bool = True,
) -> None:
    """Out-of-band programmatic selection (room-wide). Geometric forms
    resolve client-side like a gesture; `rows=` resolves kernel-side and is
    non-durable (see view-state.md §5.1)."""
    registry.select(_token(token), range=range, polygon=polygon, rows=rows, history=history)


def clear_selection(token: "str | FigureHandle") -> None:
    """Out-of-band selection clear (room-wide)."""
    registry.clear_selection(_token(token))


def reset_setup_for_tests() -> None:
    """Forget the wired namespace (test isolation only)."""
    global _namespace
    _namespace = None
