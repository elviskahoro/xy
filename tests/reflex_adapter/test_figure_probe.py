"""The @reflex_xy.figure compile probe: escape-hatch builders fail at compile.

The probed states below raise only while their module flag is armed, so the
process-wide state walk stays healthy for every other test in the session.
"""

from __future__ import annotations

import asyncio

import numpy as np
import pytest
import reflex as rx

import reflex_xy
import xy
from reflex_xy.app import FigureProbeError, _ProbeLoopError, probe_figure_builders

#: Armed per-test; every probed builder below is a no-op otherwise.
_ARM = {
    "hallucinated": False,
    "bad_config": False,
    "bad_return": False,
    "session": False,
    "async_ran": False,
    "async_bad": False,
}


class ProbeDemo(rx.State):
    n: int = 8

    @reflex_xy.figure
    def healthy(self) -> xy.Chart:
        xs = np.linspace(0.0, 1.0, self.n)
        return xy.scatter_chart(xy.scatter(xs, xs))

    @reflex_xy.figure
    def hallucinated(self):
        if not _ARM["hallucinated"]:
            return None
        return xy.polar_scatter([1.0], [1.0])  # no such factory

    @reflex_xy.figure(probe="figure")
    def bad_config(self):
        if not _ARM["bad_config"]:
            return None
        return xy.scatter_chart(xy.scatter([1.0], [1.0], colormap="virids"))

    @reflex_xy.figure
    def session_bound(self):
        if not _ARM["session"]:
            return None
        # Session-dependent by declaration: reads self.router. Against the
        # probe's default state this raises; the probe must downgrade.
        raise RuntimeError(f"no session for {self.router.session.client_token!r}")

    @reflex_xy.figure
    def bad_return(self):
        if not _ARM["bad_return"]:
            return None
        return {"x": [1.0], "y": [2.0]}  # not a chart: caught at "build" level

    @reflex_xy.figure(probe=False)
    def opted_out(self):
        raise AssertionError("probe=False builders must never run at compile")

    @reflex_xy.figure
    async def async_default(self):
        raise AssertionError("async builders are not probed by default")

    @reflex_xy.figure(probe="build")
    async def async_opted_in(self):
        _ARM["async_ran"] = True
        await asyncio.sleep(0)
        return None

    @reflex_xy.figure(probe="build")
    async def async_opted_in_bad(self):
        if not _ARM["async_bad"]:
            return None
        await asyncio.sleep(0)
        raise ValueError("async builder failed against default state")


@pytest.fixture(autouse=True)
def _disarm():
    yield
    for key in _ARM:
        _ARM[key] = False


def test_probe_runs_sync_builders_and_skips_optouts():
    # Deliberately unscoped: this one test exercises the production shape —
    # the whole-state-tree walk post_compile runs (every other builder in the
    # session must probe clean under default state, the contract it enforces).
    probed = probe_figure_builders()
    names = {name.rsplit(".", 1)[-1] for name in probed if "probe_demo" in name}
    assert "healthy" in names
    assert "opted_out" not in names
    assert "async_default" not in names
    assert "async_opted_in" in names  # explicit opt-in runs under asyncio.run
    assert _ARM["async_ran"]


def test_hallucinated_chart_api_fails_the_compile():
    """Problem 2 of the options doc, closed for the escape hatch: the
    builder body is no longer dead code until hydrate."""
    _ARM["hallucinated"] = True
    with pytest.raises(FigureProbeError, match="hallucinated") as excinfo:
        probe_figure_builders(ProbeDemo)
    assert "polar_scatter" in str(excinfo.value)
    assert isinstance(excinfo.value.__cause__, AttributeError)


def test_probe_figure_level_compiles_the_result():
    _ARM["bad_config"] = True
    with pytest.raises(FigureProbeError, match="bad_config") as excinfo:
        probe_figure_builders(ProbeDemo)
    assert "colormap" in str(excinfo.value)


def test_session_dependent_builder_downgrades_to_warning():
    _ARM["session"] = True
    with pytest.warns(RuntimeWarning, match="session_bound.*reads the session"):
        probed = probe_figure_builders(ProbeDemo)
    assert not any(name.endswith("session_bound") for name in probed)


def test_non_chart_return_fails_the_default_probe_level():
    """The default "build" level type-checks the return: a value no registry
    publish can accept must not survive to hydrate."""
    _ARM["bad_return"] = True
    with pytest.raises(FigureProbeError, match="bad_return") as excinfo:
        probe_figure_builders(ProbeDemo)
    assert "dict" in str(excinfo.value)


def test_invalid_probe_level_is_refused_at_decoration():
    with pytest.raises(ValueError, match="probe="):

        class BadProbe(rx.State):  # noqa: F841 - definition is the assertion
            @reflex_xy.figure(probe="everything")
            def chart(self):
                return None


@pytest.mark.parametrize("level", [0, 0.0, 1, True])
def test_probe_levels_are_identity_strict(level):
    """0/0.0 compare equal to False (and 1/True to each other) but are not
    probe levels; equality-based membership silently accepted them."""
    with pytest.raises(ValueError, match="probe="):

        class BadProbe(rx.State):  # noqa: F841 - definition is the assertion
            @reflex_xy.figure(probe=level)
            def chart(self):
                return None


def test_async_opted_in_runs_from_inside_a_running_event_loop():
    """The uvicorn case: `post_compile` fires from inside the server's already-
    running loop (uvicorn's `Config.load()` runs the ASGI factory inside
    `Server._serve()`'s coroutine). Pre-fix, the probe called
    `asyncio.run(builder(...))` there, which raised
    `RuntimeError: asyncio.run() cannot be called from a running event loop`
    and aborted backend startup for an explicitly-opted-in builder that
    succeeds against default state. The probe now drives the builder on a
    fresh, private loop instead, so the foreign loop is untouched and the
    builder runs to completion.

    This is the exact reproduction from the bug report's "minimal mechanism
    repro": calling `probe_figure_builders()` from inside `asyncio.run(main())`.
    """
    _ARM["async_ran"] = False

    async def main() -> list[str]:
        # `asyncio.get_running_loop()` succeeds here, so the buggy
        # `asyncio.run(builder(...))` would have raised inside this body.
        return probe_figure_builders(ProbeDemo)

    probed = asyncio.run(main())
    assert any(name.endswith("async_opted_in") for name in probed), probed
    assert _ARM["async_ran"], "the opted-in async builder never executed"


def test_async_opted_in_failure_from_inside_running_loop_still_surfaces():
    """A genuinely-failing opted-in async builder must still surface as a
    `FigureProbeError` with the builder-failure advice (`probe=False`) — the
    loop-aware path must not swallow real builder errors or reclassify them as
    probe-infrastructure failures."""
    _ARM["async_bad"] = True

    async def main() -> list[str]:
        return probe_figure_builders(ProbeDemo)

    with pytest.raises(FigureProbeError, match="async_opted_in_bad") as excinfo:
        asyncio.run(main())
    msg = str(excinfo.value)
    # Builder-failure path: the message advises disabling the probe, which is
    # correct for a genuine builder defect.
    assert "probe=False" in msg, msg
    assert isinstance(excinfo.value.__cause__, ValueError)


def test_loop_creation_failure_is_not_builder_failure_advice(monkeypatch):
    """Defensive path: when a foreign loop is running and the probe's
    worker-loop *creation* itself fails (e.g. interpreter shutdown, or a
    runtime that restricts loop creation), the surfaced
    ``FigureProbeError`` must NOT advise ``probe=False`` — the builder is
    fine, the probe infrastructure could not run here. The message must
    distinguish that from a builder failure.

    The worker thread calls ``asyncio.new_event_loop``; making that raise
    reproduces the genuine infra path (distinct from a builder that raises
    ``RuntimeError`` from its own body, which must surface as a builder
    failure with the ``probe=False`` advice — see
    :func:`test_async_opted_in_failure_from_inside_running_loop_still_surfaces`).
    """

    def _boom(*args: object, **kwargs: object) -> object:
        raise RuntimeError("loop policy disabled in this runtime")

    monkeypatch.setattr(asyncio, "new_event_loop", _boom)

    async def main() -> list[str]:
        return probe_figure_builders(ProbeDemo)

    with pytest.raises(FigureProbeError, match="async_opted_in") as excinfo:
        asyncio.run(main())
    msg = str(excinfo.value)
    # Infrastructure-distinct message: never the builder-failure advice that
    # would tell the user to disable the probe they explicitly opted into.
    assert "probe=False" not in msg, msg
    assert "could not run" in msg, msg
    assert isinstance(excinfo.value.__cause__, _ProbeLoopError)
