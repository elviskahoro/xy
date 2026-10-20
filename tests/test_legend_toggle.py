"""Legend click-to-toggle: kernel state, masked re-aggregation, client hide.

Kernel side (§34 predicate; interaction spec §10): the `legend_toggle`
message records hidden traces/categories on the Trace; `density_view`
re-bins under the mask (bypassing the unfiltered pyramid, tagging the
binning label and the reply's `filter` state), drills ship only visible
rows with canonical `shipped_sel`, selections exclude hidden rows, and
`decimate_view` skips hidden traces.

Client side: clicking a legend row hides the series — direct-tier
categorical traces re-filter their vertex buffers locally from retained
CPU columns (`_visMap` translates picks back to shipped rows), whole
traces skip draw+pick, and the kernel is notified either way.

Browser probes skip (never fail) without Chromium, like the repo's others.
"""

from __future__ import annotations

import base64
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

from conftest import density_category_chart, probe_document, run_browser_probe

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

import xy  # noqa: E402
from xy import channel  # noqa: E402
from xy._figure import Figure  # noqa: E402
from xy.export import find_chromium  # noqa: E402


def _density_fig(n: int = 400_000) -> tuple[Figure, np.ndarray]:
    rng = np.random.default_rng(7)
    x = rng.normal(size=n)
    y = rng.normal(size=n)
    cats = np.array(["A", "B", "C"])[rng.integers(0, 3, n)]
    fig = Figure()
    fig.scatter(x, y, color=cats, density=True)
    fig.build_payload()
    return fig, (x, y)


def test_legend_toggle_masks_density_rebin() -> None:
    fig, (x, y) = _density_fig()
    t = fig.traces[0]
    win = (x >= -3) & (x <= 3) & (y >= -3) & (y <= 3)

    base = fig.density_view(0, -3, 3, -3, 3, 512, 384)[0]["traces"][0]
    assert base["visible"] == int(win.sum())
    assert "filter" not in base and "-masked" not in base["binning"]

    reply = channel.handle_message(
        fig, {"type": "legend_toggle", "trace": 0, "category": 1, "hidden": True}
    )
    assert reply is None  # fire-and-forget state sync
    assert t.hidden_categories == {1}

    entry = fig.density_view(0, -3, 3, -3, 3, 512, 384)[0]["traces"][0]
    expect = int((win & (t.color_ch.codes != 1)).sum())
    assert entry["binning"].endswith("-masked")
    assert entry["visible"] == expect
    assert entry["filter"] == {"hidden_categories": [1]}

    # Untoggle restores the unmasked (pyramid-capable) path with no residue.
    channel.handle_message(
        fig, {"type": "legend_toggle", "trace": 0, "category": 1, "hidden": False}
    )
    entry = fig.density_view(0, -3, 3, -3, 3, 512, 384)[0]["traces"][0]
    assert "-masked" not in entry["binning"]
    assert entry["visible"] == int(win.sum())
    assert "filter" not in entry


def test_legend_toggle_drill_ships_only_visible_rows() -> None:
    fig, _ = _density_fig()
    t = fig.traces[0]
    channel.handle_message(
        fig, {"type": "legend_toggle", "trace": 0, "category": 1, "hidden": True}
    )
    entry = fig.density_view(0, -0.02, 0.02, -0.02, 0.02, 512, 384)[0]["traces"][0]
    assert entry["mode"] == "points"
    assert entry["filter"] == {"hidden_categories": [1]}
    # shipped_sel is canonical: the shipped subset must contain no hidden rows,
    # so pick/selection translation stays exact.
    assert len(t.shipped_sel) == entry["visible"]
    assert not np.any(t.color_ch.codes[t.shipped_sel] == 1)


def test_legend_toggle_selection_and_hidden_trace() -> None:
    fig, _ = _density_fig()
    t = fig.traces[0]
    channel.handle_message(
        fig, {"type": "legend_toggle", "trace": 0, "category": 1, "hidden": True}
    )
    sel = fig.select_range(-10, 10, -10, 10)
    assert not np.any(t.color_ch.codes[sel[0]] == 1)
    assert len(sel[0]) == int((t.color_ch.codes != 1).sum())

    channel.handle_message(fig, {"type": "legend_toggle", "trace": 0, "hidden": True})
    assert fig.density_view(0, -3, 3, -3, 3, 512, 384)[0]["traces"] == []
    assert fig.select_range(-10, 10, -10, 10) == {}


def test_legend_toggle_skips_hidden_line_decimation() -> None:
    n = 300_000
    x = np.linspace(0, 1000, n)
    y = np.sin(x / 7.0)
    fig = Figure()
    fig.line(x, y)
    fig.build_payload()
    assert fig.decimate_view(100.0, 200.0, 800)[0]["traces"]
    fig.legend_toggle(0, True)
    assert fig.decimate_view(100.0, 200.0, 800)[0]["traces"] == []


def test_legend_toggle_validation_never_mutates() -> None:
    fig, _ = _density_fig()
    t = fig.traces[0]
    # Out-of-range category, non-bool hidden, and a non-categorical trace all
    # return None (malformed-client contract) and leave state untouched.
    assert (
        channel.handle_message(
            fig, {"type": "legend_toggle", "trace": 0, "category": 99, "hidden": True}
        )
        is None
    )
    assert (
        channel.handle_message(fig, {"type": "legend_toggle", "trace": 0, "hidden": "yes"}) is None
    )
    assert t.hidden_categories == set() and t.hidden is False

    plain = Figure()
    plain.scatter(np.arange(10.0), np.arange(10.0))
    with pytest.raises(ValueError):
        plain.legend_toggle(0, True, category=0)


def test_density_entry_ships_slim_categorical_spec() -> None:
    # A categorical density trace aggregates its per-point codes into the
    # mean-color plane, but the legend still needs the encoding to offer
    # category rows — without them the §10 density-tier category toggle is
    # unreachable. The spec ships slim: categories + palette, no `buf`.
    fig, _ = _density_fig()
    entry = fig.build_payload()[0]["traces"][0]
    assert entry["tier"] == "density"
    color = entry["color"]
    assert color["mode"] == "categorical"
    assert color["categories"] == ["A", "B", "C"]
    assert len(color["palette"]) == 3
    assert "buf" not in color

    # Continuous stays deliberately unshipped: a gradient legend row on an
    # aggregated surface would claim color == density.
    cont = Figure()
    rng = np.random.default_rng(3)
    xs = rng.normal(size=1000)
    cont.scatter(xs, rng.normal(size=1000), color=np.abs(xs), density=True)
    assert "color" not in cont.build_payload()[0]["traces"][0]


def test_legend_toggle_option() -> None:
    data = {"x": np.arange(8.0), "y": np.arange(8.0)}
    disabled = xy.scatter_chart(xy.scatter("x", "y", data=data), xy.legend(toggle=False))
    assert disabled.figure().legend_options["toggle"] is False
    default = xy.scatter_chart(xy.scatter("x", "y", data=data), xy.legend())
    assert "toggle" not in default.figure().legend_options
    with pytest.raises(ValueError):
        xy.legend(toggle="yes")


_TOGGLE_PROBE = """
<script>
(async () => {
  const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
  try {
    const view = window.__fcProbeView;
    if (!view) throw new Error("no probe view captured");
    view._drawNow();
    view._raf = null;
    const sent = [];
    view.comm = { send: (m) => sent.push(m) };
    let rows = [];
    for (let i = 0; i < 200; i++) {
      rows = [...document.querySelectorAll('[data-xy-slot="legend_item"]')];
      if (rows.length >= 3) break;
      await sleep(25);
    }
    if (rows.length < 3) throw new Error(`expected 3 legend rows, got ${rows.length}`);
    const byName = (name) => rows.find((row) => row.textContent === name);
    const click = (row) => row.dispatchEvent(new MouseEvent("click"));
    const cat = view.gpuTraces[1];
    const fullN = cat.n;
    const a11yTotal = () => view._a11yPointGroups()
      .reduce((sum, g) => sum + view._a11yGroupCount(g), 0);
    const a11yBefore = a11yTotal();

    // Category toggle: filtered buffers, pick map, kernel notified.
    click(byName("A"));
    const afterHideN = cat.n;
    const hasVisMap = !!cat._visMap;
    let mappedClean = true;
    for (let j = 0; j < cat.n; j++) {
      if (cat._cpu.color[cat._visMap[j]] === 0) mappedClean = false;
    }
    const rowOff = byName("A").dataset.xyLegendOff !== undefined;
    const a11yAfterCategoryHide = a11yTotal();
    const categoryA11yCount = view._a11yGroupCount(cat);
    // The first trace owns flat rows 0..7. Advancing once from its last row
    // must reach the hidden category-A row 0 of the categorical trace, while
    // funnel stageNav keeps its separate visible-stage rule.
    const savedComm = view.comm;
    view.comm = null;
    view._a11yPointIndex = fullN - 1;
    view.canvas.dispatchEvent(new KeyboardEvent("keydown", { key: "ArrowRight", bubbles: true }));
    const hiddenCategoryReachable = view._hoverTarget?.g === cat
      && view._hoverTarget.index === 0
      && view._a11yKeyboardReadout?.total === a11yBefore;
    view.comm = savedComm;

    // Hovering a toggled-off row must not emphasize anything.
    byName("A").dispatchEvent(new PointerEvent("pointerenter"));
    const offHoverDims = view.gpuTraces.map((g) => g._legendDim ?? null);
    byName("A").dispatchEvent(new PointerEvent("pointerleave"));

    // Untoggle restores the full buffers.
    click(byName("A"));
    const restoredN = cat.n;
    const visMapCleared = !cat._visMap;

    // Whole-trace toggle: draw+pick skip flag and kernel notification.
    click(byName("alpha"));
    const alphaHidden = !!view.gpuTraces[0]._legendHidden;
    view._renderPick();
    const alphaUnpickable = view.gpuTraces[0].pickCount === 0;
    const a11yAfterTraceHide = a11yTotal();
    view.comm = null;
    view._a11yPointIndex = -1;
    view.canvas.dispatchEvent(new KeyboardEvent("keydown", { key: "Home", bubbles: true }));
    const hiddenTraceReachable = view._hoverTarget?.g === view.gpuTraces[0]
      && view._hoverTarget.index === 0
      && view._a11yKeyboardReadout?.total === a11yBefore;
    view.comm = savedComm;
    click(byName("alpha"));
    const alphaRestored = !view.gpuTraces[0]._legendHidden;

    document.body.setAttribute(
      "data-xy-legend-toggle",
      JSON.stringify({
        fullN, afterHideN, hasVisMap, mappedClean, rowOff, offHoverDims,
        restoredN, visMapCleared, alphaHidden, alphaUnpickable, alphaRestored,
        a11yBefore, a11yAfterCategoryHide, categoryA11yCount,
        hiddenCategoryReachable, a11yAfterTraceHide, hiddenTraceReachable,
        sent,
      })
    );
  } catch (err) {
    document.body.setAttribute(
      "data-xy-legend-toggle-error",
      String((err && err.stack) || err)
    );
  }
})();
</script>
"""


_DENSITY_TOGGLE_PROBE = """
<script>
(async () => {
  const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
  const b64 = (s) => Uint8Array.from(atob(s), (c) => c.charCodeAt(0)).buffer;
  try {
    const view = window.__fcProbeView;
    if (!view) throw new Error("no probe view captured");
    view._drawNow();
    view._raf = null;
    const sent = [];
    view.comm = { send: (m) => sent.push(m) };
    let rows = [];
    for (let i = 0; i < 200; i++) {
      rows = [...document.querySelectorAll('[data-xy-slot="legend_item"]')];
      if (rows.length >= 3) break;
      await sleep(25);
    }
    if (rows.length < 3) throw new Error(`expected 3 category rows, got ${rows.length}`);
    const byName = (name) => rows.find((row) => row.textContent === name);
    const click = (row) => row.dispatchEvent(new MouseEvent("click"));
    const g = view.gpuTraces[0];
    const gridTotal = () => {
      const d = g.density;
      let s = 0;
      for (let i = 0; i < d.grid.length; i++) s += d.grid[i];
      return Math.round(s);
    };
    const totalBefore = gridTotal();

    // Toggle category B off: the kernel must be notified AND a masked
    // re-bin must be REQUESTED even though the standing home aggregate
    // still covers the view (§10: filter-dirty beats aggregate-stands and
    // the T13 duplicate memo).
    click(byName("B"));
    await sleep(50);
    const maskReq = sent.find((m) => m.type === "density_view");

    // Feed the client the kernel's actual masked reply (precomputed by the
    // Python kernel below, byte-faithful envelope). It must REPLACE the
    // standing unfiltered texture, not land facts-only (§34: a covering
    // texture binned under another hidden set is a stale aggregate).
    const reply = window.__fcMaskedReply;
    reply.message.seq = maskReq ? maskReq.seq : -1;
    view._onKernelMsg(reply.message, reply.buffers.map(b64));
    const totalMasked = gridTotal();
    const keyMasked = g.density._filterKey;

    // Untoggle: same window, same screen — but a different hidden set, so
    // the "answered duplicate" memo must not swallow the restore request.
    click(byName("B"));
    await sleep(50);
    const restoreReq = sent.filter((m) => m.type === "density_view").length;
    const toggleMsgs = sent.filter((m) => m.type === "legend_toggle");

    document.body.setAttribute(
      "data-xy-density-toggle",
      JSON.stringify({
        totalBefore, toggleMsgs, maskReqSent: !!maskReq,
        totalMasked, keyMasked, densityViewCount: restoreReq,
      })
    );
  } catch (err) {
    document.body.setAttribute(
      "data-xy-density-toggle-error",
      String((err && err.stack) || err)
    );
  }
})();
</script>
"""


def test_browser_density_category_toggle_rebins() -> None:
    """The §10 density-tier category toggle, end to end against the real
    client: category rows exist (slim color spec), a toggle forces a masked
    re-bin request past every T13 elision layer (aggregate-stands, the
    answered-duplicate memo), the kernel's stamped reply repaints the
    standing texture (never facts-only across filter keys), and the
    untoggle re-requests. Each assertion is one of the elisions that
    silently swallowed the round-trip."""
    chromium = find_chromium()
    if not chromium:
        pytest.skip("no chromium available for the density toggle probe")

    chart, cat, x, y = density_category_chart()

    # The kernel's own masked reply for the home window — the byte-faithful
    # envelope a live transport would deliver after the toggle's
    # legend_toggle message.
    fig = chart.figure()
    fig.build_payload()
    assert (
        channel.handle_message(
            fig, {"type": "legend_toggle", "trace": 0, "category": 1, "hidden": True}
        )
        is None
    )
    reply = channel.handle_message(
        fig,
        {
            "type": "density_view",
            "trace": 0,
            "seq": 0,
            "x0": float(x.min()),
            "x1": float(x.max()),
            "y0": float(y.min()),
            "y1": float(y.max()),
            "w": 512,
            "h": 384,
        },
    )
    assert reply is not None
    masked_message, masked_buffers = reply
    entry = masked_message["traces"][0]
    assert entry["binning"].endswith("-masked") and entry["filter"] == {"hidden_categories": [1]}

    reply_js = json.dumps(
        {
            "message": masked_message,
            "buffers": [base64.b64encode(bytes(b)).decode() for b in masked_buffers],
        }
    )

    document = probe_document(
        chart,
        _DENSITY_TOGGLE_PROBE,
        head=f"<script>window.__fcMaskedReply = {reply_js};</script>",
    )

    with tempfile.TemporaryDirectory() as td:
        page = Path(td) / "density_toggle.html"
        payload = run_browser_probe(
            chromium, document, page, "data-xy-density-toggle", label="density toggle"
        )

    # Home grid bins every point.
    assert payload["totalBefore"] == pytest.approx(cat.size, rel=0.01), payload
    # The toggle synced the kernel...
    assert [(m["trace"], m["category"], m["hidden"]) for m in payload["toggleMsgs"]] == [
        (0, 1, True),
        (0, 1, False),
    ], payload
    # ...and REQUESTED the masked re-bin (aggregate-stands + dup-memo bypass).
    assert payload["maskReqSent"] is True, payload
    # The stamped reply repainted the standing texture: ~2/3 of the points.
    assert payload["totalMasked"] == pytest.approx((cat != 1).sum(), rel=0.02), payload
    assert payload["keyMasked"] == "1", payload
    # The untoggle re-requested despite the answered same-window memo.
    assert payload["densityViewCount"] == 2, payload


_DRILL_DIRTY_PROBE = """
<script>
(async () => {
  const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
  const b64 = (s) => Uint8Array.from(atob(s), (c) => c.charCodeAt(0)).buffer;
  try {
    const view = window.__fcProbeView;
    if (!view) throw new Error("no probe view captured");
    view._drawNow();
    view._raf = null;
    const sent = [];
    view.comm = { send: (m) => sent.push(m) };
    let rows = [];
    for (let i = 0; i < 200; i++) {
      rows = [...document.querySelectorAll('[data-xy-slot="legend_item"]')];
      if (rows.length >= 3) break;
      await sleep(25);
    }
    if (rows.length < 3) throw new Error(`expected 3 category rows, got ${rows.length}`);
    const byName = (name) => rows.find((row) => row.textContent === name);
    const g = view.gpuTraces[0];

    byName("B").dispatchEvent(new MouseEvent("click"));
    await sleep(50);
    const maskReq = sent.find((m) => m.type === "density_view");
    const dirtyAfterToggle = !!g._filterDirty;

    // Feed the kernel's masked POINTS reply (a window under the drill
    // budget). Its filter stamp matches the client's mask, so it is
    // admitted — but it lands no aggregate under that mask: the only
    // standing grid is still the unmasked home texture.
    const reply = window.__fcMaskedDrillReply;
    reply.message.seq = maskReq ? maskReq.seq : -1;
    view._onKernelMsg(reply.message, reply.buffers.map(b64));
    const drillApplied = !!g.drill;
    const dirtyAfterDrill = !!g._filterDirty;

    // The elision the flag exists to defeat (§10): with the unmasked home
    // aggregate still standing (and filter-blind), request planning must
    // keep forcing the full-window masked re-bin until a mask-stamped
    // AGGREGATE lands — otherwise the stale texture draws forever once the
    // drill retires.
    const plan = view._densityRequestPlan(g, view.view);
    const planForcesFullWindow = !!(plan && plan.step === false);

    document.body.setAttribute(
      "data-xy-drill-dirty",
      JSON.stringify({ maskReqSent: !!maskReq, dirtyAfterToggle, drillApplied,
                       dirtyAfterDrill, planForcesFullWindow })
    );
  } catch (err) {
    document.body.setAttribute(
      "data-xy-drill-dirty-error",
      String((err && err.stack) || err)
    );
  }
})();
</script>
"""


def test_browser_masked_drill_reply_keeps_filter_dirty() -> None:
    """A masked points-mode reply must NOT clear `_filterDirty`: it lands no
    aggregate under the mask, so the unmasked home texture is still the only
    standing grid. If the flag clears, `lodAggregateStands` (filter-blind)
    re-arms the T13 elision and the stale unmasked surface draws indefinitely
    once the drill retires, while the legend row reads hidden."""
    chromium = find_chromium()
    if not chromium:
        pytest.skip("no chromium available for the drill filter-dirty probe")

    chart, cat, x, y = density_category_chart()

    fig = chart.figure()
    fig.build_payload()
    channel.handle_message(
        fig, {"type": "legend_toggle", "trace": 0, "category": 1, "hidden": True}
    )
    reply = channel.handle_message(
        fig,
        {
            "type": "density_view",
            "trace": 0,
            "seq": 0,
            "x0": -0.02,
            "x1": 0.02,
            "y0": -0.02,
            "y1": 0.02,
            "w": 512,
            "h": 384,
        },
    )
    assert reply is not None
    drill_message, drill_buffers = reply
    entry = drill_message["traces"][0]
    assert entry["mode"] == "points"
    assert entry["filter"] == {"hidden_categories": [1]}

    reply_js = json.dumps(
        {
            "message": drill_message,
            "buffers": [base64.b64encode(bytes(b)).decode() for b in drill_buffers],
        }
    )

    document = probe_document(
        chart,
        _DRILL_DIRTY_PROBE,
        head=f"<script>window.__fcMaskedDrillReply = {reply_js};</script>",
    )

    with tempfile.TemporaryDirectory() as td:
        page = Path(td) / "drill_dirty.html"
        payload = run_browser_probe(
            chromium, document, page, "data-xy-drill-dirty", label="drill filter-dirty"
        )

    assert payload["maskReqSent"] is True, payload
    assert payload["dirtyAfterToggle"] is True, payload
    assert payload["drillApplied"] is True, payload
    # The contract under test: only a mask-stamped aggregate clears the flag.
    assert payload["dirtyAfterDrill"] is True, payload
    assert payload["planForcesFullWindow"] is True, payload


_STANDALONE_TOGGLE_PROBE = """
<script>
(async () => {
  const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
  try {
    const view = window.__fcProbeView;
    if (!view) throw new Error("no probe view captured");
    // Standalone (kernel-less): the retained home sample overlay is the only
    // point representation. Keep comm null so lodSampleForView runs its
    // overlay path (#225 — kernel-attached clients never draw it).
    view.comm = null;
    view._drawNow();
    view._raf = null;
    const g = view.gpuTraces.find((t) => t.tier === "density");
    if (!g) throw new Error("no density trace");
    if (!g.sampleOverlay) throw new Error("no home sample overlay at init");

    // Record the sampled-tier overlay drawn in one frame (mirrors the
    // §28 zoom-out probe): {n, alpha} when lodSampleForView returned an
    // overlay, else null.
    const sampleDrawn = () => {
      let drawn = null;
      const real = view._drawPoints;
      view._drawPoints = function (d, xm, ym, alpha) {
        if (d && d.tier === "sampled") drawn = { n: d.n, alpha: alpha === undefined ? 1 : alpha };
        return real.apply(this, arguments);
      };
      view._drawNow();
      view._drawPoints = real;
      return drawn;
    };
    const badgeOn = () =>
      view._reductionBadgeItems().some((i) => i.indexOf("sampled") === 0);

    // Wait for legend rows to populate, then resolve by label.
    let rows = [];
    for (let i = 0; i < 200; i++) {
      rows = [...document.querySelectorAll('[data-xy-slot="legend_item"]')];
      if (rows.length >= 3) break;
      await sleep(25);
    }
    if (rows.length < 3) throw new Error(`expected 3 legend rows, got ${rows.length}`);
    const byName = (name) => rows.find((row) => row.textContent === name);
    const click = (row) => row.dispatchEvent(new MouseEvent("click"));

    const before = sampleDrawn();
    const beforeBadge = badgeOn();
    const overlayAliveBefore = !!g.sampleOverlay;
    const fullN = g.sampleOverlay.n;

    // Toggle category B off (the §10 density-tier path:
    // _applyCategoryVisibility -> lodDropDensityCache + filter overlays).
    click(byName("B"));
    const after = sampleDrawn();
    const afterBadge = badgeOn();
    const afterState = {
      sampleOverlayAlive: !!g.sampleOverlay,
      shownOverlayIsSample: g._shownSampleOverlay === g.sampleOverlay,
      sampleFadedOut: !!g._sampleFadedOut,
      cacheLen: (g.densityCache || []).length,
      cacheHasDensity: (g.densityCache || []).some((d) => d === g.density),
      cacheOverlayIsSample: (g.densityCache || []).some(
        (d) => d && d.overlay === g.sampleOverlay
      ),
      filterDirty: !!g._filterDirty,
    };

    // A second frame at home zoom: the bug stranded the overlay permanently
    // across frames; the fix keeps it drawn every frame.
    const after2 = sampleDrawn();
    const after2Badge = badgeOn();

    // An in-window pan (view stays inside the home window) then redraw:
    // confirms the overlay is not stranded by a home-range pan either.
    const v0 = view.view0;
    const shift = (v0.x1 - v0.x0) * 0.02;
    view.view = { x0: v0.x0 + shift, x1: v0.x1 + shift, y0: v0.y0, y1: v0.y1 };
    const afterPan = sampleDrawn();
    view.view = { x0: v0.x0, x1: v0.x1, y0: v0.y0, y1: v0.y1 };

    // Untoggle B: kept categories keep drawing, the hidden one returns.
    click(byName("B"));
    const afterUntoggle = sampleDrawn();
    const afterUntoggleBadge = badgeOn();

    document.body.setAttribute(
      "data-xy-standalone-toggle",
      JSON.stringify({
        fullN,
        before, beforeBadge, overlayAliveBefore,
        after, afterBadge, afterState,
        after2, after2Badge,
        afterPan,
        afterUntoggle, afterUntoggleBadge,
      })
    );
  } catch (err) {
    document.body.setAttribute(
      "data-xy-standalone-toggle-error",
      String((err && err.stack) || err)
    );
  }
})();
</script>
"""


def test_browser_standalone_density_toggle_keeps_sample_overlay() -> None:
    """Standalone (kernel-less) density charts: the retained home sample
    overlay is the only point representation (#225). Toggling a legend
    category must keep drawing the surviving (filtered) overlay at home
    zoom — lodDropDensityCache must leave the LIVE grid in the cache so
    lodSampleForView can still reach g.density.overlay === g.sampleOverlay.

    Regression for the bug where lodDropDensityCache reset
    ``g.densityCache = []`` (a truthy empty array), defeating the
    ``g.densityCache || (g.density ? [g.density] : [])`` fallback in
    lodSampleForView and permanently hiding the points tier after one
    toggle at home zoom (the zoom-out home-restore is a no-op there, so
    nothing ever re-added an overlay-carrying cache entry).
    """
    chromium = find_chromium()
    if not chromium:
        pytest.skip("no chromium available for the standalone toggle probe")

    # 60k points: fits the 200k resolvability budget so the home sample
    # overlay draws at full alpha at home zoom (the #225 gate suppresses
    # it above the budget — the bug only manifests when it first draws).
    chart, _cat, _x, _y = density_category_chart(n=60_000)

    document = probe_document(chart, _STANDALONE_TOGGLE_PROBE)

    with tempfile.TemporaryDirectory() as td:
        page = Path(td) / "standalone_toggle.html"
        payload = run_browser_probe(
            chromium,
            document,
            page,
            "data-xy-standalone-toggle",
            label="standalone density toggle",
        )

    full_n = payload["fullN"]

    # Pre-toggle: the home sample overlay draws at full alpha, badge on.
    assert payload["overlayAliveBefore"] is True, payload
    assert payload["before"] is not None, payload
    assert payload["before"]["alpha"] == 1, payload
    assert payload["before"]["n"] == full_n, payload
    assert payload["beforeBadge"] is True, payload

    # Post-toggle (the fix): the filtered overlay STILL draws at full
    # alpha at home zoom — not null (the bug), with the badge still on.
    after = payload["after"]
    assert after is not None, payload
    assert after["alpha"] == 1, payload
    assert 0 < after["n"] < full_n, payload  # category B filtered out, rest kept
    assert payload["afterBadge"] is True, payload

    st = payload["afterState"]
    # g.sampleOverlay survived the cache drop (the documented intent).
    assert st["sampleOverlayAlive"] is True, payload
    # The live grid stayed in the rebuilt cache (not a truthy empty []).
    assert st["cacheLen"] == 1, payload
    assert st["cacheHasDensity"] is True, payload
    assert st["cacheOverlayIsSample"] is True, payload
    # _drawDensitySample resolved the overlay and recorded it on the trace.
    assert st["shownOverlayIsSample"] is True, payload
    assert st["sampleFadedOut"] is False, payload
    # The masked re-bin request is still armed (the fix does not elide it).
    assert st["filterDirty"] is True, payload

    # No stranding across frames at home zoom.
    assert payload["after2"] is not None, payload
    assert payload["after2"]["alpha"] == 1, payload
    assert payload["after2Badge"] is True, payload
    # No stranding across a home-range pan either.
    assert payload["afterPan"] is not None, payload

    # Untoggle restores the full overlay; kept categories kept drawing
    # throughout and the hidden category's points return.
    assert payload["afterUntoggle"] is not None, payload
    assert payload["afterUntoggle"]["n"] == full_n, payload
    assert payload["afterUntoggleBadge"] is True, payload


def test_browser_legend_click_toggles_series() -> None:
    chromium = find_chromium()
    if not chromium:
        pytest.skip("no chromium available for the legend toggle probe")

    codes = np.array(["A", "A", "A", "A", "A", "B", "B", "B"])
    data = {"x": np.arange(8.0), "y": np.arange(8.0)}
    chart = xy.scatter_chart(
        xy.scatter("x", "y", data=data, name="alpha"),
        xy.scatter("x", "y", data=data, color=codes),
        xy.legend(),
        width=520,
        height=340,
    )

    document = probe_document(chart, _TOGGLE_PROBE)

    with tempfile.TemporaryDirectory() as td:
        page = Path(td) / "legend_toggle.html"
        payload = run_browser_probe(
            chromium, document, page, "data-xy-legend-toggle", label="legend toggle"
        )

    assert payload["fullN"] == 8 and payload["afterHideN"] == 3, payload
    assert payload["hasVisMap"] is True and payload["mappedClean"] is True, payload
    assert payload["rowOff"] is True, payload
    # Off rows are inert for hover emphasis.
    assert payload["offHoverDims"] == [None, None], payload
    assert payload["restoredN"] == 8 and payload["visMapCleared"] is True, payload
    assert payload["alphaHidden"] is True and payload["alphaUnpickable"] is True, payload
    assert payload["alphaRestored"] is True, payload
    # Visual legend filtering must not silently remove ordinary point-series
    # data from the screen-reader walk or change its "Point N of total" count.
    assert payload["a11yBefore"] == 16, payload
    assert payload["a11yAfterCategoryHide"] == 16, payload
    assert payload["categoryA11yCount"] == 8, payload
    assert payload["hiddenCategoryReachable"] is True, payload
    assert payload["a11yAfterTraceHide"] == 16, payload
    assert payload["hiddenTraceReachable"] is True, payload
    # Kernel stays in sync: every click shipped a legend_toggle message.
    kinds = [
        (m.get("type"), m.get("trace"), m.get("category"), m.get("hidden")) for m in payload["sent"]
    ]
    assert kinds == [
        ("legend_toggle", 1, 0, True),
        ("legend_toggle", 1, 0, False),
        ("legend_toggle", 0, None, True),
        ("legend_toggle", 0, None, False),
    ], payload["sent"]


# --- the visible-row cache is bounded, and it is reported (§27) --------------


def test_legend_vis_cache_is_dropped_when_nothing_is_hidden() -> None:
    """Un-hiding everything releases the index, instead of pinning 8 B/row.

    The cache exists so a pan/zoom sequence does not repeat the O(N) code scan
    while the predicate holds. Once no category is hidden there is no consumer
    for it at all, so keeping it is pure retention — and it was never dropped,
    never invalidated and never reported.
    """
    from xy import interaction

    fig, _ = _density_fig(200_000)
    trace = fig.traces[0]
    assert trace._legend_vis_cache is None

    channel.handle_message(
        fig, {"type": "legend_toggle", "trace": 0, "category": 1, "hidden": True}
    )
    interaction.density_view(fig, trace.id, -4.0, 4.0, -4.0, 4.0, 256, 192)
    cached = trace._legend_vis_cache
    assert cached is not None and cached[1].size > 0
    # ...and it shows up in the report while it is held.
    assert fig.memory_report()["legend_vis_cache_bytes"] == cached[1].nbytes

    # The un-hide itself releases it. Waiting for the next `density_view` to
    # notice would pin the index whenever that view never comes — and a legend
    # click restoring every series is a perfectly ordinary last interaction.
    channel.handle_message(
        fig, {"type": "legend_toggle", "trace": 0, "category": 1, "hidden": False}
    )
    assert trace._legend_vis_cache is None
    assert fig.memory_report()["legend_vis_cache_bytes"] == 0

    # ...and a view afterwards neither resurrects nor re-pins it.
    interaction.density_view(fig, trace.id, -4.0, 4.0, -4.0, 4.0, 256, 192)
    assert trace._legend_vis_cache is None
    assert fig.memory_report()["legend_vis_cache_bytes"] == 0


def test_legend_vis_cache_survives_a_partial_unhide() -> None:
    """Only an empty hidden set releases it; with one category still hidden the
    cache is still live and must be re-keyed, not dropped."""
    from xy import interaction

    fig, _ = _density_fig(200_000)
    trace = fig.traces[0]
    for code in (0, 1):
        channel.handle_message(
            fig, {"type": "legend_toggle", "trace": 0, "category": code, "hidden": True}
        )
    interaction.density_view(fig, trace.id, -4.0, 4.0, -4.0, 4.0, 256, 192)
    two_hidden = trace._legend_vis_cache
    assert two_hidden is not None

    channel.handle_message(
        fig, {"type": "legend_toggle", "trace": 0, "category": 0, "hidden": False}
    )
    assert trace._legend_vis_cache is not None  # still one category hidden
    interaction.density_view(fig, trace.id, -4.0, 4.0, -4.0, 4.0, 256, 192)
    one_hidden = trace._legend_vis_cache
    assert one_hidden is not None
    assert one_hidden[0] != two_hidden[0]  # re-keyed on the new hidden set
    assert one_hidden[1].size > two_hidden[1].size  # and more rows are visible

    channel.handle_message(
        fig, {"type": "legend_toggle", "trace": 0, "category": 1, "hidden": False}
    )
    assert trace._legend_vis_cache is None
    assert fig.memory_report()["legend_vis_cache_bytes"] == 0


def test_whole_trace_hide_leaves_the_category_cache_alone() -> None:
    """`hidden=True` with no category is the trace-level switch; it shares
    nothing with the per-category predicate and must not drop its cache."""
    from xy import interaction

    fig, _ = _density_fig(200_000)
    trace = fig.traces[0]
    channel.handle_message(
        fig, {"type": "legend_toggle", "trace": 0, "category": 1, "hidden": True}
    )
    interaction.density_view(fig, trace.id, -4.0, 4.0, -4.0, 4.0, 256, 192)
    assert trace._legend_vis_cache is not None

    channel.handle_message(fig, {"type": "legend_toggle", "trace": 0, "hidden": False})
    assert trace.hidden_categories == {1}
    assert trace._legend_vis_cache is not None


def test_legend_vis_cache_bytes_is_counted_in_resident_total() -> None:
    """§27: if a number isn't in the report, it isn't real."""
    from xy import interaction

    fig, _ = _density_fig(200_000)
    channel.handle_message(
        fig, {"type": "legend_toggle", "trace": 0, "category": 1, "hidden": True}
    )
    interaction.density_view(fig, fig.traces[0].id, -4.0, 4.0, -4.0, 4.0, 256, 192)
    report = fig.memory_report()
    assert report["legend_vis_cache_bytes"] > 0
    assert report["resident_array_bytes"] >= (
        report["canonical_capacity_bytes"]
        + report["channel_bytes"]
        + report["pyramid_bytes"]
        + report["bin_color_bytes"]
        + report["legend_vis_cache_bytes"]
    )
