"""Standalone (kernel-less) export of a per-point RGBA scatter with authored
markers must paint each marker in its own color.

Regression for the `renderStandalone` hydration bug: the loop wholescale-
replaced `g._cpu` (dropping the build-retained `g._cpu.rgba` for
`color: {mode: "direct_rgba"}` scatters) and re-attached the u8 RGBA column
under `g._cpu.color`. The Canvas2D authored-marker overlay reads
`g._cpu.rgba` (via `_authoredScatterRgba`) and silently fell back to the flat
trace color `[0.3, 0.47, 0.66, 1]` when it was missing, so every authored
marker painted steel-blue in the static HTML export while the widget path
painted the per-point RGBA colors correctly.

This drives the real client in headless Chromium and asserts both the
post-hydration CPU record shape and the rendered overlay pixels.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import xy
from conftest import probe_document, run_browser_probe
from xy.export import find_chromium

# Per-point RGBA paint (straight-alpha, packed u8 on the wire):
#   red=[255,0,0,255]  green=[0,255,0,255]  blue=[0,0,255,255]  yellow=[255,255,0,255]
_COLORS = ["#ff0000", "#00ff00", "#0000ff", "#ffff00"]

# `probe_document` injects `window.__fcProbeView = xy.renderStandalone(...)` so
# this probe reads the live standalone view. Each marker paints on the Canvas2D
# overlay (the GPU point path queues authored-marker draws into
# `_authoredScatterDraws` and returns early).
_PROBE = """
<script>
setTimeout(() => {
  try {
    const view = window.__fcProbeView;
    view._drawNow();
    view.gl.finish();
    const g = view.gpuTraces.find((t) => t.trace.kind === "scatter");
    if (!g) throw new Error("no scatter trace");
    // --- CPU-record shape after renderStandalone hydration ---
    const colorMode = g.colorMode;
    const hasRgba = g._cpu && g._cpu.rgba instanceof Uint8Array;
    const rgbaLen = hasRgba ? g._cpu.rgba.length : 0;
    const colorIsUndef = g._cpu && g._cpu.color === undefined;
    const hasX = g._cpu && g._cpu.x !== undefined && g._cpu.x.length === 4;
    const hasY = g._cpu && g._cpu.y !== undefined && g._cpu.y.length === 4;
    // --- _authoredScatterRgba reads per-point colors from g._cpu.rgba ---
    const authored = [0,1,2,3].map((i) =>
      view._authoredScatterRgba(g, i).map((v) => Math.round(v * 255)));
    // --- overlay pixel scan at each projected marker position ---
    const overlay = view.overlay.getContext("2d");
    const radius = Math.round(20 * view.dpr);
    const counts = [0,1,2,3].map((i) => {
      const x = view._decodeValue(g._cpu.x, g.xMeta, i);
      const y = view._decodeValue(g._cpu.y, g.yMeta, i);
      const [px, py] = view._projectDataPoint(g.xAxis, g.yAxis, x, y, null);
      const w = radius * 2 + 1;
      const ox = Math.max(0, Math.round(px * view.dpr) - radius);
      const oy = Math.max(0, Math.round(py * view.dpr) - radius);
      const aw = Math.min(view.overlay.width - ox, w);
      const ah = Math.min(view.overlay.height - oy, w);
      const ids = overlay.getImageData(ox, oy, aw, ah).data;
      let lit = 0;
      for (let i = 0; i < ids.length; i += 4) {
        if (ids[i+3] > 30) lit++;
      }
      return lit;
    });
    // Count fallback steel-blue [77,120,168] pixels anywhere in the plot rect.
    const plotScan = overlay.getImageData(
      Math.round(view.plot.x * view.dpr),
      Math.round(view.plot.y * view.dpr),
      Math.round(view.plot.w * view.dpr),
      Math.round(view.plot.h * view.dpr),
    ).data;
    let fallback = 0;
    for (let i = 0; i < plotScan.length; i += 4) {
      if (plotScan[i] >= 60 && plotScan[i] <= 100 &&
          plotScan[i+1] >= 100 && plotScan[i+1] <= 140 &&
          plotScan[i+2] >= 140 && plotScan[i+2] <= 190 &&
          plotScan[i+3] > 100) fallback++;
    }
    document.body.setAttribute("data-xy-rgba-probe", JSON.stringify({
      colorMode, hasRgba, rgbaLen, colorIsUndef, hasX, hasY,
      authored, counts, fallback,
    }));
  } catch (err) {
    document.body.setAttribute(
      "data-xy-rgba-probe-error",
      String((err && err.stack) || err),
    );
  }
}, 0);
</script>
"""


def _scatter_chart():
    return xy.scatter_chart(
        xy.scatter(
            [0.0, 1.0, 0.0, 1.0],
            [0.0, 0.0, 1.0, 1.0],
            color=_COLORS,
            size=40,
            _marker_glyph="\u25cf",
        ),
        xy.x_axis(),
        xy.y_axis(),
        width=400,
        height=400,
    )


def test_standalone_direct_rgba_authored_markers_paint_per_point(tmp_path: Path) -> None:
    chromium = find_chromium()
    if chromium is None:
        pytest.skip("Chromium unavailable")

    document = probe_document(_scatter_chart(), _PROBE)
    result = run_browser_probe(
        chromium,
        document,
        tmp_path / "direct_rgba_standalone.html",
        "data-xy-rgba-probe",
        label="direct_rgba standalone authored-marker probe",
    )

    assert result["colorMode"] == 3, "scatter should be direct_rgba (colorMode 3)"
    assert result["hasRgba"] is True, "g._cpu.rgba must survive renderStandalone hydration"
    assert result["rgbaLen"] == 16, f"g._cpu.rgba should be length 16, got {result['rgbaLen']}"
    assert result["colorIsUndef"] is True, (
        "g._cpu.color must NOT hold the RGBA column for direct_rgba"
    )
    assert result["hasX"] is True and result["hasY"] is True, "x/y CPU views must be hydrated"

    # _authoredScatterRgba returns straight-alpha normalized RGBA per point.
    assert result["authored"] == [
        [255, 0, 0, 255],
        [0, 255, 0, 255],
        [0, 0, 255, 255],
        [255, 255, 0, 255],
    ], f"per-point authored-marker colors wrong: {result['authored']}"

    # Each marker paints on the Canvas2D overlay — the 4 colored fills must be
    # present; without g._cpu.rgba every marker falls back to the flat trace
    # color and produces zero distinct lit markers.
    assert all(c > 0 for c in result["counts"]), (
        f"some authored marker did not paint: {result['counts']}"
    )
    assert result["fallback"] == 0, (
        f"fallback steel-blue pixels present in plot: {result['fallback']}"
    )
