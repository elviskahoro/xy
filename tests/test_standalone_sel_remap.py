"""Standalone (kernel-less) box/lasso selection must remap through `_visInv`.

After a legend category toggle, `_filterScatterRows` (`js/src/50_chartview.ts`)
gathers the scatter vertex buffers into visible (`_visMap`) order, shrinks `g.n`
to the visible count, and builds `g._visInv` so shipped selection indices map to
their filtered vertex positions. The selection mask is a per-vertex attribute
consumed by `gl.drawArrays(..., g.n)`, so its index space MUST be the current
(gathered) vertex order, not the shipped order.

The kernel-connected path (`_applySelectionBuffers` in `js/src/54_kernel.ts`)
already remaps through `_visInv`. The standalone paths `_selectLocal` /
`_selectLocalPolygon` (`js/src/53_interaction.ts`) historically did not — they
looped `i < g.n` indexing the full shipped-order `_cpu` columns and uploaded the
raw mask. After a legend category hide this selected the wrong visible points,
never selected visible rows at shipped positions `>= g.n`, and reported a wrong
`_selectionCount`.

These probes drive a standalone `to_html` page in headless Chromium, toggle a
legend category, then read back the uploaded `g.selBuf` via `getBufferSubData`
after box/lasso selections and assert the per-vertex mask and count are correct.
Browser probes skip (never fail) without Chromium, like the repo's others.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

from conftest import probe_document, run_browser_probe

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

import xy  # noqa: E402
from xy.export import find_chromium  # noqa: E402


# Six interleaved categorical rows: 0=A(0,0), 1=B(1,0), 2=A(2,0),
# 3=B(3,0), 4=A(4,0), 5=B(5,0) — the exact dataset from the bug report.
def _interleaved_chart():
    codes = np.array(["A", "B", "A", "B", "A", "B"])
    data = {"x": np.arange(6.0), "y": np.zeros(6)}
    return xy.scatter_chart(
        xy.scatter("x", "y", data=data, color=codes),
        xy.legend(),
        width=520,
        height=340,
    )


# A single categorical trace with contiguous A-then-B blocks; after hiding B
# the visible rows form an identity-prefix of shipped order, so the OLD code
# happened to produce a correct mask. This control guards against a regression
# where the remap logic breaks the no-filter / identity-prefix case.
def _block_chart():
    codes = np.array(["A", "A", "A", "B", "B", "B"])
    data = {"x": np.arange(6.0), "y": np.zeros(6)}
    return xy.scatter_chart(
        xy.scatter("x", "y", data=data, color=codes),
        xy.legend(),
        width=520,
        height=340,
    )


_SEL_REMAP_PROBE = """
<script>
(async () => {
  const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
  try {
    const view = window.__fcProbeView;
    if (!view) throw new Error("no probe view captured");
    view._drawNow();
    view._raf = null;
    const sent = [];
    // Standalone means comm === null; keep it that way so _sendSelect routes
    // to _selectLocal, but capture legend_toggle wire messages by stubbing
    // only for the legend click and restoring null before brushing.
    const captureComm = { send: (m) => sent.push(m) };
    let rows = [];
    for (let i = 0; i < 200; i++) {
      rows = [...document.querySelectorAll('[data-xy-slot="legend_item"]')];
      if (rows.length >= 2) break;
      await sleep(25);
    }
    if (rows.length < 2) throw new Error(`expected 2 legend rows, got ${rows.length}`);
    const byName = (name) => rows.find((row) => row.textContent === name);
    const click = (row) => row.dispatchEvent(new MouseEvent("click"));

    const cat = view.gpuTraces.find((g) => g.trace.kind === "scatter");
    if (!cat) throw new Error("no scatter trace");
    const fullN = cat.n;

    // GL mask readback — the canonical helper from test_view_state_client.py.
    const selectionMask = (trace) => {
      const n = trace.n;
      if (!trace.selBuf) return new Array(n).fill(0);
      const mask = new Float32Array(n);
      view.gl.bindBuffer(view.gl.ARRAY_BUFFER, trace.selBuf);
      view.gl.getBufferSubData(view.gl.ARRAY_BUFFER, 0, mask);
      return [...mask];
    };

    // Capture the dispatched `xy:select` event's total (guarantee 9).
    let lastSelectTotal = null;
    view.root.addEventListener("xy:select", (e) => {
      lastSelectTotal = (e.detail && e.detail.total !== undefined)
        ? e.detail.total : null;
    });

    // --- Hide category B: visMap=[0,2,4], visInv=[0,-1,1,-1,2,-1], n=3 ---
    view.comm = captureComm;
    click(byName("B"));
    view.comm = null;
    const afterHideN = cat.n;
    const visMap = cat._visMap ? [...cat._visMap] : null;
    const visInv = cat._visInv ? [...cat._visInv] : null;
    let visibleCodeOk = true;
    for (let j = 0; j < cat.n; j++) {
      if (cat._cpu.color[cat._visMap[j]] !== 0) visibleCodeOk = false;
    }

    // Box-select ONLY the position of visible row 4 (x=4). Expected: mask[2]=1
    // because visInv[4]=2, count=1.
    view._selectLocal(3.5, 4.5, -1, 1, { dispatch: false });
    const boxRow4 = { mask: selectionMask(cat), count: view._selectionCount };

    // Box-select ONLY the position of HIDDEN row 1 (x=1). Expected: nothing,
    // because visInv[1]=-1. count=0.
    view._selectLocal(0.5, 1.5, -1, 1, { dispatch: false });
    const boxHiddenRow1 = { mask: selectionMask(cat), count: view._selectionCount };

    // Box-select ONLY visible row 2 (x=2). Expected: mask[1]=1 (visInv[2]=1).
    view._selectLocal(1.5, 2.5, -1, 1, { dispatch: false });
    const boxRow2 = { mask: selectionMask(cat), count: view._selectionCount };

    // Full-plot box select over the filtered chart. Expected: all visible
    // rows selected, mask=[1,1,1], count=3.
    view._selectLocal(-1, 6, -1, 1, { dispatch: false });
    const boxAllVisible = { mask: selectionMask(cat), count: view._selectionCount };

    // Dispatched `select` event total must equal the corrected count.
    lastSelectTotal = null;
    view._selectLocal(-1, 6, -1, 1, { dispatch: true });
    const boxEventTotal = lastSelectTotal;
    const boxAllVisibleEvents = { mask: selectionMask(cat), count: view._selectionCount };

    // Lasso enclosing ONLY visible row 4's position. Expected: mask[2]=1.
    view._selectLocalPolygon(
      [[3.5, -0.5], [4.5, -0.5], [4.5, 0.5], [3.5, 0.5]],
      { dispatch: false },
    );
    const lassoRow4 = { mask: selectionMask(cat), count: view._selectionCount };

    // Lasso enclosing ONLY the hidden row 1's position. Expected: nothing.
    view._selectLocalPolygon(
      [[0.5, -0.5], [1.5, -0.5], [1.5, 0.5], [0.5, 0.5]],
      { dispatch: false },
    );
    const lassoHiddenRow1 = { mask: selectionMask(cat), count: view._selectionCount };

    // Dispatched `select` (polygon) event total must match.
    lastSelectTotal = null;
    view._selectLocalPolygon(
      [[3.5, -0.5], [4.5, -0.5], [4.5, 0.5], [3.5, 0.5]],
      { dispatch: true },
    );
    const lassoEventTotal = lastSelectTotal;

    // --- Untoggle B: full buffers restored, _visInv deleted ---
    view.comm = captureComm;
    click(byName("B"));
    view.comm = null;
    const restoredN = cat.n;
    const visInvCleared = !cat._visInv;
    view._selectLocal(-1, 6, -1, 1, { dispatch: false });
    const boxFullAfterRestore = { mask: selectionMask(cat), count: view._selectionCount };
    const restoreAllOnes =
      boxFullAfterRestore.mask.length === restoredN
      && boxFullAfterRestore.mask.every((v) => v === 1);

    document.body.setAttribute(
      "data-xy-sel-remap",
      JSON.stringify({
        fullN, afterHideN, visMap, visInv, visibleCodeOk,
        boxRow4, boxHiddenRow1, boxRow2, boxAllVisible, boxAllVisibleEvents,
        boxEventTotal,
        lassoRow4, lassoHiddenRow1, lassoEventTotal,
        restoredN, visInvCleared, boxFullAfterRestore, restoreAllOnes,
        sent,
      })
    );
  } catch (err) {
    document.body.setAttribute(
      "data-xy-sel-remap-error",
      String((err && err.stack) || err)
    );
  }
})();
</script>
"""


def test_standalone_selection_remasks_after_legend_category_toggle() -> None:
    chromium = find_chromium()
    if not chromium:
        pytest.skip("no chromium available for the standalone sel-remap probe")

    chart = _interleaved_chart()
    document = probe_document(chart, _SEL_REMAP_PROBE)
    with tempfile.TemporaryDirectory() as td:
        page = Path(td) / "sel_remap.html"
        payload = run_browser_probe(
            chromium,
            document,
            page,
            "data-xy-sel-remap",
            label="standalone selection remap probe",
        )

    # Filter established the expected gathered state.
    assert payload["fullN"] == 6, payload
    assert payload["afterHideN"] == 3, payload
    assert list(payload["visMap"]) == [0, 2, 4], payload
    assert list(payload["visInv"]) == [0, -1, 1, -1, 2, -1], payload
    assert payload["visibleCodeOk"] is True, payload

    # Guarantee 1: a single visible row's box selects exactly its filtered vertex.
    assert list(payload["boxRow4"]["mask"]) == [0, 0, 1], payload
    assert payload["boxRow4"]["count"] == 1, payload

    # Guarantee 2: a box enclosing only a HIDDEN row selects nothing.
    assert list(payload["boxHiddenRow1"]["mask"]) == [0, 0, 0], payload
    assert payload["boxHiddenRow1"]["count"] == 0, payload

    # Guarantee 3: a different single visible row lands on its own filtered vertex.
    assert list(payload["boxRow2"]["mask"]) == [0, 1, 0], payload
    assert payload["boxRow2"]["count"] == 1, payload

    # Guarantee 5: full-plot box select highlights every visible point.
    assert list(payload["boxAllVisible"]["mask"]) == [1, 1, 1], payload
    assert payload["boxAllVisible"]["count"] == 3, payload
    assert list(payload["boxAllVisibleEvents"]["mask"]) == [1, 1, 1], payload
    assert payload["boxAllVisibleEvents"]["count"] == 3, payload

    # Guarantee 9: dispatched `select` event total matches the corrected count.
    assert payload["boxEventTotal"] == 3, payload

    # Guarantee 4: lasso path behaves identically to the box path.
    assert list(payload["lassoRow4"]["mask"]) == [0, 0, 1], payload
    assert payload["lassoRow4"]["count"] == 1, payload
    assert list(payload["lassoHiddenRow1"]["mask"]) == [0, 0, 0], payload
    assert payload["lassoHiddenRow1"]["count"] == 0, payload
    assert payload["lassoEventTotal"] == 1, payload

    # Guarantee 6: untoggle restores the full buffers and clears _visInv, and a
    # subsequent full-plot box select selects every shipped row.
    assert payload["restoredN"] == 6, payload
    assert payload["visInvCleared"] is True, payload
    assert payload["boxFullAfterRestore"]["count"] == 6, payload
    assert payload["restoreAllOnes"] is True, payload

    # The legend_toggle wire messages were sent on click (B hidden, then restored).
    kinds = [(m.get("type"), m.get("category"), m.get("hidden")) for m in payload["sent"]]
    assert kinds == [("legend_toggle", 1, True), ("legend_toggle", 1, False)], payload["sent"]


# A small in-process JS-source contract check so the suite still has teeth
# when no Chromium binary is available: the standalone mask builders must
# iterate the full shipped-order CPU rows and remap through `_visInv`. This
# mirrors the source-string guards used elsewhere (e.g. tests/test_funnel.py).
def test_select_local_source_remaps_through_vis_inv() -> None:
    src = (ROOT / "js" / "src" / "53_interaction.ts").read_text(encoding="utf-8")

    # Both standalone builders must compute `full = Math.min(cx.length, cy.length)`
    # and iterate over it (not `g.n`), so visible rows at shipped positions >= g.n
    # after a category filter are still examined.
    assert src.count("Math.min(cx.length, cy.length)") >= 2, src
    # Both must read `g._visInv` and remap each shipped row to its filtered vertex.
    assert src.count("const inv = g._visInv;") >= 2, src
    assert src.count("const j = inv ? (i < inv.length ? inv[i] : -1) : i;") >= 2, src
    # Both must guard the mask write on the filtered vertex range.
    assert src.count("if (j >= 0 && j < g.n) { mask[j] = 1;") >= 2, src
    # The shipped-index mask write must be gone from the whole file: every mask
    # write now goes through the remapped `j`, never the loop index `i`.
    assert "mask[i] = 1" not in src, src
    # The mask-building loop must not regress to `i < g.n`; both builders now
    # iterate the full shipped-order CPU rows.
    assert "for (let i = 0; i < g.n; i++) {\n        const dx = cx[i]" not in src, src
    assert "for (let i = 0; i < g.n; i++) {\n        const x = cx[i]" not in src, src


_CONTROL_PROBE = """
<script>
(async () => {
  const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
  try {
    const view = window.__fcProbeView;
    if (!view) throw new Error("no probe view captured");
    view._drawNow();
    view._raf = null;
    let rows = [];
    for (let i = 0; i < 200; i++) {
      rows = [...document.querySelectorAll('[data-xy-slot="legend_item"]')];
      if (rows.length >= 2) break;
      await sleep(25);
    }
    if (rows.length < 2) throw new Error(`expected 2 legend rows, got ${rows.length}`);

    const cat = view.gpuTraces.find((g) => g.trace.kind === "scatter");
    if (!cat) throw new Error("no scatter trace");
    const fullN = cat.n;
    const noFilter = !cat._visInv;

    const selectionMask = (trace) => {
      const n = trace.n;
      if (!trace.selBuf) return new Array(n).fill(0);
      const mask = new Float32Array(n);
      view.gl.bindBuffer(view.gl.ARRAY_BUFFER, trace.selBuf);
      view.gl.getBufferSubData(view.gl.ARRAY_BUFFER, 0, mask);
      return [...mask];
    };

    // Never toggled: full-plot box select over the unfiltered identity buffers.
    view._selectLocal(-1, 6, -1, 1, { dispatch: false });
    const boxFull = { mask: selectionMask(cat), count: view._selectionCount };

    // Hide B (the trailing block) — the visible prefix is identity-ordered, so
    // the OLD code happened to be correct here too. After the fix, the remap
    // path (inv[i] === i for the visible prefix) must still produce [1,1,1].
    const byName = (name) => rows.find((row) => row.textContent === name);
    const click = (row) => row.dispatchEvent(new MouseEvent("click"));
    click(byName("B"));
    const afterHideN = cat.n;
    view._selectLocal(-1, 6, -1, 1, { dispatch: false });
    const boxHidden = { mask: selectionMask(cat), count: view._selectionCount };

    // Box enclosing only the first visible row (x=0) after the hide.
    view._selectLocal(-0.5, 0.5, -1, 1, { dispatch: false });
    const boxFirst = { mask: selectionMask(cat), count: view._selectionCount };

    document.body.setAttribute(
      "data-xy-sel-control",
      JSON.stringify({ fullN, noFilter, boxFull, afterHideN, boxHidden, boxFirst })
    );
  } catch (err) {
    document.body.setAttribute(
      "data-xy-sel-control-error",
      String((err && err.stack) || err)
    );
  }
})();
</script>
"""


def test_standalone_selection_identity_and_block_order_control() -> None:
    chromium = find_chromium()
    if not chromium:
        pytest.skip("no chromium available for the standalone sel-control probe")

    chart = _block_chart()
    document = probe_document(chart, _CONTROL_PROBE)
    with tempfile.TemporaryDirectory() as td:
        page = Path(td) / "sel_control.html"
        payload = run_browser_probe(
            chromium,
            document,
            page,
            "data-xy-sel-control",
            label="standalone selection control probe",
        )

    # Guarantee 7: unfiltered identity path is unchanged — full-plot select lights
    # every shipped row.
    assert payload["fullN"] == 6, payload
    assert payload["noFilter"] is True, payload
    assert list(payload["boxFull"]["mask"]) == [1, 1, 1, 1, 1, 1], payload
    assert payload["boxFull"]["count"] == 6, payload

    # Block-ordered hide (identity prefix) — the fix's identity remap must still
    # produce the correct mask and count, guarding against a regression that
    # only handles interleaved orders.
    assert payload["afterHideN"] == 3, payload
    assert list(payload["boxHidden"]["mask"]) == [1, 1, 1], payload
    assert payload["boxHidden"]["count"] == 3, payload
    assert list(payload["boxFirst"]["mask"]) == [1, 0, 0], payload
    assert payload["boxFirst"]["count"] == 1, payload
