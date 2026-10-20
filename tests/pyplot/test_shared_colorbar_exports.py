"""The figure-level shared colorbar (``fig.colorbar(mappable, ax=[ax1, ...])``)
must reach every static export — SVG, HTML, and PNG — for both the single-axes
``ax=[ax]`` form and multi-panel ``ax=axs`` form, mirroring the bare 16-px
gradient strip ``stitch_png``'s grid branch already paints.

Regressions for the bug where the shared colorbar was plumbed only into
``_to_png -> stitch_png`` and was therefore silently dropped from SVG/HTML
(and from multi-panel PNG, since ``stitch_png``'s absolute branch returned
before the colorbar block).
"""

from __future__ import annotations

import io
import struct
from xml.etree import ElementTree

import numpy as np
import pytest

import xy.pyplot as plt
from xy.pyplot._grid import _SHARED_COLORBAR_BAND, _SHARED_COLORBAR_PAD, _SHARED_COLORBAR_STRIP


def _png_dimensions(data: bytes) -> tuple[int, int]:
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    ihdr = data.index(b"IHDR")
    width, height = struct.unpack(">II", data[ihdr + 4 : ihdr + 12])
    return int(width), int(height)


def _svg_root_dimensions(data: bytes) -> tuple[int, int]:
    root = ElementTree.fromstring(data)
    return int(root.attrib["width"]), int(root.attrib["height"])


def _colorbar_rect(svg: bytes) -> ElementTree.Element | None:
    """The shared colorbar <rect> element in the composed SVG, or None."""
    root = ElementTree.fromstring(svg)
    for node in root.iter():
        if node.tag.endswith("rect") and node.attrib.get("data-xy-shared-colorbar") == "true":
            return node
    return None


@pytest.fixture(autouse=True)
def _clean_state() -> None:
    plt.close("all")
    plt.rcParams.reset()
    yield
    plt.close("all")
    plt.rcParams.reset()


def test_colorbar_with_axes_list_stores_shared_colorbar_state() -> None:
    """``ax=[ax]`` (the matplotlib shared-colorbar idiom) sets figure-level state."""
    fig, ax = plt.subplots()
    image = ax.imshow(np.arange(25).reshape(5, 5))

    fig.colorbar(image, ax=[ax])

    assert ax._colorbar is None
    assert fig._shared_colorbar is not None
    assert fig._shared_colorbar["colormap"] == "viridis"
    assert fig._shared_colorbar["orientation"] == "vertical"
    assert fig._shared_colorbar["domain"] == [0.0, 24.0]


def test_single_axes_shared_colorbar_reaches_svg_html_and_png() -> None:
    """A single-axes figure with `ax=[ax]` exports the colorbar in every format."""
    cmap = "viridis"
    fig, ax = plt.subplots(figsize=(6.4, 4.8), dpi=100)
    image = ax.imshow(np.arange(25).reshape(5, 5), cmap=cmap)
    fig.colorbar(image, ax=[ax])

    # SVG: the composed document grows by `_SHARED_COLORBAR_BAND` and emits a
    # `<linearGradient>` plus a `<rect data-xy-shared-colorbar>` strip.
    svg_buf = io.BytesIO()
    fig.savefig(svg_buf, format="svg")
    svg = svg_buf.getvalue()
    assert b"data-xy-shared-colorbar" in svg
    assert f"xy-shared-colorbar-{cmap}".encode() in svg
    assert _svg_root_dimensions(svg) == (640, 480 + _SHARED_COLORBAR_BAND)
    rect = _colorbar_rect(svg)
    assert rect is not None
    assert int(rect.attrib["width"]) == round(640 * 0.85) - round(640 * 0.15)
    assert int(rect.attrib["height"]) == _SHARED_COLORBAR_STRIP
    # Strip lives in the band reserved below the axes panel area.
    assert int(rect.attrib["y"]) == 480 + _SHARED_COLORBAR_PAD

    # HTML: composed document includes the gradient strip.
    html_buf = io.BytesIO()
    fig.savefig(html_buf, format="html")
    html_out = html_buf.getvalue()
    assert b"data-xy-shared-colorbar" in html_out
    assert b"linear-gradient(to right" in html_out

    # PNG: the canvas grows by the same band and the strip is painted.
    png_buf = io.BytesIO()
    fig.savefig(png_buf, format="png")
    png = png_buf.getvalue()
    assert png.startswith(b"\x89PNG\r\n\x1a\n")
    assert _png_dimensions(png) == (640, 480 + _SHARED_COLORBAR_BAND)


def test_single_axes_shared_colorbar_appears_in_notebook_view() -> None:
    """The interactive notebook `_repr_html_` includes the shared colorbar."""
    fig, ax = plt.subplots(figsize=(6.4, 4.8), dpi=100)
    fig.colorbar(ax.imshow(np.arange(25).reshape(5, 5)), ax=[ax])

    doc, width, height = fig._to_notebook_html()

    # The fast single-axes tight-bbox path is bypassed because of the shared
    # colorbar; the CSS-grid size now includes the colorbar band.
    assert b"xy-shared-colorbar" in doc.encode()
    # 480px panel + 8px grid padding + 52px colorbar band == 540.  Without the
    # colorbar band the height would be 488 — assert it grew by the band size.
    assert height == 480 + 8 + _SHARED_COLORBAR_BAND


def test_multi_panel_shared_colorbar_reaches_svg_html_and_png() -> None:
    """The matplotlib idiom `fig.colorbar(im, ax=axs)` across panels exports in
    every format — the `stitch_png` absolute-layout regression is also covered."""
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 4.8), dpi=100)
    for ax in axes:
        ax.imshow(np.arange(25).reshape(5, 5), cmap="plasma")
    fig.colorbar(axes[0].images[0], ax=axes)

    svg_buf = io.BytesIO()
    fig.savefig(svg_buf, format="svg")
    svg = svg_buf.getvalue()
    assert b"data-xy-shared-colorbar" in svg
    assert b"xy-shared-colorbar-plasma" in svg
    # Multi-panel takes the absolute layout path, so the SVG canvas itself
    # grows by the colorbar band (panels stay at their gridspec rects).
    assert _svg_root_dimensions(svg) == (640, 480 + _SHARED_COLORBAR_BAND)
    rect = _colorbar_rect(svg)
    assert rect is not None
    assert int(rect.attrib["y"]) == 480 + _SHARED_COLORBAR_PAD

    html_buf = io.BytesIO()
    fig.savefig(html_buf, format="html")
    html_out = html_buf.getvalue()
    assert b"data-xy-shared-colorbar" in html_out
    # Absolute HTML places the strip with position:absolute at canvas_height+12.
    assert b"position:absolute" in html_out

    # The absolute-branch PNG previously dropped the colorbar entirely; the
    # canvas now grows by the band too.
    png_buf = io.BytesIO()
    fig.savefig(png_buf, format="png")
    assert _png_dimensions(png_buf.getvalue()) == (640, 480 + _SHARED_COLORBAR_BAND)


def test_shared_colorbar_with_suptitle_does_not_drop_either() -> None:
    """Suptitle plus a shared colorbar render together in SVG and HTML."""
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 4.8), dpi=100)
    for ax in axes:
        ax.imshow(np.arange(25).reshape(5, 5), cmap="viridis")
    fig.colorbar(axes[0].images[0], ax=axes)
    fig.suptitle("Two panels, one colorbar")

    svg_buf = io.BytesIO()
    fig.savefig(svg_buf, format="svg")
    svg = svg_buf.getvalue()
    assert b"Two panels, one colorbar" in svg
    assert b"data-xy-shared-colorbar" in svg

    html_buf = io.BytesIO()
    fig.savefig(html_buf, format="html")
    html_out = html_buf.getvalue()
    assert b"Two panels, one colorbar" in html_out
    assert b"data-xy-shared-colorbar" in html_out


def test_shared_colorbar_with_figure_legend_renders_both() -> None:
    """A figure-level legend and a shared colorbar both render in the SVG."""
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 4.8), dpi=100)
    for ax in axes:
        ax.imshow(np.arange(25).reshape(5, 5), cmap="plasma")
        ax.plot([0, 1], [0, 1], label="line")
    fig.colorbar(axes[0].images[0], ax=axes)
    fig.legend()

    svg_buf = io.BytesIO()
    fig.savefig(svg_buf, format="svg")
    svg = svg_buf.getvalue()
    assert b"xy-figure-legend" in svg
    assert b"data-xy-shared-colorbar" in svg


def test_add_axes_free_form_with_shared_colorbar_renders_in_every_format() -> None:
    """Free-form `add_axes` panels (the absolute path) honor the colorbar."""
    fig = plt.figure(figsize=(6.4, 4.8), dpi=100)
    ax1 = fig.add_axes([0.1, 0.15, 0.35, 0.7])
    ax2 = fig.add_axes([0.55, 0.15, 0.35, 0.7])
    ax1.imshow(np.arange(25).reshape(5, 5), cmap="cividis")
    ax2.imshow(np.arange(25).reshape(5, 5), cmap="cividis")
    fig.colorbar(ax1.images[0], ax=[ax1, ax2])

    svg_buf = io.BytesIO()
    fig.savefig(svg_buf, format="svg")
    svg = svg_buf.getvalue()
    assert b"data-xy-shared-colorbar" in svg
    assert b"xy-shared-colorbar-cividis" in svg
    assert _svg_root_dimensions(svg) == (640, 480 + _SHARED_COLORBAR_BAND)

    png_buf = io.BytesIO()
    fig.savefig(png_buf, format="png")
    assert _png_dimensions(png_buf.getvalue()) == (640, 480 + _SHARED_COLORBAR_BAND)

    html_buf = io.BytesIO()
    fig.savefig(html_buf, format="html")
    assert b"data-xy-shared-colorbar" in html_buf.getvalue()


def test_tight_layout_with_shared_colorbar_renders() -> None:
    """`layout='tight'` plus a shared colorbar still produces the strip."""
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 4.8), dpi=100, layout="tight")
    for ax in axes:
        ax.imshow(np.arange(25).reshape(5, 5), cmap="viridis")
    fig.colorbar(axes[0].images[0], ax=axes)

    svg_buf = io.BytesIO()
    fig.savefig(svg_buf, format="svg")
    assert b"data-xy-shared-colorbar" in svg_buf.getvalue()


def test_bbox_inches_tight_with_shared_colorbar_keeps_strip_in_png() -> None:
    """`bbox_inches='tight'` crops margins but must retain the gradient strip."""
    fig, ax = plt.subplots(figsize=(6.4, 4.8), dpi=100)
    fig.colorbar(ax.imshow(np.arange(25).reshape(5, 5), cmap="viridis"), ax=[ax])

    png_buf = io.BytesIO()
    fig.savefig(png_buf, format="png", bbox_inches="tight")
    pixels = np.asarray(plt.imread(io.BytesIO(png_buf.getvalue())))
    # The viridis ramp paints >100 distinct colors; a stripped PNG would only
    # carry the axes colors (well under 100 for a 5x5 imshow).
    assert len(np.unique(pixels.reshape(-1, pixels.shape[-1]), axis=0)) > 100


def test_per_axes_colorbar_path_unchanged_by_the_shared_colorbar_fix() -> None:
    """`ax=ax` (per-axes colorbar) must continue to render and not be diverted."""
    fig, ax = plt.subplots(figsize=(6.4, 4.8), dpi=100)
    image = ax.imshow(np.arange(25).reshape(5, 5), cmap="viridis")
    fig.colorbar(image, ax=ax)

    assert ax._colorbar is not None
    assert fig._shared_colorbar is None

    # Per-axes: the SVG dimensions stay at figsize (no shared-colorbar band)
    # and the per-axes colorbar's gradient id is present.
    svg_buf = io.BytesIO()
    fig.savefig(svg_buf, format="svg")
    svg = svg_buf.getvalue()
    assert b"xy-colorbar-viridis" in svg
    assert b"data-xy-shared-colorbar" not in svg
    assert _svg_root_dimensions(svg) == (640, 480)

    png_buf = io.BytesIO()
    fig.savefig(png_buf, format="png")
    assert _png_dimensions(png_buf.getvalue()) == (640, 480)


def test_no_shared_colorbar_leaves_svg_html_png_at_figsize() -> None:
    """A plain figure (no colorbar) is unchanged across formats."""
    fig, ax = plt.subplots(figsize=(6.4, 4.8), dpi=100)
    ax.plot([0, 1], [0, 1])

    svg_buf = io.BytesIO()
    fig.savefig(svg_buf, format="svg")
    assert b"data-xy-shared-colorbar" not in svg_buf.getvalue()
    assert _svg_root_dimensions(svg_buf.getvalue()) == (640, 480)

    png_buf = io.BytesIO()
    fig.savefig(png_buf, format="png")
    assert _png_dimensions(png_buf.getvalue()) == (640, 480)


def test_clear_resets_shared_colorbar_so_exports_drop_the_strip() -> None:
    """`Figure.clear()` resets the shared colorbar; later exports omit it."""
    fig, ax = plt.subplots(figsize=(6.4, 4.8), dpi=100)
    fig.colorbar(ax.imshow(np.arange(25).reshape(5, 5)), ax=[ax])
    assert fig._shared_colorbar is not None

    fig.clear()
    assert fig._shared_colorbar is None

    # Re-plotting without re-adding the colorbar must produce a strip-less SVG.
    ax = fig.add_subplot(111)
    ax.plot([0, 1], [0, 1])
    svg_buf = io.BytesIO()
    fig.savefig(svg_buf, format="svg")
    assert b"data-xy-shared-colorbar" not in svg_buf.getvalue()


def test_shared_colorbar_set_label_stored_in_figure_state() -> None:
    """`Colorbar.set_label` on the shared colorbar updates the figure state
    that the export paths read."""
    fig, ax = plt.subplots(figsize=(6.4, 4.8), dpi=100)
    cbar = fig.colorbar(ax.imshow(np.arange(25).reshape(5, 5)), ax=[ax])
    cbar.set_label("intensity (W/m^2)")

    assert fig._shared_colorbar["label"] == "intensity (W/m^2)"


def test_shared_colorbar_strip_geometry_matches_stitch_png_grid_branch() -> None:
    """SVG, PNG, and the `stitch_png` grid-branch constant all agree on the
    16-px strip inside the 52-px band, placed at 12 px past the panel area."""
    fig, ax = plt.subplots(figsize=(6.4, 4.8), dpi=100)
    fig.colorbar(ax.imshow(np.arange(25).reshape(5, 5)), ax=[ax])

    # The exported SVG strip geometry must match the documented constants.
    svg_buf = io.BytesIO()
    fig.savefig(svg_buf, format="svg")
    rect = _colorbar_rect(svg_buf.getvalue())
    assert rect is not None
    assert int(rect.attrib["height"]) == _SHARED_COLORBAR_STRIP
    assert int(rect.attrib["y"]) == 480 + _SHARED_COLORBAR_PAD
    # Cross-check the constants against `stitch_png`'s documented behavior.
    assert (_SHARED_COLORBAR_BAND, _SHARED_COLORBAR_STRIP, _SHARED_COLORBAR_PAD) == (52, 16, 12)


def test_shared_colorbar_does_not_inject_chart_payload_colorbar_key() -> None:
    """A shared colorbar is figure chrome, not a per-axes chart `colorbar`
    payload; the composed HTML must not declare a `colorbar` key on any panel."""
    fig, ax = plt.subplots(figsize=(6.4, 4.8), dpi=100)
    fig.colorbar(ax.imshow(np.arange(25).reshape(5, 5)), ax=[ax])

    html_buf = io.BytesIO()
    fig.savefig(html_buf, format="html")
    html_out = html_buf.getvalue()
    assert b"data-xy-shared-colorbar" in html_out
    # The shared colorbar is rendered as a single figure-level gradient div,
    # never duplicated as a panel `colorbar` payload spec.
    assert html_out.count(b'"colorbar":') == 0


def test_absolute_mode_shared_colorbar_keeps_suplabels_in_panel_area() -> None:
    """Figure-fraction suplabels in absolute mode stay anchored to the panel
    area (canvas_size[1]) when the shared colorbar grows the grid by 52 px.

    Without the panel-area wrapper a `top:99%` supxlabel would resolve against
    the grown grid height and land inside the colorbar band rather than at the
    bottom of the original figure, where matplotlib draws suplabels.
    """
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 4.8), dpi=100)
    for ax in axes:
        ax.imshow(np.arange(25).reshape(5, 5), cmap="viridis")
    fig.colorbar(axes[0].images[0], ax=axes)
    fig.supxlabel("bottom label")
    fig.supylabel("left label")

    html_buf = io.BytesIO()
    fig.savefig(html_buf, format="html")
    html_out = html_buf.getvalue().decode("utf-8")

    # The panel-area wrapper sized to canvas_size[1] (480 px) is present, and
    # the colored strip lives outside the wrapper at the band top (492 px).
    wrapper_marker = "<div style='position:absolute;left:0;top:0;width:640px;height:480px'>"
    assert wrapper_marker in html_out
    wrapper_start = html_out.find(wrapper_marker)
    label_div_marker = "<div class='xy-figure-label'"
    label_pos = html_out.find(label_div_marker, wrapper_start)
    assert label_pos > wrapper_start
    # The colorbar div sits outside the wrapper (its top is 480 + 12 = 492 px).
    colorbar_top_marker = "position:absolute;left:96px;top:492px"
    assert colorbar_top_marker in html_out
    colorbar_pos = html_out.find(colorbar_top_marker)
    assert colorbar_pos > wrapper_start


def test_absolute_mode_shared_colorbar_keeps_suplabels_in_panel_area_svg() -> None:
    """SVG figure-fraction suplabels anchor to the panel area, not the
    colorbar-grown canvas, mirroring matplotlib's figure coordinates."""
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 4.8), dpi=100)
    for ax in axes:
        ax.imshow(np.arange(25).reshape(5, 5), cmap="viridis")
    fig.colorbar(axes[0].images[0], ax=axes)
    fig.supxlabel("bottom label")

    svg_buf = io.BytesIO()
    fig.savefig(svg_buf, format="svg")
    root = ElementTree.fromstring(svg_buf.getvalue())

    # The shared colorbar strip is at y = canvas_size[1] + 12 = 492 px, the
    # bottom of the panel area plus the band's top padding.
    colorbar_rects = [
        node
        for node in root.iter()
        if node.tag.endswith("rect") and node.attrib.get("data-xy-shared-colorbar") == "true"
    ]
    assert colorbar_rects
    assert int(colorbar_rects[0].attrib["y"]) == 480 + _SHARED_COLORBAR_PAD

    # The supxlabel is at figure-fraction y=0.01 of the panel area (480 px),
    # i.e. near the bottom of the original figure — NOT inside the colorbar
    # band (which begins at y=480).  matplotlib's intent is `y * canvas_size[1]`;
    # any extra `_SHARED_COLORBAR_BAND` offset would be a regression.
    supxlabel = next(
        node
        for node in root.iter()
        if node.tag.endswith("text") and "".join(node.itertext()) == "bottom label"
    )
    label_y = float(supxlabel.attrib["y"])
    assert label_y < 480 + _SHARED_COLORBAR_PAD
    assert label_y > 460  # near the bottom of the panel area, not the band
